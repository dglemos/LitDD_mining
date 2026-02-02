import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa


def extract_score(item):
    if isinstance(item, dict):
        return item.get("score")
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        return item[1]
    try:
        if isinstance(item, pa.Scalar):
            return extract_score(item.as_py())
    except Exception:
        pass
    return np.nan


def parse_ids(s):
    if pd.isna(s):
        return []
    s = str(s).strip()
    if not s:
        return []
    if ";" in s:
        return [tok.strip() for tok in s.split(";") if tok.strip()]
    return [s]


def map_ids(ids, mapping):
    out = []
    for gid in ids:
        out.extend(mapping.get(gid, []))
    return out


def unique(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def load_parquets_from_dir(parquet_dir: str) -> pd.DataFrame:
    pattern = os.path.join(parquet_dir, "*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")
    frames = [pd.read_parquet(p) for p in files]
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet_dir",
        type=Path,
        required=True,
        help="Directory containing LLM output parquet files",
    )
    parser.add_argument(
        "--g2p_csv",
        type=Path,
        required=True,
        help="Path to ddg2p.csv",
    )
    parser.add_argument(
        "--gene2pubtator3",
        type=Path,
        required=True,
        help="Path to gene2pubtator3",
    )
    parser.add_argument(
        "--gene_info",
        type=Path,
        required=True,
        help="Path to NCBI gene_info",
    )
    parser.add_argument(
        "--out_path",
        type=Path,
        default=Path("final_tiab_mappings.parquet"),
        help="Output parquet path",
    )
    args = parser.parse_args()

    all_df = load_parquets_from_dir(str(args.parquet_dir))
    all_df = all_df[all_df["llm_dis_map"].notna()]
    llm_final_df = all_df.reset_index(drop=True)

    scores_max = (
        llm_final_df["top5_cross"]
        .explode()
        .apply(extract_score)
        .groupby(level=0)
        .max()
        .reindex(llm_final_df.index, fill_value=np.nan)
    )

    filtered = llm_final_df.loc[
        llm_final_df["llm_dis_map"].notna()
        & (llm_final_df["llm_dis_map"].str.startswith("G2P"))
        & scores_max.ge(0.9).fillna(False)
    ]

    # make sure mappings are to true G2P IDs, i.e. no hallucinations
    g2p = pd.read_csv(args.g2p_csv)

    valid_ids = set(g2p["g2p id"])

    tmp = (
        filtered.assign(
            _idx=lambda df: df.index,
            id_list=lambda df: df["llm_dis_map"].str.split(";"),
        )
        .explode("id_list")
        .dropna(subset=["id_list"])
    )

    # Clean tokens: trim whitespace and any stray quotes in the source
    tmp["id_clean"] = tmp["id_list"].str.strip().str.strip("'\"")

    # Keep only valid, non-empty IDs
    tmp = tmp[tmp["id_clean"].isin(valid_ids) & tmp["id_clean"].ne("")]

    # Aggregate back without quotes, preserving order
    agg = tmp.groupby("_idx", sort=False)["id_clean"].apply(lambda s: ";".join(s))

    filtered_clean = filtered.loc[agg.index].copy()
    filtered_clean["llm_dis_map"] = agg

    filtered_clean["pmid"] = pd.to_numeric(filtered_clean["pmid"], errors="coerce").astype(
        "Int64"
    )

    # add pubtator gene mappings for tiab (uses GNorm2)
    cols = ["pmid", "type", "entity_id", "mention", "source"]

    pubtator_df = pd.read_csv(
        args.gene2pubtator3,
        sep="\t",
        header=None,
        names=cols,
        dtype={
            "pmid": "int64",
            "type": "category",
            "entity_id": "string",
            "mention": "string",
            "source": "category",
        },
        low_memory=False,
    )

    pubtator_df["entity_id"] = (
        pubtator_df["entity_id"].str.split(";").str[0].str.strip()
    )
    pubtator_df["entity_id"] = pd.to_numeric(
        pubtator_df["entity_id"], errors="coerce"
    ).astype("Int64")

    # get ncbi gene ids
    cols = [
        "#tax_id",
        "GeneID",
        "Symbol",
        "LocusTag",
        "Synonyms",
        "dbXrefs",
        "chromosome",
        "map_location",
        "description",
        "type_of_gene",
        "Symbol_from_nomenclature_authority",
        "Full_name_from_nomenclature_authority",
        "Nomenclature_status",
        "Other_designations",
        "Modification_date",
        "Feature_type",
    ]

    ncbi_gene_info = pd.read_csv(
        args.gene_info,
        sep="\t",
        engine="pyarrow",
        usecols=cols,
        na_values="-",
        keep_default_na=False,
        dtype_backend="pyarrow",
    )

    # Normalize first column name
    ncbi_gene_info.rename(columns=lambda c: c.lstrip("#"), inplace=True)

    # limit to homo sapiens
    ncbi_gene_info = ncbi_gene_info[ncbi_gene_info["tax_id"] == 9606]

    pubtator_df = pubtator_df[
        pubtator_df["entity_id"].isin(ncbi_gene_info["GeneID"])
    ]
    pubtator_df = pubtator_df.merge(
        ncbi_gene_info[["GeneID", "Symbol"]],
        how="left",
        left_on="entity_id",
        right_on="GeneID",
    )

    # Drop rows where Symbol is NaN
    tmp = pubtator_df.dropna(subset=["Symbol"])

    gene_per_pmid = (
        tmp.groupby("pmid")["Symbol"].unique().map(list).reset_index(name="gene_symbols")
    )

    filtered_clean_genes = filtered_clean.merge(
        gene_per_pmid, left_on="pmid", right_on="pmid", how="left"
    )

    g2p_clean = g2p.copy()
    for col in ["g2p id", "gene symbol", "disease name"]:
        if col in g2p_clean.columns:
            g2p_clean[col] = g2p_clean[col].astype(str).str.strip()

    id_to_genes = (
        g2p_clean.dropna(subset=["g2p id", "gene symbol"])
        .groupby("g2p id")["gene symbol"]
        .apply(list)
        .to_dict()
    )

    id_to_names = (
        g2p_clean.dropna(subset=["g2p id", "disease name"])
        .groupby("g2p id")["disease name"]
        .apply(list)
        .to_dict()
    )

    ids_col = filtered_clean_genes["llm_dis_map"].apply(parse_ids)

    filtered_clean_genes["dis_map_gene"] = ids_col.apply(
        lambda ids: map_ids(ids, id_to_genes)
    )
    filtered_clean_genes["dis_map_name"] = ids_col.apply(
        lambda ids: map_ids(ids, id_to_names)
    )

    filtered_clean_genes["dis_map_gene"] = filtered_clean_genes[
        "dis_map_gene"
    ].apply(unique)
    filtered_clean_genes["dis_map_name"] = filtered_clean_genes[
        "dis_map_name"
    ].apply(unique)

    dis_sets = filtered_clean_genes["dis_map_gene"].apply(
        lambda x: set(x) if isinstance(x, (list, tuple, set)) else set()
    )
    top_sets = filtered_clean_genes["gene_symbols"].apply(
        lambda x: set(x) if isinstance(x, (list, tuple, set)) else set()
    )

    filtered_clean_genes["dis_map_correct_gene_all"] = [
        int(len(a) > 0 and a.issubset(b)) for a, b in zip(dis_sets, top_sets)
    ]

    # ensure llm map is correct gene for g2p id
    filtered_clean_genes = filtered_clean_genes[
        filtered_clean_genes["dis_map_correct_gene_all"] == 1
    ]

    filtered_clean_genes.to_parquet(args.out_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
This script is a reimplementation of the final_data_clean.py script.

Description:
It reads the parquet output file from the LLM run and filter out the invalid results.

Usage:
  python final_data_clean_v2.py \
    --llm_file pubmed_bert_positive_2025_crossencoded_shard0-of-1__llm.parquet \
    --g2p_file g2p_2025.csv \
    --gene2pubtator gene2pubtator.gz \
    --output_csv final_cleaned_data.csv
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import re
import sys

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read a parquet file and print a summary.")
    parser.add_argument(
        "--llm_file",
        required=True,
        help="Output parquet file of the LLM step",
    )
    parser.add_argument(
        "--g2p_file",
        required=True,
        help="G2P CSV file",
    )
    parser.add_argument(
        "--gene2pubtator",
        required=True,
        help="gene2pubtator compressed file",
    )
    parser.add_argument(
        "--score_cutoff",
        type=float,
        default=0.9,
        help="Minimum top5_cross score to keep a row",
    )
    parser.add_argument(
        "--output_csv",
        required=True,
        help="Output CSV file with PMID and G2P_IDs",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug printing",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    parquet_path = args.llm_file
    g2p_path = args.g2p_file
    gene2pubtator_path = args.gene2pubtator
    score_cutoff = args.score_cutoff
    output_csv = args.output_csv
    debug = args.debug

    total_count= 0
    valid_count = 0

    if not os.path.exists(parquet_path):
        print(f"File not found: {parquet_path}", file=sys.stderr)
        return 1
    if not os.path.exists(g2p_path):
        print(f"File not found: {g2p_path}", file=sys.stderr)
        return 1
    if not os.path.exists(gene2pubtator_path):
        print(f"File not found: {gene2pubtator_path}", file=sys.stderr)
        return 1

    # Build G2P map: g2p id -> [gene symbol, previous gene symbols...]
    g2p_map_gene = {}
    g2p_map_disease = {}
    with open(g2p_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            g2p_id = (row.get("g2p id") or "").strip()
            if not g2p_id:
                continue
            gene_symbol = (row.get("gene symbol") or "").strip()
            prev = (row.get("previous gene symbols") or "").strip()
            prev_list = [p.strip() for p in prev.split(";") if p.strip()]
            values = []
            if gene_symbol:
                values.append(gene_symbol)
            values.extend(prev_list)
            g2p_map_gene[g2p_id] = values
            # disease
            g2p_disease = row.get("disease name").strip()
            g2p_map_disease[g2p_id] = g2p_disease # Not used in current script but can be useful for future extensions

    # Stream through parquet and build dicts: pmid -> llm_dis_map, pmid -> top5_cross
    pmid_to_llm = {}
    pmid_to_top5 = {}
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(columns=["pmid", "llm_dis_map", "top5_cross"]):
        pmids = batch.column(0).to_pylist()
        llm_vals = batch.column(1).to_pylist()
        top5_vals = batch.column(2).to_pylist()
        for pmid, llm_val, top5_val in zip(pmids, llm_vals, top5_vals):
            pmid_to_llm[pmid] = llm_val
            # Build dict with scores: g2p id -> score
            for item in top5_val:
                label = item.get("label", "")
                score = item.get("score", None)
                # Extract the g2p id from the label
                m = re.search(r"^(G2P\d+)\b", label or "")
                g2p_id = m.group(1) if m else None
                if pmid not in pmid_to_top5:
                    pmid_to_top5[pmid] = {g2p_id: score}
                else:
                    pmid_to_top5[pmid][g2p_id] = score

    pmid_set = set(pmid_to_llm.keys())

    # Build gene2pubtator map: first column -> list of fourth column values.
    gene2pubtator_map = {}
    with gzip.open(gene2pubtator_path, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            key = parts[0]
            if key not in pmid_set:
                continue
            value = parts[3]
            values = [v.strip() for v in value.split("|") if v.strip()]
            gene2pubtator_map.setdefault(key, []).extend(values)

    with open(output_csv, "w", newline="", encoding="utf-8") as out_f:
        out_writer = csv.writer(out_f)
        out_writer.writerow(["PMID", "G2P_IDs"])

        for pmid in pmid_to_llm:
            if pmid_to_llm[pmid] == "NO MATCH":
                continue
            list_of_g2ps = pmid_to_llm[pmid].split(";")
            list_of_genes_for_pmid = gene2pubtator_map.get(pmid, [])
            genes_for_pmid_set = set(list_of_genes_for_pmid)

            if debug:
                print(f"\nPMID: {pmid}; genes: {list_of_genes_for_pmid}")

            for g2p in list_of_g2ps:
                total_count += 1
                g2p = g2p.strip()

                # Filter by top5_cross score
                if score_cutoff:
                    top5_scores = pmid_to_top5.get(pmid, {})
                    score = top5_scores.get(g2p, None)
                    if score is not None and score < score_cutoff:
                        if debug:
                            print(f"{pmid}\t{g2p}\tSCORE BELOW CUTOFF ({score:.2f} < {score_cutoff})")
                        continue

                if g2p not in g2p_map_gene:
                    continue

                g2p_genes = g2p_map_gene[g2p]
                if any(g in genes_for_pmid_set for g in g2p_genes):
                    valid_count += 1
                    if debug:
                        print(f"{pmid}\t{g2p}\tVALID")
                    out_writer.writerow([pmid, g2p])
                else:
                    continue

    print(f"Loaded: {parquet_path}")
    print(f"Total mappings: {total_count}")
    print(f"Valid mappings: {valid_count}")
    print(f"Wrote: {output_csv}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

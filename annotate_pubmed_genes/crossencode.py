#!/usr/bin/env python3

"""
Script to:
    - cross-encode PubMed `tiab` text against grouped gene/disease-domain entries
    - keep the top-k matches per row
    - write shard-level parquet outputs with a structured `top_matches` column
It should be run on a GPU for best performance and is designed to be shardable
across multiple processes.
"""

import os
import gc
import argparse
from pathlib import Path
import traceback
from typing import List, Tuple, Optional, Dict, Any
import heapq

import torch
import polars as pl
import numpy as np
from sentence_transformers import CrossEncoder
import pandas as pd


PARQUET_COMPRESSION = "zstd"
SKIP_IF_EXISTS = True


def get_device(device_str: Optional[str] = None) -> str:
    if device_str:
        return device_str
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def pick_torch_dtype(dtype_str: str = "auto") -> Optional[torch.dtype]:
    if dtype_str == "fp32":
        return torch.float32
    if dtype_str == "fp16":
        return torch.float16
    if dtype_str == "bf16":
        return torch.bfloat16

    # auto
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return None


def load_crossencoder(
    model_path: str,
    device_str: Optional[str] = None,
    dtype_str: str = "auto",
) -> Tuple[CrossEncoder, str]:
    device = get_device(device_str)
    dtype = pick_torch_dtype(dtype_str)

    model_kwargs = {}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    model = CrossEncoder(
        model_path,
        device=device,
        model_kwargs=model_kwargs,
    )

    # perf knobs for Ampere/Hopper
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        # sanity print
        try:
            print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
            print(f"[INFO] Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        except Exception:
            pass

    return model, device


def detect_sep(candidate_path: str) -> str:
    suffix = Path(candidate_path).suffix.lower()
    if suffix in {".tsv", ".tab"}:
        return "\t"
    if suffix == ".csv":
        return ","
    if suffix == ".txt":
        with open(candidate_path, "r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                if "\t" in stripped:
                    return "\t"
                if "," in stripped:
                    return ","
                break
        raise ValueError(
            "Could not detect delimiter for .txt candidate file; expected tab- or comma-delimited content."
        )
    raise ValueError(
        f"Unsupported candidate file extension: {suffix or '<none>'}. Use .csv, .tsv, .tab, or .txt."
    )


def build_candidate_records(candidate_path: str) -> List[Dict[str, Any]]:
    candidate_pd = pd.read_csv(
        candidate_path,
        sep=detect_sep(candidate_path),
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        engine="python",
    )
    candidates = pl.from_pandas(candidate_pd)

    required_cols = ["gene_symbol", "disease_domain", "disease_synonym"]
    missing_cols = [col for col in required_cols if col not in candidates.columns]
    if missing_cols:
        raise ValueError(f"Input candidate file missing required columns: {missing_cols}")

    candidates = candidates.with_columns(
        pl.col("gene_symbol").cast(pl.Utf8, strict=False).str.strip_chars(),
        pl.col("disease_domain").cast(pl.Utf8, strict=False).str.strip_chars(),
        pl.col("disease_synonym").cast(pl.Utf8, strict=False).str.strip_chars(),
    )

    candidates = candidates.filter(
        (pl.col("gene_symbol") != "") & (pl.col("disease_domain") != "")
    )

    grouped = candidates.group_by(["gene_symbol", "disease_domain"]).agg(
        pl.col("disease_synonym")
        .filter(pl.col("disease_synonym") != "")
        .unique()
        .sort()
        .alias("disease_synonyms")
    )

    records: List[Dict[str, Any]] = []
    for row in grouped.iter_rows(named=True):
        synonyms = row["disease_synonyms"] or []
        parts = [
            f"Gene {row['gene_symbol']}",
            f"associated with {row['disease_domain']}",
        ]
        if synonyms:
            parts.append(f"related disease terms: {', '.join(synonyms)}")
        records.append(
            {
                "gene_symbol": row["gene_symbol"],
                "disease_domain": row["disease_domain"],
                "disease_synonyms": synonyms,
                "match_text": ". ".join(parts),
            }
        )

    return records


def load_shard_df(input_parquet: str, shard: int, num_shards: int) -> pl.DataFrame:
    df_shard = (
        pl.scan_parquet(input_parquet)
        .with_row_index(name="row_nr")
        .filter((pl.col("row_nr") % pl.lit(num_shards)) == pl.lit(shard))
        .sort("row_nr")
        .collect(streaming=True)
    )
    return df_shard


def update_topk_heaps_from_block(
    heaps: List[List[Tuple[float, int]]],  # min-heaps of (score, candidate_idx)
    scores_block: np.ndarray,              # shape (C, L_block)
    g_block_indices: List[int],
    k: int,
) -> None:
    C, L = scores_block.shape
    tk = min(k, L)

    for i in range(C):
        row = scores_block[i]
        if tk < L:
            idxs = np.argpartition(row, -tk)[-tk:]
        else:
            idxs = np.arange(L)
        heap = heaps[i]
        for j in idxs:
            s = float(row[j])
            candidate_idx = g_block_indices[j]
            if len(heap) < k:
                heapq.heappush(heap, (s, candidate_idx))
            else:
                if s > heap[0][0]:
                    heapq.heapreplace(heap, (s, candidate_idx))


def score_block_streaming(
    model: CrossEncoder,
    chunk_texts: List[str],
    g_block: List[str],
    pair_batch_size: int,
) -> np.ndarray:
    """
    Scores all (text, g2p) pairs for this chunk and g_block without
    materializing a gigantic C*L list of Python tuples.

    Returns shape (C, L) float32 array.
    """
    C = len(chunk_texts)
    L = len(g_block)
    out = np.empty((C, L), dtype=np.float32)

    for i, t in enumerate(chunk_texts):
        scores_row: List[float] = []
        for j in range(0, L, pair_batch_size):
            pairs = [(t, g) for g in g_block[j:j + pair_batch_size]]
            # predict returns list/np array
            scores_part = model.predict(pairs, batch_size=pair_batch_size)
            scores_row.extend(scores_part.tolist() if hasattr(scores_part, "tolist") else list(scores_part))
        out[i, :] = np.asarray(scores_row, dtype=np.float32)

    return out


def crossencode_topk_for_chunk(
    model: CrossEncoder,
    chunk_texts: List[str],
    candidate_records: List[Dict[str, Any]],
    top_k: int = 5,
    pair_batch_size: int = 512,
    g_block_size: int = 2000,
) -> List[List[Dict[str, Any]]]:
    C = len(chunk_texts)
    if C == 0:
        return []
    if top_k <= 0:
        raise ValueError("top_k must be >= 1")

    candidate_texts = [record["match_text"] for record in candidate_records]
    heaps: List[List[Tuple[float, int]]] = [[] for _ in range(C)]

    for start in range(0, len(candidate_texts), g_block_size):
        end = min(start + g_block_size, len(candidate_texts))
        g_block = candidate_texts[start:end]
        g_block_indices = list(range(start, end))

        scores_block = score_block_streaming(
            model=model,
            chunk_texts=chunk_texts,
            g_block=g_block,
            pair_batch_size=pair_batch_size,
        )

        update_topk_heaps_from_block(heaps, scores_block, g_block_indices, top_k)

        del scores_block

    topk_lists: List[List[Dict[str, Any]]] = []
    for heap in heaps:
        sorted_desc = sorted(heap, key=lambda x: x[0], reverse=True)
        topk_lists.append(
            [
                {
                    "gene_symbol": candidate_records[candidate_idx]["gene_symbol"],
                    "disease_domain": candidate_records[candidate_idx]["disease_domain"],
                    "disease_synonyms": candidate_records[candidate_idx]["disease_synonyms"],
                    "match_text": candidate_records[candidate_idx]["match_text"],
                    "score": float(score),
                }
                for (score, candidate_idx) in sorted_desc
            ]
        )
    return topk_lists


def process_shard(
    input_parquet: str,
    candidate_file: str,
    out_dir: str,
    model_path: str,
    device: Optional[str] = None,
    dtype: str = "auto",
    chunk_size: int = 256,          # bumped for A100
    pair_batch_size: int = 512,     # bumped for A100
    g_block_size: int = 2000,       # >=1600 => single block for your case
    top_k: int = 5,
    shard: int = 0,
    num_shards: int = 1,
    skip_if_exists: bool = SKIP_IF_EXISTS,
    compression: str = PARQUET_COMPRESSION,
) -> bool:
    os.makedirs(out_dir, exist_ok=True)

    base = os.path.basename(input_parquet)
    stem = os.path.splitext(base)[0]
    out_path = os.path.join(out_dir, f"{stem}_crossencoded_shard{shard}-of-{num_shards}.parquet")

    if skip_if_exists and os.path.exists(out_path):
        print(f"Skipping (already exists): {out_path}")
        return True

    if chunk_size <= 0:
        raise ValueError("chunk_size must be >= 1")
    if pair_batch_size <= 0:
        raise ValueError("pair_batch_size must be >= 1")
    if g_block_size <= 0:
        raise ValueError("g_block_size must be >= 1")
    if top_k <= 0:
        raise ValueError("top_k must be >= 1")
    if num_shards <= 0:
        raise ValueError("num_shards must be >= 1")
    if shard < 0 or shard >= num_shards:
        raise ValueError("shard must satisfy 0 <= shard < num_shards")

    # Load grouped candidate list
    try:
        print(f"[INFO] Loading candidate file from: {candidate_file}")
        candidate_records = build_candidate_records(candidate_file)
    except Exception:
        print(f"[ERROR] Failed to load/prepare candidate file: {candidate_file}")
        traceback.print_exc()
        return False

    if not candidate_records:
        print("[ERROR] Candidate file produced no valid grouped entries.")
        return False

    print(f"[INFO] Grouped candidate entries (M): {len(candidate_records)}")

    # Load shard of the input parquet
    try:
        print(f"[INFO] Loading shard {shard}/{num_shards} from: {input_parquet}")
        df_shard = load_shard_df(input_parquet, shard, num_shards)
    except Exception:
        print(f"[ERROR] Failed to load shard {shard}/{num_shards} from {input_parquet}")
        traceback.print_exc()
        return False

    if df_shard.height == 0:
        print(f"[INFO] No rows for shard {shard}/{num_shards}; nothing to do.")
        return True

    # Ensure tiab exists + sanitize nulls
    if "tiab" not in df_shard.columns:
        print("[ERROR] Input parquet must contain a 'tiab' column.")
        return False

    tiab_list = df_shard.get_column("tiab").fill_null("").to_list()
    N = len(tiab_list)
    print(f"[INFO] Shard rows (N_shard): {N}")

    # Load model
    try:
        print(f"[INFO] Loading CrossEncoder from: {model_path}")
        model, device_str = load_crossencoder(model_path, device, dtype)
        print(f"[INFO] Model loaded on device: {device_str}")
    except Exception:
        print(f"[ERROR] Failed to load CrossEncoder: {model_path}")
        traceback.print_exc()
        return False

    # Compute top-k per row in chunks
    all_topk: List[Optional[List[Dict[str, Any]]]] = [None] * N

    try:
        total_chunks = (N + chunk_size - 1) // chunk_size
        for ci, chunk_start in enumerate(range(0, N, chunk_size), start=1):
            chunk_end = min(chunk_start + chunk_size, N)
            print(f"[INFO] Processing chunk {ci}/{total_chunks}: rows {chunk_start}:{chunk_end}")
            chunk_texts = tiab_list[chunk_start:chunk_end]

            topk_chunk = crossencode_topk_for_chunk(
                model=model,
                chunk_texts=chunk_texts,
                candidate_records=candidate_records,
                top_k=top_k,
                pair_batch_size=pair_batch_size,
                g_block_size=g_block_size,
            )

            all_topk[chunk_start:chunk_end] = topk_chunk
            print(f"[INFO] Finished chunk {ci}/{total_chunks}")

            # Clean up per-chunk only (avoid doing this in inner loops)
            del chunk_texts, topk_chunk
            if device_str.startswith("cuda") and (ci % 10 == 0):
                torch.cuda.empty_cache()
            # gc.collect() only if you actually observe host RAM creep; keep it light:
            if ci % 25 == 0:
                gc.collect()

    except Exception:
        print(f"[ERROR] Failure during crossencoding for shard {shard}")
        traceback.print_exc()
        return False

    # Attach result column and write shard parquet
    try:
        topk_dtype = pl.List(
            pl.Struct([
                pl.Field("gene_symbol", pl.Utf8),
                pl.Field("disease_domain", pl.Utf8),
                pl.Field("disease_synonyms", pl.List(pl.Utf8)),
                pl.Field("match_text", pl.Utf8),
                pl.Field("score", pl.Float64),
            ])
        )
        print(f"[INFO] Writing output to: {out_path}")
        df_out = df_shard.with_columns(
            pl.Series(name="top_matches", values=all_topk, dtype=topk_dtype)
        )
        df_out.write_parquet(out_path, compression=compression)
        print(f"[INFO] Wrote {df_out.height} rows to {out_path}")
    except Exception:
        print(f"[ERROR] Failed to write output parquet: {out_path}")
        traceback.print_exc()
        return False
    finally:
        del df_shard
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_parquet", type=Path, required=True, help="Path to input parquet file with a 'tiab' column")
    ap.add_argument(
        "--candidate_file",
        type=Path,
        required=True,
        help="Path to CSV/TSV with columns: gene_symbol, disease_domain, disease_synonym",
    )
    ap.add_argument("--out_dir", type=Path, required=True, help="Output directory for shard parquets")
    ap.add_argument("--model_path", type=str, required=True, help="CrossEncoder model path")
    ap.add_argument("--device", type=str, default=None, help="Device string, e.g., cuda:0, cuda:1")
    ap.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp32", "fp16", "bf16"], help="Model dtype")
    ap.add_argument("--chunk_size", type=int, default=256, help="Number of tiab rows to process per chunk")
    ap.add_argument("--pair_batch_size", type=int, default=512, help="CrossEncoder prediction batch size")
    ap.add_argument("--g_block_size", type=int, default=2000, help="How many grouped candidate entries to score at once")
    ap.add_argument("--top_k", type=int, default=5, help="Top-K gene/domain entries to keep per text")
    ap.add_argument("--shard", type=int, default=0, help="Shard index for row-wise sharding")
    ap.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    ap.add_argument("--skip_if_exists", action="store_true", help="Skip if shard output already exists")
    ap.add_argument("--no_skip_if_exists", dest="skip_if_exists", action="store_false")
    ap.set_defaults(skip_if_exists=SKIP_IF_EXISTS)
    ap.add_argument("--compression", type=str, default=PARQUET_COMPRESSION, help="Parquet compression (e.g., zstd, snappy)")
    args = ap.parse_args()

    ok = process_shard(
        input_parquet=args.input_parquet,
        candidate_file=args.candidate_file,
        out_dir=args.out_dir,
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        chunk_size=args.chunk_size,
        pair_batch_size=args.pair_batch_size,
        g_block_size=args.g_block_size,
        top_k=args.top_k,
        shard=args.shard,
        num_shards=args.num_shards,
        skip_if_exists=args.skip_if_exists,
        compression=args.compression,
    )
    if not ok:
        raise SystemExit(1)

if __name__ == "__main__":
    main()

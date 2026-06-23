#!/usr/bin/env python3

"""
Script to run vLLM over cross-encoded PubMed shards:
    - build structured prompts from each row's `tiab` and top candidate gene/domain matches
    - generate LLM outputs in batches
    - write shard-level parquet outputs with prompt and mapping fields

Run on GPU for best performance.
Supports optional sharding for distributed processing across multiple workers.
Designed for incremental saving to avoid data loss and enable monitoring of progress.
"""

import os
import re
import gc
import glob
import ast
import json
import argparse
from itertools import combinations
import pandas as pd
import pyarrow as pa
from vllm import LLM, SamplingParams
import torch
import numpy as np

try:
    from vllm.sampling_params import StructuredOutputsParams
except Exception:
    StructuredOutputsParams = None


def build_llm_prompt(tiab, candidate_structs, top_k):
    return (
        f"""System/Developer Instruction:
        You are an expert in genetic disease curation. Your task is to map a scientific Title+Abstract (TIAB) to zero or more candidate gene-disease domain pairs.

        You will receive:
        - A TIAB
        - Up to {top_k} candidate gene-disease domain pairs, provided as structured fields:
          GENE_SYMBOL, DISEASE_DOMAIN, DISEASE_SYNONYMS

        Task:
        Determine whether the TIAB supports any of the candidate gene-disease domain pairs.

        You must follow all rules below. Do not invent genes or disease domains. Only select from the provided candidates.

        How to decide:
        - A candidate is supported only if the TIAB matches the candidate gene and the candidate disease.
        - Gene match and disease match are the primary criteria.
        - Disease match can be based on the same disease name, a clear synonym, or a clearly matching phenotype description.

        Selection:
        - Return one object if one candidate is clearly supported.
        - Return multiple objects only if the TIAB clearly describes multiple distinct gene-disease matches.
        - Return an empty list only if no provided candidate has both a supported gene match and a supported disease-domain match in the TIAB.

        Output:
        Return exactly one line and nothing else:
        ANSWER: []
        or
        ANSWER: [{"gene_symbol":"GENE","disease_domain":"DOMAIN"}]
        or
        ANSWER: [{"gene_symbol":"GENE1","disease_domain":"DOMAIN1"},{"gene_symbol":"GENE2","disease_domain":"DOMAIN2"}]

    TIAB:
    {tiab}

    Candidate Gene-Disease Pairs (structured):
    """
        + "\n".join(candidate_structs)
        + "\nReturn exactly one line in the schema above."
    )


def extract_last_answer(text):
    matches = re.findall(r"ANSWER:\s*(.*)", text or "")
    return matches[-1].strip() if matches else None


def extract_candidate_item(item):
    if isinstance(item, dict):
        return item
    try:
        if pa is not None and isinstance(item, pa.Scalar):
            item = item.as_py()
            if isinstance(item, dict):
                return item
    except Exception:
        pass
    return None


def extract_score(item):
    if isinstance(item, dict):
        return item.get("score")
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        return item[1]
    try:
        if pa is not None and isinstance(item, pa.Scalar):
            return extract_score(item.as_py())
    except Exception:
        pass
    return np.nan


def to_candidate_records(x, top_k):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []

    if isinstance(x, (list, tuple, np.ndarray)):
        records = []
        for it in x.tolist() if isinstance(x, np.ndarray) else x:
            record = extract_candidate_item(it)
            if record:
                records.append(
                    {
                        "gene_symbol": str(record.get("gene_symbol", "")).strip(),
                        "disease_domain": str(record.get("disease_domain", "")).strip(),
                        "disease_synonyms": [
                            str(s).strip()
                            for s in (record.get("disease_synonyms") or [])
                            if str(s).strip()
                        ],
                        "match_text": str(record.get("match_text", "")).strip(),
                        "score": record.get("score"),
                    }
                )
        return records[:top_k]

    if isinstance(x, str):
        obj = None
        try:
            obj = json.loads(x)
        except Exception:
            try:
                obj = ast.literal_eval(x)
            except Exception:
                return []
        return to_candidate_records(obj, top_k)

    try:
        if pa is not None and isinstance(x, pa.Scalar):
            return to_candidate_records(x.as_py(), top_k)
    except Exception:
        pass

    return []


def format_candidate_structs(candidate_records):
    if not candidate_records:
        return []
    structs = []
    for idx, data in enumerate(candidate_records, start=1):
        synonyms = data.get("disease_synonyms") or []
        line = (
            f"{idx}) GENE_SYMBOL: {data.get('gene_symbol', '')} | "
            f"DISEASE_DOMAIN: {data.get('disease_domain', '')} | "
            f"DISEASE_SYNONYMS: {', '.join(synonyms)}"
        )
        structs.append(line)
    return structs


def extract_gene_domain_pairs(candidate_records):
    pairs = []
    for data in candidate_records or []:
        gene_symbol = str(data.get("gene_symbol", "")).strip()
        disease_domain = str(data.get("disease_domain", "")).strip()
        if gene_symbol and disease_domain:
            pair = {
                "gene_symbol": gene_symbol,
                "disease_domain": disease_domain,
            }
            if pair not in pairs:
                pairs.append(pair)
    return pairs


def build_allowed_answer_choices(candidate_pairs):
    choices = []
    for n in range(1, len(candidate_pairs) + 1):
        for combo in combinations(candidate_pairs, n):
            payload = json.dumps(list(combo), separators=(",", ":"))
            choices.append(f"ANSWER: {payload}")
    choices.append("ANSWER: []")
    return choices


def build_guided_sampling_params(candidate_pairs, temperature, top_p, max_tokens):
    choices = build_allowed_answer_choices(candidate_pairs)
    if StructuredOutputsParams is not None:
        return SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            structured_outputs=StructuredOutputsParams(choice=choices),
        )
    raise RuntimeError(
        "No structured decoding API available (StructuredOutputsParams)."
    )


def parse_answer_payload(answer_text):
    if answer_text is None:
        return []
    answer_text = answer_text.strip()
    if not answer_text:
        return []
    try:
        payload = json.loads(answer_text)
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    out = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        gene_symbol = str(item.get("gene_symbol", "")).strip()
        disease_domain = str(item.get("disease_domain", "")).strip()
        if gene_symbol and disease_domain:
            out.append(
                {
                    "gene_symbol": gene_symbol,
                    "disease_domain": disease_domain,
                }
            )
    return out


def save_progress(df, generated_texts, out_parquet):
    df["generated_text"] = generated_texts
    df["llm_answer"] = [extract_last_answer(t) for t in generated_texts]
    df["llm_gene_disease_map"] = df["llm_answer"].apply(parse_answer_payload)
    df.to_parquet(out_parquet, index=False)
    print(f"[PROGRESS] Saved current progress to {out_parquet}")


def batched_indices(start, end, batch_size):
    i = start
    while i < end:
        j = min(i + batch_size, end)
        yield i, j
        i = j


def select_shards_for_worker(all_paths, shard_index, num_shards):
    if shard_index is None or num_shards is None:
        return all_paths
    return [p for i, p in enumerate(all_paths) if (i % num_shards) == shard_index]


def list_input_shards(shards_dir):
    """
    List only input shard parquets and exclude any LLM outputs written by this
    script (which use the "__llm.parquet" suffix).
    """
    all_parquets = sorted(glob.glob(os.path.join(shards_dir, "*.parquet")))
    input_parquets = [p for p in all_parquets if "__llm" not in os.path.basename(p)]
    excluded = len(all_parquets) - len(input_parquets)
    return input_parquets, excluded


def run_llm_over_cross_shards(
    shards_dir,
    llm_model,
    out_dir=None,
    top_k=5,
    batch_size=32,
    temperature=0.0,
    top_p=1.0,
    max_tokens=64,
    save_pickle=False,
    shard_index=None,
    num_shards=None,
    save_every=1000,
    tensor_parallel_size=None,
    max_model_len=None,
    disable_guided_decoding=False,
):
    """
    - Reads *.parquet from shards_dir
    - Builds prompts from 'tiab' and 'top_matches'
    - Runs vLLM in batches
    - Incrementally writes output parquet per shard every `save_every` rows,
      overwriting the same file each time:
        columns added: 'candidate_matches', 'llm_prompt', 'generated_text', 'llm_answer', 'llm_gene_disease_map'
    - Shard-aware: if shard_index/num_shards are provided, each worker processes
      its subset of shard files (index % num_shards == shard_index).
    """
    os.makedirs(out_dir or shards_dir, exist_ok=True)
    out_dir = out_dir or shards_dir
    if top_k <= 0:
        raise ValueError("top_k must be >= 1")

    if os.path.abspath(out_dir) == os.path.abspath(shards_dir):
        print(
            "[WARN] out_dir == shards_dir; input discovery will exclude __llm outputs."
        )

    # Initialize LLM once for all shards
    sampling_params = SamplingParams(
        temperature=temperature, top_p=top_p, max_tokens=max_tokens
    )
    guided_enabled = not disable_guided_decoding
    if guided_enabled and StructuredOutputsParams is None:
        print(
            "[WARN] Structured decoding is unavailable in this vLLM build; "
            "falling back to unconstrained decoding."
        )
        guided_enabled = False
    llm_kwargs = {}
    if tensor_parallel_size is not None:
        llm_kwargs["tensor_parallel_size"] = int(tensor_parallel_size)
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = int(max_model_len)
    llm = LLM(model=llm_model, **llm_kwargs)

    shard_paths, excluded = list_input_shards(shards_dir)
    if excluded:
        print(f"[INFO] Excluding {excluded} __llm parquet(s) from input discovery.")
    shard_paths = select_shards_for_worker(shard_paths, shard_index, num_shards)

    print(f"Found {len(shard_paths)} parquet shard(s) for this worker.")

    for shard_path in shard_paths:
        print(f"Processing shard: {os.path.basename(shard_path)}")
        df = pd.read_parquet(shard_path)
        if "top_matches" not in df.columns:
            raise ValueError(f"Input shard missing required column 'top_matches': {shard_path}")
        scores_max = (
            df["top_matches"]
            .explode()
            .apply(extract_score)
            .groupby(level=0)
            .max()
            .reindex(df.index, fill_value=np.nan)
        )
        df = df.loc[scores_max.ge(0.01).fillna(False)].copy()

        df["candidate_matches"] = df["top_matches"].apply(
            lambda x: to_candidate_records(x, top_k)
        )
        df["candidate_gene_disease_pairs"] = df["candidate_matches"].apply(
            extract_gene_domain_pairs
        )

        # Build prompts with structured candidate fields
        df["candidate_structs"] = df["candidate_matches"].apply(
            format_candidate_structs
        )
        df["llm_prompt"] = df.apply(
            lambda row: build_llm_prompt(
                tiab=row.get("tiab", ""),
                candidate_structs=row.get("candidate_structs", []),
                top_k=top_k,
            ),
            axis=1,
        )

        N = len(df)
        if N == 0:
            print("[INFO] No rows passed the score filter for this shard.")
            continue

        print("Sample candidates:", df["candidate_matches"].iloc[0])
        print("Prompt preview:\n", df["llm_prompt"].iloc[0][:800])
        print(f"Total rows in shard: {N}")
        generated_texts = [""] * N  # Will fill incrementally

        base = os.path.splitext(os.path.basename(shard_path))[0]
        out_parquet = os.path.join(out_dir, f"{base}__llm.parquet")

        # Process in chunks of save_every rows
        for chunk_start in range(0, N, save_every):
            chunk_end = min(chunk_start + save_every, N)
            print(f"  Processing rows {chunk_start}..{chunk_end - 1}")

            # Generate in vLLM batches over this chunk
            for b_start, b_end in batched_indices(chunk_start, chunk_end, batch_size):
                batch_prompts = df["llm_prompt"].iloc[b_start:b_end].tolist()
                if guided_enabled:
                    batch_candidate_pairs = df["candidate_gene_disease_pairs"].iloc[b_start:b_end].tolist()
                    batch_sampling_params = [
                        build_guided_sampling_params(
                            candidate_pairs=pairs,
                            temperature=temperature,
                            top_p=top_p,
                            max_tokens=max_tokens,
                        )
                        for pairs in batch_candidate_pairs
                    ]
                    outputs = llm.generate(batch_prompts, batch_sampling_params)
                else:
                    outputs = llm.generate(batch_prompts, sampling_params)
                for j, out in enumerate(outputs):
                    item = out.outputs[0]
                    generated_texts[b_start + j] = item.text
                    if getattr(item, "finish_reason", None) == "length":
                        print(
                            f"[WARN] Row {b_start + j} reached max_tokens={max_tokens}; "
                            "output may be truncated."
                        )

            # Save after this chunk (overwrite file)
            save_progress(df, generated_texts, out_parquet)

        print(f"[DONE] Shard completed: {os.path.basename(shard_path)}")

        # Optional: free some memory between shards (vLLM keeps its KV cache though)
        torch.cuda.empty_cache()
        gc.collect()

    print("All shards processed.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--shards_dir", required=True, type=str)
    p.add_argument("--llm_model", required=True, type=str)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_tokens", type=int, default=64)
    p.add_argument("--save_pickle", action="store_true")  # kept for API compatibility
    p.add_argument("--shard_index", type=int, default=None)
    p.add_argument("--num_shards", type=int, default=None)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--tensor_parallel_size", type=int, default=None)
    p.add_argument("--max_model_len", type=int, default=None)
    p.add_argument("--disable_guided_decoding", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_llm_over_cross_shards(
        shards_dir=args.shards_dir,
        llm_model=args.llm_model,
        out_dir=args.out_dir,
        top_k=args.top_k,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        save_pickle=args.save_pickle,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        save_every=args.save_every,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        disable_guided_decoding=args.disable_guided_decoding,
    )

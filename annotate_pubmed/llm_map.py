#!/usr/bin/env python3

"""
Script to run vLLM over cross-encoded PubMed shards:
    - build structured prompts from each row's `tiab` and top-5 G2P candidates
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


def build_llm_prompt(tiab, candidate_structs):
    return (
        f"""System/Developer Instruction:
        You are an expert in genetic disease curation. Your task is to map a scientific Title+Abstract (TIAB) to one or more candidate G2P LGMDE threads.

        You will receive:
        - A TIAB
        - Up to 5 candidate LGMDE threads, provided as structured fields:
          G2P_ID, GENE, DISEASE, ALLELIC_REQUIREMENT, MECHANISM

        Task:
        Determine whether the TIAB supports any of the candidate G2P records.

        You must follow all rules below. Do not invent any G2P IDs. Only select from the 5 provided candidates.

        How to decide:
        - A candidate is supported only if the TIAB matches the candidate gene and the candidate disease.
        - Gene match and disease match are the primary criteria.
        - Disease match can be based on the same disease name, a clear synonym, or a clearly matching phenotype description.
        - Use ALLELIC_REQUIREMENT only when the TIAB provides inheritance or zygosity information.
        - Use MECHANISM only when the TIAB provides molecular mechanism information.
        - Do not reject a candidate only because ALLELIC_REQUIREMENT or MECHANISM is not mentioned in the TIAB.

        Selection:
        - Return one G2P ID if one candidate is clearly supported.
        - Return multiple G2P IDs only if the TIAB clearly describes multiple distinct gene-disease matches.
        - Return NO MATCH if none of the candidates are supported by the TIAB.

        Output:
        Return exactly one line and nothing else:
        ANSWER: G2PID
        or
        ANSWER: G2PID;G2PID
        or
        ANSWER: NO MATCH

    TIAB:
    {tiab}

    Candidate LGMDE Threads (structured):
    """
        + "\n".join(candidate_structs)
        + "\nReturn exactly one line in the schema above."
    )


def extract_last_answer(text):
    matches = re.findall(r"ANSWER:\s*(.*)", text or "")
    return matches[-1].strip() if matches else None


def extract_label_from_item(item):
    if isinstance(item, dict):
        return str(item.get("label", "")).strip() or None
    if isinstance(item, (list, tuple)) and len(item) >= 1:
        return str(item[0]).strip() or None
    try:
        if pa is not None and isinstance(item, pa.Scalar):
            item = item.as_py()
            if isinstance(item, dict):
                return str(item.get("label", "")).strip() or None
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                return str(item[0]).strip() or None
    except Exception:
        pass
    if isinstance(item, str):
        s = item.strip()
        return s or None
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


def to_labels(x):
    # Normalize None/NaN
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []

    # If it’s already a list/tuple/np.ndarray, iterate
    if isinstance(x, (list, tuple, np.ndarray)):
        labels = []
        for it in x.tolist() if isinstance(x, np.ndarray) else x:
            lab = extract_label_from_item(it)
            if lab:
                labels.append(lab)
        return labels[:5]

    # If it’s a string, try JSON then literal_eval
    if isinstance(x, str):
        obj = None
        try:
            obj = json.loads(x)
        except Exception:
            try:
                obj = ast.literal_eval(x)
            except Exception:
                return []
        return to_labels(obj)

    # PyArrow List/Struct scalars at the top level
    try:
        if pa is not None and isinstance(x, pa.Scalar):
            return to_labels(x.as_py())
    except Exception:
        pass

    return []


def parse_structured_candidate(label_str):
    if not isinstance(label_str, str) or not label_str.strip():
        return {}
    parts = [p.strip() for p in label_str.split(" --- ")]
    keys = [
        "G2P_ID",
        "GENE",
        "GENE_MIM",
        "HGNC_ID",
        "PREVIOUS_GENE_SYMBOLS",
        "DISEASE",
        "DISEASE_MIM",
        "DISEASE_MONDO",
        "ALLELIC_REQUIREMENT",
        "CROSS_CUTTING_MODIFIER",
        "CONFIDENCE",
        "INFERRED_VARIANT_CONSEQUENCE",
        "VARIANT_TYPES",
        "MOLECULAR_MECHANISM",
        "MOLECULAR_MECHANISM_CATEGORISATION",
    ]
    data = {}
    for i, key in enumerate(keys):
        data[key] = parts[i] if i < len(parts) else ""
    return data


def format_candidate_structs(labels):
    if not labels:
        return []
    structs = []
    for idx, label in enumerate(labels, start=1):
        data = parse_structured_candidate(label)
        if not data:
            continue
        line = (
            f"{idx}) G2P_ID: {data.get('G2P_ID', '')} | "
            f"GENE: {data.get('GENE', '')} | "
            f"DISEASE: {data.get('DISEASE', '')} | "
            f"ALLELIC_REQUIREMENT: {data.get('ALLELIC_REQUIREMENT', '')} | "
            f"MOLECULAR_MECHANISM: {data.get('MOLECULAR_MECHANISM', '')}"
        )
        structs.append(line)
    return structs


def extract_candidate_ids(labels):
    ids = []
    for label in labels or []:
        data = parse_structured_candidate(label)
        g2p_id = str(data.get("G2P_ID", "")).strip()
        if g2p_id and g2p_id not in ids:
            ids.append(g2p_id)
    return ids


def build_allowed_answer_choices(candidate_ids):
    choices = []
    for n in range(1, len(candidate_ids) + 1):
        for combo in combinations(candidate_ids, n):
            choices.append(f"ANSWER: {';'.join(combo)}")
    choices.append("ANSWER: NO MATCH")
    return choices


def build_guided_sampling_params(candidate_ids, temperature, top_p, max_tokens):
    choices = build_allowed_answer_choices(candidate_ids)
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


def save_progress(df, generated_texts, out_parquet):
    df["generated_text"] = generated_texts
    df["llm_dis_map"] = [extract_last_answer(t) for t in generated_texts]
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
    - Builds prompts from 'tiab' and 'top5_cross'
    - Runs vLLM in batches
    - Incrementally writes output parquet per shard every `save_every` rows,
      overwriting the same file each time:
        columns added: 'top_5_cross_lgmde', 'llm_prompt', 'generated_text', 'llm_dis_map'
    - Shard-aware: if shard_index/num_shards are provided, each worker processes
      its subset of shard files (index % num_shards == shard_index).
    """
    os.makedirs(out_dir or shards_dir, exist_ok=True)
    out_dir = out_dir or shards_dir

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
        scores_max = (
            df["top5_cross"]
            .explode()
            .apply(extract_score)
            .groupby(level=0)
            .max()
            .reindex(df.index, fill_value=np.nan)
        )
        df = df.loc[scores_max.ge(0.01).fillna(False)].copy()

        df["top_5_cross_lgmde"] = df["top5_cross"].apply(to_labels)
        df["candidate_ids"] = df["top_5_cross_lgmde"].apply(extract_candidate_ids)

        # Build prompts with structured candidate fields
        df["candidate_structs"] = df["top_5_cross_lgmde"].apply(
            format_candidate_structs
        )
        df["llm_prompt"] = df.apply(
            lambda row: build_llm_prompt(
                tiab=row.get("tiab", ""),
                candidate_structs=row.get("candidate_structs", []),
            ),
            axis=1,
        )

        # temp just to check working
        print("Sample candidates:", df["top_5_cross_lgmde"].iloc[0] if len(df) else [])
        print("Prompt preview:\n", df["llm_prompt"].iloc[0][:800])

        N = len(df)
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
                    batch_candidate_ids = df["candidate_ids"].iloc[b_start:b_end].tolist()
                    batch_sampling_params = [
                        build_guided_sampling_params(
                            candidate_ids=ids,
                            temperature=temperature,
                            top_p=top_p,
                            max_tokens=max_tokens,
                        )
                        for ids in batch_candidate_ids
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

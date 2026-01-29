import os
import re
import gc
import glob
import ast
import json
import argparse
import pandas as pd
import pyarrow as pa
from vllm import LLM, SamplingParams
import torch
import numpy as np


def build_llm_prompt(tiab, candidate_structs):
    return (
        f"""System/Developer Instruction:
        You are an expert in genetic disease curation. Your task is to map a scientific Title+Abstract (TIAB) to one or more candidate G2P LGMDE threads.

        You will receive:
        - A TIAB
        - Up to 5 candidate LGMDE threads, provided as structured fields:
          G2P_ID, GENE, DISEASE, ALLELIC_REQUIREMENT, INHERITANCE, MECHANISM, EVIDENCE, VARIANT_TYPES, MOLECULAR_MECHANISM

        Goal:
        Select the best matching G2P ID(s) from the provided candidates, or return NO MATCH if none meet the required criteria.

        You must follow all rules below. Do not invent any G2P IDs. Only select from the 5 provided candidates.

        HARD FILTERING RULES (mandatory)
        A candidate (and the overall TIAB) MUST satisfy ALL of the following or it is ineligible:

        1) Gene overlap (mandatory)
        - The TIAB must explicitly mention at least one gene symbol (or alias) that exactly matches a GENE field in a candidate.
        - If no candidate shares a gene with the TIAB, return NO MATCH.

        2) Human evidence (mandatory)
        - The TIAB must include human subjects or explicit human diagnostic findings (for example: case reports, patient cohorts, family studies).
        - If the TIAB contains only non-human models (animal models, cell lines, in vitro) with no human patients, return NO MATCH.

        3) Disease type (mandatory)
        - The TIAB must describe germline or inherited disease.
        - If the TIAB describes only somatic variation (for example tumor sequencing, cancer-only studies) with no inherited or syndromic context, return NO MATCH.

        4) Study type exclusion
        - If the TIAB is explicitly a GWAS, polymorphism association study, or common-variant risk study without rare pathogenic variant interpretation, return NO MATCH.

        5) Negation exclusion
        - If the TIAB explicitly states that variants in a gene do NOT cause a disease, that candidate is ineligible.

        INFORMATION EXTRACTION (for ranking only)
        From the TIAB, identify when present:
        - Mentioned gene(s)
        - Disease name(s) and synonyms
        - Key phenotypes and affected systems
        - Inheritance or allelic clues (dominant, recessive, X-linked, biallelic, heterozygous, de novo, consanguinity)
        - Whether findings are emphasized in the title or opening sentence

        CANDIDATE SCORING AND RANKING
        For candidates that pass HARD FILTERING:
        Score using the following priorities (highest to lowest):
        1) Gene match (required for all candidates)

        2) Allelic requirement compatibility
        - If the TIAB explicitly states inheritance or zygosity, it must be compatible with the candidate.
        - If inheritance or zygosity is NOT stated, do NOT reject candidates based on allelic requirement alone.

        3) Disease name or clear synonym match
        - Strong positive evidence when present.

        4) Phenotype overlap
        - Hallmark or system-level phenotype overlap is positive evidence.
        - Partial matches are acceptable.

        Important:
        - If gene and allelic requirement clearly match but phenotype or disease naming differs, prefer the gene + allelic match. This may reflect differences in disease labeling rather than biology.

        SELECTION RULES
        1) Single best candidate:
        Return exactly one G2P ID if only one candidate clearly ranks highest based on the scoring rules above.

        2) Multiple candidates:
        Return multiple G2P IDs (semicolon-separated) only if the TIAB clearly describes multiple distinct gene–disease associations that independently match separate candidates.

        3) No match:
        Return NO MATCH ONLY if:
        - No candidate passes gene overlap filtering, or
        - The TIAB is non-human only, or
        - All candidates fail mandatory compatibility (for example: explicit dominant inheritance in TIAB versus strictly biallelic candidate).

        Do NOT force a match if all candidates are incompatible.

        OUTPUT FORMAT (STRICT)
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
    parts = [p.strip() for p in label_str.split(" - ")]
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
            f"INHERITANCE: {data.get('CROSS_CUTTING_MODIFIER', '')} | "
            f"MECHANISM: {data.get('MOLECULAR_MECHANISM', '')} | "
            f"EVIDENCE: {data.get('CONFIDENCE', '')} | "
            f"VARIANT_TYPES: {data.get('VARIANT_TYPES', '')}"
        )
        structs.append(line)
    return structs


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
    max_tokens=2048,
    save_pickle=False,
    shard_index=None,
    num_shards=None,
    save_every=1000,
    tensor_parallel_size=None,
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
    llm_kwargs = {}
    if tensor_parallel_size is not None:
        llm_kwargs["tensor_parallel_size"] = int(tensor_parallel_size)
    llm = LLM(model=llm_model, **llm_kwargs)

    shard_paths, excluded = list_input_shards(shards_dir)
    if excluded:
        print(f"[INFO] Excluding {excluded} __llm parquet(s) from input discovery.")
    shard_paths = select_shards_for_worker(shard_paths, shard_index, num_shards)

    print(f"Found {len(shard_paths)} parquet shard(s) for this worker.")

    for shard_path in shard_paths:
        print(f"Processing shard: {os.path.basename(shard_path)}")
        df = pd.read_parquet(shard_path)

        df["top_5_cross_lgmde"] = df["top5_cross"].apply(to_labels)

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
                outputs = llm.generate(batch_prompts, sampling_params)
                for j, out in enumerate(outputs):
                    generated_texts[b_start + j] = out.outputs[0].text

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
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--save_pickle", action="store_true")  # kept for API compatibility
    p.add_argument("--shard_index", type=int, default=None)
    p.add_argument("--num_shards", type=int, default=None)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--tensor_parallel_size", type=int, default=None)
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
    )

#!/usr/bin/env bash
set -euo pipefail

: "${ANNOTATE_PUBMED_DIR:?ANNOTATE_PUBMED_DIR is required}"
: "${LLM_OUT_DIR:?LLM_OUT_DIR is required}"
: "${G2P_CSV:?G2P_CSV is required}"
: "${GENE2PUBTATOR3:?GENE2PUBTATOR3 is required}"
: "${FINAL_OUT_CSV:?FINAL_OUT_CSV is required}"
: "${SCORE_CUTOFF:?SCORE_CUTOFF is required}"

source activate g2p-llm

shopt -s nullglob
llm_files=( "${LLM_OUT_DIR}"/*__llm.parquet )
shopt -u nullglob

if [[ "${#llm_files[@]}" -eq 0 ]]; then
  echo "No LLM parquet files found in ${LLM_OUT_DIR}" >&2
  exit 1
fi

if [[ "${#llm_files[@]}" -ne 1 ]]; then
  echo "final_data_clean_v2.py expects a single parquet file, found ${#llm_files[@]} in ${LLM_OUT_DIR}" >&2
  printf '  %s\n' "${llm_files[@]}" >&2
  exit 1
fi

final_args=(
  --llm_file "${llm_files[0]}"
  --g2p_file "${G2P_CSV}"
  --gene2pubtator "${GENE2PUBTATOR3}"
  --output_csv "${FINAL_OUT_CSV}"
  --score_cutoff "${SCORE_CUTOFF}"
)

if [[ -n "${GEMINI_CONFIG:-}" ]]; then
  final_args+=(--gemini_config "${GEMINI_CONFIG}")
fi

if [[ "${FINAL_RESUME:-0}" == "1" ]]; then
  final_args+=(--resume)
fi

python "${ANNOTATE_PUBMED_DIR}/final_data_clean_v2.py" "${final_args[@]}"

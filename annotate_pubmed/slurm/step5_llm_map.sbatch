#!/usr/bin/env bash
set -euo pipefail

: "${ANNOTATE_PUBMED_DIR:?ANNOTATE_PUBMED_DIR is required}"
: "${CONDA_SH:?CONDA_SH is required}"
: "${CONDA_ENV_PATH:?CONDA_ENV_PATH is required}"
: "${CROSSENCODED_DIR:?CROSSENCODED_DIR is required}"
: "${LLM_MODEL_DIR:?LLM_MODEL_DIR is required}"
: "${LLM_OUT_DIR:?LLM_OUT_DIR is required}"
: "${LLM_BATCH_SIZE:?LLM_BATCH_SIZE is required}"
: "${LLM_MAX_TOKENS:?LLM_MAX_TOKENS is required}"
: "${LLM_TEMPERATURE:?LLM_TEMPERATURE is required}"
: "${LLM_TOP_P:?LLM_TOP_P is required}"
: "${SHARD_INDEX:?SHARD_INDEX is required}"
: "${NUM_SHARDS:?NUM_SHARDS is required}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV_PATH}"
export PATH="${CONDA_PREFIX}/bin:$PATH"

llm_args=(
  --shards_dir "${CROSSENCODED_DIR}"
  --llm_model "${LLM_MODEL_DIR}"
  --out_dir "${LLM_OUT_DIR}"
  --batch_size "${LLM_BATCH_SIZE}"
  --max_tokens "${LLM_MAX_TOKENS}"
  --temperature "${LLM_TEMPERATURE}"
  --top_p "${LLM_TOP_P}"
  --shard_index "${SHARD_INDEX}"
  --num_shards "${NUM_SHARDS}"
)

if [[ -n "${LLM_MAX_MODEL_LEN:-}" ]]; then
  llm_args+=(--max_model_len "${LLM_MAX_MODEL_LEN}")
fi

python "${ANNOTATE_PUBMED_DIR}/llm_map.py" "${llm_args[@]}"

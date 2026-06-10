#!/usr/bin/env bash
set -euo pipefail

: "${ANNOTATE_PUBMED_DIR:?ANNOTATE_PUBMED_DIR is required}"
: "${CONDA_SH:?CONDA_SH is required}"
: "${CONDA_ENV_PATH:?CONDA_ENV_PATH is required}"
: "${PUBMED_PARQUET_DIR:?PUBMED_PARQUET_DIR is required}"
: "${BERT_PROCESSED_DIR:?BERT_PROCESSED_DIR is required}"
: "${BERT_MODEL_DIR:?BERT_MODEL_DIR is required}"
: "${SHARD_INDEX:?SHARD_INDEX is required}"
: "${NUM_SHARDS:?NUM_SHARDS is required}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV_PATH}"
export PATH="${CONDA_PREFIX}/bin:$PATH"

bert_args=(
  --input_dir "${PUBMED_PARQUET_DIR}"
  --processed_dir "${BERT_PROCESSED_DIR}"
  --bert_model "${BERT_MODEL_DIR}"
  --shard "${SHARD_INDEX}"
  --num_shards "${NUM_SHARDS}"
  --device "cuda:0"
)

if [[ -n "${SELECT_YEAR:-}" ]]; then
  bert_args+=(--select_year "${SELECT_YEAR}")
fi

python "${ANNOTATE_PUBMED_DIR}/bert_predict.py" "${bert_args[@]}"

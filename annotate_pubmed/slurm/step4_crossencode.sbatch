#!/usr/bin/env bash
set -euo pipefail

: "${ANNOTATE_PUBMED_DIR:?ANNOTATE_PUBMED_DIR is required}"
: "${CONDA_SH:?CONDA_SH is required}"
: "${CONDA_ENV_PATH:?CONDA_ENV_PATH is required}"
: "${BERT_POSITIVES_PATH:?BERT_POSITIVES_PATH is required}"
: "${G2P_CSV:?G2P_CSV is required}"
: "${CROSSENCODER_MODEL_DIR:?CROSSENCODER_MODEL_DIR is required}"
: "${CROSSENCODED_DIR:?CROSSENCODED_DIR is required}"
: "${SHARD_INDEX:?SHARD_INDEX is required}"
: "${NUM_SHARDS:?NUM_SHARDS is required}"
: "${TOP_K:?TOP_K is required}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV_PATH}"
export PATH="${CONDA_PREFIX}/bin:$PATH"

python "${ANNOTATE_PUBMED_DIR}/crossencode.py" \
  --input_parquet "${BERT_POSITIVES_PATH}" \
  --g2p_csv "${G2P_CSV}" \
  --model_path "${CROSSENCODER_MODEL_DIR}" \
  --out_dir "${CROSSENCODED_DIR}" \
  --device "cuda:0" \
  --top_k "${TOP_K}" \
  --shard "${SHARD_INDEX}" \
  --num_shards "${NUM_SHARDS}"

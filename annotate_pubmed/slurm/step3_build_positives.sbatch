#!/usr/bin/env bash
set -euo pipefail

: "${ANNOTATE_PUBMED_DIR:?ANNOTATE_PUBMED_DIR is required}"
: "${CONDA_SH:?CONDA_SH is required}"
: "${CONDA_ENV_PATH:?CONDA_ENV_PATH is required}"
: "${BERT_PROCESSED_DIR:?BERT_PROCESSED_DIR is required}"
: "${BERT_POSITIVES_PATH:?BERT_POSITIVES_PATH is required}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV_PATH}"
export PATH="${CONDA_PREFIX}/bin:$PATH"

python "${ANNOTATE_PUBMED_DIR}/build_bert_positives.py" \
  --processed_dir "${BERT_PROCESSED_DIR}" \
  --out_path "${BERT_POSITIVES_PATH}"

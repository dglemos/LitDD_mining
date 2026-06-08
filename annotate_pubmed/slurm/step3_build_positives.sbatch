#!/usr/bin/env bash
set -euo pipefail

: "${ANNOTATE_PUBMED_DIR:?ANNOTATE_PUBMED_DIR is required}"
: "${BERT_PROCESSED_DIR:?BERT_PROCESSED_DIR is required}"
: "${BERT_POSITIVES_PATH:?BERT_POSITIVES_PATH is required}"

source activate g2p-llm

python "${ANNOTATE_PUBMED_DIR}/build_bert_positives.py" \
  --processed_dir "${BERT_PROCESSED_DIR}" \
  --out_path "${BERT_POSITIVES_PATH}"

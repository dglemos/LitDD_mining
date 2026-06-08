#!/usr/bin/env bash
set -euo pipefail

: "${ANNOTATE_PUBMED_DIR:?ANNOTATE_PUBMED_DIR is required}"
: "${PUBMED_DOWNLOAD_DIR:?PUBMED_DOWNLOAD_DIR is required}"
: "${PUBMED_PARQUET_DIR:?PUBMED_PARQUET_DIR is required}"

source activate g2p-llm

download_args=(
  --home_dir "${PUBMED_DOWNLOAD_DIR}"
  --output_dir "${PUBMED_PARQUET_DIR}"
)

if [[ -n "${SINCE_DATE:-}" ]]; then
  download_args+=(--since_date "${SINCE_DATE}")
fi

python "${ANNOTATE_PUBMED_DIR}/download_pubmed.py" "${download_args[@]}"

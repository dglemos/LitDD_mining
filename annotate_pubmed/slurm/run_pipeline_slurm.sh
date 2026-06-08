#!/usr/bin/env bash
set -euo pipefail

# Submit the annotate_pubmed pipeline to SLURM with strict step ordering.
# Edit variables below to match your environment.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" # LitDD_mining/annotate_pubmed/slurm
DEFAULT_ANNOTATE_PUBMED_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)" # LitDD_mining/annotate_pubmed
DEFAULT_WORK_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)" # LitDD_mining
SBATCH_DIR="${SCRIPT_DIR}" # LitDD_mining/annotate_pubmed/slurm

abspath_existing() {
  local path="$1"
  if [[ -d "${path}" ]]; then
    (cd "${path}" && pwd)
    return
  fi

  local parent
  parent="$(cd "$(dirname "${path}")" && pwd)"
  printf '%s/%s\n' "${parent}" "$(basename "${path}")"
}

# ---- User config ----
# Paths
# Directory containing the annotate_pubmed Python scripts.
ANNOTATE_PUBMED_DIR="${ANNOTATE_PUBMED_DIR:-${DEFAULT_ANNOTATE_PUBMED_DIR}}"
# Working directory for job execution and default output locations.
WORK_DIR="${WORK_DIR:-${DEFAULT_WORK_DIR}}"
PUBMED_DOWNLOAD_DIR="${PUBMED_DOWNLOAD_DIR:-${WORK_DIR}/download}"
PUBMED_PARQUET_DIR="${PUBMED_PARQUET_DIR:-${WORK_DIR}/download/parquet_download_files}"
# Optional: only download PubMed files whose remote Last-Modified date is on or after YYYY-MM-DD.
SINCE_DATE="${SINCE_DATE:-}"
# Optional: override the default publication-year filter used by bert_predict.py.
SELECT_YEAR="${SELECT_YEAR:-}"
BERT_PROCESSED_DIR="${BERT_PROCESSED_DIR:-${WORK_DIR}/bert_processed}"
BERT_POSITIVES_PATH="${BERT_POSITIVES_PATH:-${WORK_DIR}/pubmed_bert_positive.parquet}"
CROSSENCODED_DIR="${CROSSENCODED_DIR:-${WORK_DIR}/crossencoded_shards}"
LLM_OUT_DIR="${LLM_OUT_DIR:-${WORK_DIR}/llm_outputs}"
FINAL_OUT_CSV="${FINAL_OUT_CSV:-${FINAL_OUT_PATH:-${WORK_DIR}/final_tiab_mappings.csv}}"

# Models / inputs
BERT_MODEL_DIR="${BERT_MODEL_DIR:-/path/to/LitDD_BERT}"
CROSSENCODER_MODEL_DIR="${CROSSENCODER_MODEL_DIR:-/path/to/LitDD_crossencoder}"
# vLLM model identifier or local model path for llm_map.py.
LLM_MODEL_DIR="${LLM_MODEL_DIR:-/path/to/llm_model}"
G2P_CSV="${G2P_CSV:-/path/to/ddg2p.csv}"
GENE2PUBTATOR3="${GENE2PUBTATOR3:-/path/to/gene2pubtator3}"

# LLM params
LLM_BATCH_SIZE="${LLM_BATCH_SIZE:-32}"
LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-64}"
LLM_TEMPERATURE="${LLM_TEMPERATURE:-0.0}"
LLM_TOP_P="${LLM_TOP_P:-1.0}"
# Optional: pass through vLLM max_model_len to llm_map.py.
LLM_MAX_MODEL_LEN="${LLM_MAX_MODEL_LEN:-32768}"

# Cross-encoder params
TOP_K="${TOP_K:-5}"

# Final filtering
SCORE_CUTOFF="${SCORE_CUTOFF:-0.9}"
GEMINI_CONFIG="${GEMINI_CONFIG:-}"
FINAL_RESUME="${FINAL_RESUME:-0}"

# Optional sharding for steps that support it
SHARD_INDEX="${SHARD_INDEX:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"

# SLURM settings (edit for your cluster)
SLURM_ACCOUNT="${SLURM_ACCOUNT:-my_account}"
SLURM_PARTITION_CPU="${SLURM_PARTITION_CPU:-cpu}"
SLURM_PARTITION_GPU="${SLURM_PARTITION_GPU:-gpu}"
CPUS="${CPUS:-4}"
MEM_CPU="${MEM_CPU:-16G}"
MEM_CPU_LARGE="${MEM_CPU_LARGE:-48G}"
MEM_GPU="${MEM_GPU:-48G}"
TIME_CPU_SHORT="${TIME_CPU_SHORT:-02:00:00}"
TIME_CPU_LONG="${TIME_CPU_LONG:-08:00:00}"
TIME_GPU="${TIME_GPU:-12:00:00}"
GPUS="${GPUS:-1}"

# ---- End user config ----

ANNOTATE_PUBMED_DIR="$(abspath_existing "${ANNOTATE_PUBMED_DIR}")"
WORK_DIR="$(abspath_existing "${WORK_DIR}")"
PUBMED_DOWNLOAD_DIR="$(abspath_existing "${PUBMED_DOWNLOAD_DIR}")"
PUBMED_PARQUET_DIR="$(abspath_existing "${PUBMED_PARQUET_DIR}")"
BERT_PROCESSED_DIR="$(abspath_existing "${BERT_PROCESSED_DIR}")"
BERT_POSITIVES_PATH="$(abspath_existing "${BERT_POSITIVES_PATH}")"
CROSSENCODED_DIR="$(abspath_existing "${CROSSENCODED_DIR}")"
LLM_OUT_DIR="$(abspath_existing "${LLM_OUT_DIR}")"
FINAL_OUT_CSV="$(abspath_existing "${FINAL_OUT_CSV}")"
BERT_MODEL_DIR="$(abspath_existing "${BERT_MODEL_DIR}")"
CROSSENCODER_MODEL_DIR="$(abspath_existing "${CROSSENCODER_MODEL_DIR}")"
LLM_MODEL_DIR="$(abspath_existing "${LLM_MODEL_DIR}")"
G2P_CSV="$(abspath_existing "${G2P_CSV}")"
GENE2PUBTATOR3="$(abspath_existing "${GENE2PUBTATOR3}")"
LOG_DIR="${LOG_DIR:-${WORK_DIR}/logs}"
LOG_DIR="$(abspath_existing "${LOG_DIR}")"

mkdir -p "${LOG_DIR}"

export ANNOTATE_PUBMED_DIR PUBMED_DOWNLOAD_DIR PUBMED_PARQUET_DIR SINCE_DATE SELECT_YEAR
export BERT_PROCESSED_DIR BERT_POSITIVES_PATH CROSSENCODED_DIR LLM_OUT_DIR FINAL_OUT_CSV
export BERT_MODEL_DIR CROSSENCODER_MODEL_DIR LLM_MODEL_DIR G2P_CSV GENE2PUBTATOR3
export LLM_BATCH_SIZE LLM_MAX_TOKENS LLM_TEMPERATURE LLM_TOP_P LLM_MAX_MODEL_LEN TOP_K SCORE_CUTOFF
export GEMINI_CONFIG FINAL_RESUME
export SHARD_INDEX NUM_SHARDS

submit_job() {
  local name="$1"
  local partition="$2"
  local cpus="$3"
  local mem="$4"
  local time="$5"
  local gpus="${6:-0}"
  local dep="${7:-}"
  local script="$8"

  local args=(
    "--job-name=${name}"
    "--partition=${partition}"
    "--account=${SLURM_ACCOUNT}"
    "--cpus-per-task=${cpus}"
    "--mem=${mem}"
    "--time=${time}"
    "--output=${LOG_DIR}/${name}.%j.out"
    "--error=${LOG_DIR}/${name}.%j.err"
  )

  if [[ "${gpus}" -gt 0 ]]; then
    args+=("--gres=gpu:${gpus}")
  fi

  if [[ -n "${dep}" ]]; then
    args+=("--dependency=afterok:${dep}")
  fi

  sbatch "${args[@]}" "${script}" | awk '{print $4}'
}

echo "Submitting pipeline jobs..."

jid1=$(submit_job "pubmed_download" "${SLURM_PARTITION_CPU}" "${CPUS}" "${MEM_CPU}" "${TIME_CPU_LONG}" 0 "" "${SBATCH_DIR}/step1_download.sbatch")
jid2=$(submit_job "bert_predict" "${SLURM_PARTITION_GPU}" "${CPUS}" "${MEM_GPU}" "${TIME_GPU}" "${GPUS}" "${jid1}" "${SBATCH_DIR}/step2_bert_predict.sbatch")
jid3=$(submit_job "bert_positives" "${SLURM_PARTITION_CPU}" "${CPUS}" "${MEM_CPU}" "${TIME_CPU_SHORT}" 0 "${jid2}" "${SBATCH_DIR}/step3_build_positives.sbatch")
jid4=$(submit_job "crossencode" "${SLURM_PARTITION_GPU}" "${CPUS}" "${MEM_GPU}" "${TIME_GPU}" "${GPUS}" "${jid3}" "${SBATCH_DIR}/step4_crossencode.sbatch")
jid5=$(submit_job "llm_map" "${SLURM_PARTITION_GPU}" "${CPUS}" "${MEM_GPU}" "${TIME_GPU}" "${GPUS}" "${jid4}" "${SBATCH_DIR}/step5_llm_map.sbatch")
jid6=$(submit_job "final_clean" "${SLURM_PARTITION_CPU}" "${CPUS}" "${MEM_CPU_LARGE}" "${TIME_CPU_LONG}" 0 "${jid5}" "${SBATCH_DIR}/step6_final_clean.sbatch")

echo "Submitted jobs:"
echo "  1) pubmed_download     ${jid1}"
echo "  2) bert_predict        ${jid2}"
echo "  3) bert_positives      ${jid3}"
echo "  4) crossencode         ${jid4}"
echo "  5) llm_map             ${jid5}"
echo "  6) final_clean         ${jid6}"
echo "Logs: ${LOG_DIR}"

# Annotate PubMed Pipeline (SLURM)

This pipeline runs the `annotate_pubmed` scripts in order on SLURM with strict
`afterok` dependencies. GPU steps are:
- `bert_predict.py`
- `crossencode.py`
- `llm_map.py`

All other steps run on CPU.

## 1) Edit the driver config

Open `annotate_pubmed/slurm/run_pipeline_slurm.sh` and set:
- `ANNOTATE_PUBMED_DIR` for the directory containing the `annotate_pubmed`
  Python scripts
- `WORK_DIR` for the pipeline working directory, logs, and default output base
- All input/output paths (PubMed download/parquet dirs, model dirs, etc.)
- Optional `SINCE_DATE` in `YYYY-MM-DD` format to download only files with a
  remote `Last-Modified` date on or after that date
- Optional `SELECT_YEAR` to override the default publication-year filter used
  by `bert_predict.py`
- `LLM_MODEL_DIR` to the vLLM model identifier or local model path you want to use
- Optional `LLM_MAX_MODEL_LEN` to pass `--max_model_len` through to `llm_map.py`
- SLURM account/partition names and resource sizes
- Optional sharding (`SHARD_INDEX`, `NUM_SHARDS`)
- Optional Gemini final-clean settings (`GEMINI_CONFIG`, `FINAL_RESUME`)

The final step now writes a CSV via `FINAL_OUT_CSV`.
By default, `ANNOTATE_PUBMED_DIR` resolves to the `annotate_pubmed` directory
next to `slurm`, and `WORK_DIR` resolves to the repo root.
Configured paths are normalized to absolute paths by `run_pipeline_slurm.sh`
before jobs are submitted.
Default data/output paths are built from `WORK_DIR`, including:
- `PUBMED_DOWNLOAD_DIR=${WORK_DIR}/download`
- raw PubMed XML files in `${WORK_DIR}/download/raw_download_files`
- `PUBMED_PARQUET_DIR=${WORK_DIR}/download/parquet_download_files`
- `BERT_PROCESSED_DIR=${WORK_DIR}/bert_processed`
- `CROSSENCODED_DIR=${WORK_DIR}/crossencoded_shards`
- `LLM_OUT_DIR=${WORK_DIR}/llm_outputs`

## 2) Submit the pipeline

From the repo root:

```bash
bash annotate_pubmed/slurm/run_pipeline_slurm.sh
```

The script submits 6 jobs in sequence and prints each SLURM job ID.
Logs are written to `${WORK_DIR}/logs` by default unless `LOG_DIR` is set.
The sbatch wrappers use absolute paths for scripts and configured inputs/outputs,
so they do not depend on changing into `WORK_DIR` first.

## 3) Conda environment

Each SLURM step script runs:

```bash
source activate g2p-llm
```

If your cluster requires initializing conda (e.g. `source /path/to/conda.sh`),
add that line above `source activate g2p-llm` in each step script under
`annotate_pubmed/slurm/`.

## 4) Pipeline steps (in order)

1. `download_pubmed.py`
2. `bert_predict.py` (GPU)
3. `build_bert_positives.py`
4. `crossencode.py` (GPU)
5. `llm_map.py` (GPU)
6. `final_data_clean_v2.py`

The corresponding SLURM wrappers are:
- `step1_download.sbatch`
- `step2_bert_predict.sbatch`
- `step3_build_positives.sbatch`
- `step4_crossencode.sbatch`
- `step5_llm_map.sbatch`
- `step6_final_clean.sbatch`

`download_pubmed.py` now performs XML-to-parquet conversion by default, so the
old standalone `pubmed_to_parquet.py` step is no longer part of the pipeline.
Step 1 now exits nonzero if any download or parquet conversion fails, so
downstream jobs will not run on partial data.

Steps 1 to 5 still match the current script CLIs directly. Step 6 now uses
`final_data_clean_v2.py`, which expects a single `*__llm.parquet` file rather
than a directory of shard outputs. The SLURM wrapper therefore looks for exactly
one LLM parquet in `LLM_OUT_DIR` and fails fast if multiple files are present.

By default, step 6 runs `final_data_clean_v2.py` without Gemini. If
`GEMINI_CONFIG` is set in `run_pipeline_slurm.sh`, step 6 also passes
`--gemini_config` to `final_data_clean_v2.py`. Set `FINAL_RESUME=1` to add
`--resume`, which is recommended only when Gemini analysis is enabled because
that step can be slow.

That means the current pipeline is aligned with the scripts for the common
single-output case, but the old end-to-end sharded final merge behavior is no
longer available in step 6.

## 5) Common tweaks

- Change GPU/CPU resources in `run_pipeline_slurm.sh`.
- Increase `LLM_MAX_TOKENS` or `LLM_BATCH_SIZE` for larger LLM context/throughput.
- Use sharding by setting `NUM_SHARDS > 1` and running multiple submissions with
  different `SHARD_INDEX` values.

## 6) Troubleshooting

- If a step fails, downstream jobs will not start (dependency is `afterok`).
- Check logs in `${WORK_DIR}/logs` by default, or in `LOG_DIR` if you set it.

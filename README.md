# LitDD Mining
Pipeline to map PubMed literature to DD G2P disease threads and extract
TIAB (title + abstract) mappings using BERT, a CrossEncoder, and an LLM.

## Overview
The workflow runs in stages:
1) Download PubMed baseline and update XML files.
2) Convert XML to parquet.
3) Run BERT filtering to keep likely DD/G2P-relevant TIABs.
4) Keep only BERT positives.
5) Cross-encode against G2P LGMDE threads to get top-K candidates.
6) Use an LLM to select the best G2P ID(s) from those candidates.
7) Clean and enrich the results with PubTator and NCBI gene info.

## Requirements
- Python 3.9+ recommended
- GPU recommended for BERT, CrossEncoder, and LLM steps
- External tools: `wget` for download step

Python dependencies (typical set):
```bash
pip install torch transformers datasets pyarrow polars pandas numpy pubmed_parser lxml requests sentence-transformers vllm evaluate scikit-learn
```

## Setup
Before running, update the placeholder paths in the scripts:
- `path_to_pubmed_download_dir`
- `path_to_pubmed_download/raw_download_files`
- `path_to_pubmed_download/parquet_download_files`
- `path_to_lit_dd_BERT`
- `path_to_litdd_crossencoder`
- `path_to_ddg2p_csv`
- `path_to_gene_info`

Search for `path_to_` in:
- `annotate_pubmed/download_pubmed.py`
- `annotate_pubmed/pubmed_to_parquet.py`
- `annotate_pubmed/bert_predict.py`
- `annotate_pubmed/crossencode.py`
- `annotate_pubmed/llm_map.py`
- `annotate_pubmed/final_data_clean.py`

## Run the pipeline

### 1) Download PubMed XML
Edit `home_dir` in `annotate_pubmed/download_pubmed.py`, then run:
```bash
python annotate_pubmed/download_pubmed.py
```
Outputs to `path_to_pubmed_download_dir/raw_download_files`.

### 2) Convert XML to parquet
Edit `download_dir` and `output_dir` in `annotate_pubmed/pubmed_to_parquet.py`, then run:
```bash
python annotate_pubmed/pubmed_to_parquet.py
```
Outputs to `path_to_pubmed_download/parquet_download_files`.

### 3) Run BERT prediction over parquet
Edit `BERT_MODEL_PATH` and `INPUT_DIR` in `annotate_pubmed/bert_predict.py`, then run:
```bash
python annotate_pubmed/bert_predict.py --device cuda:0
```
Outputs to `bert_processed/` (or the path in `PROCESSED_DIR`).

### 4) Build positives parquet
```bash
python annotate_pubmed/build_bert_positives.py \
  --processed_dir bert_processed \
  --out_path pubmed_bert_positive.parquet
```

### 5) Cross-encode against G2P
```bash
python annotate_pubmed/crossencode.py \
  --input_parquet pubmed_bert_positive.parquet \
  --g2p_csv /path/to/ddg2p.csv \
  --model_path /path/to/litdd_crossencoder \
  --out_dir crossencoded_shards \
  --device cuda:0
```

### 6) LLM mapping
```bash
python annotate_pubmed/llm_map.py \
  --shards_dir crossencoded_shards \
  --llm_model /path/to/llm \
  --out_dir llm_shards \
  --batch_size 32 \
  --save_every 1000
```

### 7) Final cleaning and enrichment
Edit the input parquet paths and resource locations in
`annotate_pubmed/final_data_clean.py`, then run:
```bash
python annotate_pubmed/final_data_clean.py
```
Outputs `final_tiab_mappings.parquet`.

## Optional: model training helpers
These scripts are for building and training the models used above:
- `train_test/final_traintest_dataset.py`: build HF datasets from `annotated_tiab.csv`
- `train_test/mine_hard_negatives.py`: mine hard negatives
- `train_test/bert_finetune.py`: finetune BERT classifier
- `train_test/crossencode_finetune.py`: finetune CrossEncoder

## Notes
- BERT filtering enforces English-only and `pubdate > 1980`.
- Cross-encoding and LLM steps can be sharded via `--shard`/`--num_shards`.
- The LLM prompt is strict: only returns IDs from top-5 candidates or `NO MATCH`.

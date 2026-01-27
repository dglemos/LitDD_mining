# LitDD Mining
Pipeline to map PubMed literature to DD G2P disease threads and extract
TIAB (title + abstract) mappings using BERT, a CrossEncoder and an LLM.

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
pip install -r requirements.txt
```

## Setup models
To avoid failures while loading the model from huggingface,
pre-download the models of your preference to a local directory.

### BERT
Model example: `tmy100000001/LitDD_BERT`
```bash
hf download tmy100000001/LitDD_BERT \
  --local-dir <path_to_models> \
  --cache-dir <path_to_models_cache>
```

### CrossEncoder
Model example: `tmy100000001/LitDD_crossencoder`
```bash
hf download tmy100000001/LitDD_crossencoder \
  --local-dir <path_to_models> \
  --cache-dir <path_to_models_cache>
```

### LLM
```bash
hf download <model_name> \
  --local-dir <path_to_models> \
  --cache-dir <path_to_models_cache>
```

## Run the pipeline

### 1) Download PubMed XML
```bash
python annotate_pubmed/download_pubmed.py --home_dir <path_to_pubmed_download_dir>
```
Outputs to `path_to_pubmed_download_dir/raw_download_files`.

### 2) Convert XML to parquet
```bash
python annotate_pubmed/pubmed_to_parquet.py --download_dir <path_to_pubmed_raw_download_dir> \
--output_dir <path_to_pubmed_parquet_dir>
```

### 3) Run BERT prediction over parquet
```bash
python annotate_pubmed/bert_predict.py --input_dir <path_to_pubmed_parquet_dir> \
--processed_dir <bert_processed_dir> \
--bert_model <path_to_bert_model>
```
Outputs to `bert_processed_dir`

### 4) Build positives parquet
```bash
python annotate_pubmed/build_bert_positives.py \
  --processed_dir <bert_processed_dir> \
  --out_path pubmed_bert_positive.parquet
```
Outputs to file `pubmed_bert_positive.parquet`

### 5) Cross-encode against G2P
```bash
python annotate_pubmed/crossencode.py \
  --input_parquet pubmed_bert_positive.parquet \
  --g2p_csv <path_to_ddg2p.csv> \
  --model_path <path_to_litdd_crossencoder> \
  --out_dir <path_to_crossencoded_shards> \
  --device cuda:0
```

### 6) LLM mapping
```bash
python annotate_pubmed/llm_map.py \
  --shards_dir <path_to_crossencoded_shards> \
  --llm_model <path_to_llm> \
  --out_dir <path_to_output_dir> \
  --batch_size 32 --max_tokens 256 \
  --temperature 0.0 --top_p 1.0
```

### 7) Final cleaning and enrichment
```bash
python annotate_pubmed/final_data_clean.py
```

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

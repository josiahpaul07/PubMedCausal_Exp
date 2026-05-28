# PUBMEDCAUSAL: Biomedical Causal Relation Extraction and Detection

A comprehensive benchmark suite for causal relation extraction and detection on biomedical text, featuring 30,000 annotated PubMed sentences with extensive evaluation across BERT models, LLMs (3B-70B), and transfer learning experiments.

## Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Dataset](#dataset)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
  - [1. Data Preparation](#1-data-preparation)
  - [2. BERT Detection Baselines](#2-bert-detection-baselines)
  - [3. LLM Inference (Zero-shot)](#3-llm-inference-zero-shot)
  - [4. LLM Fine-tuning](#4-llm-fine-tuning)
  - [5. Transfer Learning](#5-transfer-learning)
  - [6. Ablation Studies](#6-ablation-studies)
  - [7. Error Analysis](#7-error-analysis)
  - [8. Results Analysis](#8-results-analysis)
- [Evaluation Metrics](#evaluation-metrics)
- [Parameters Used](#parameters-used)

---

## Overview

This repository contains the complete codebase for benchmarking biomedical causal relation extraction and detection. It supports:

- **Two tasks**: Causal detection (binary classification) and extraction (cause-effect span identification with attributes)
- **Three task splits**: Combined, X-only (multiple pairs), Y-only (single pair)
- **Multiple model families**: BERT variants, LLMs (3B-70B), with LoRA fine-tuning
- **Six prompting strategies**: zero-shot, few-shot, CoT, CoT+few-shot, ReAct, least-to-most
- **Transfer learning**: Cross-dataset evaluation on biocause1, fincausal, altlex, CTB datasets
- **Three-tier evaluation**: Exact F1, Token F1, Cosine F1 (semantic similarity)

---

## Repository Structure

```
PUBMEDCAUSAL/
│
├── 30k_train.json                      # Raw training data (15,000 sentences)
├── 30k_test.json                       # Raw test data (15,000 sentences)
│
├── data/                               # Data preparation scripts
│   ├── prepare_detection_data.py       # Creates detection dataset
│   └── prepared/
│       ├── detection_train.json        # Prepared detection training (15K)
│       ├── detection_test.json         # Prepared detection test (15K)
│       ├── extraction_combined/        # All causal sentences
│       │   ├── train.json
│       │   └── test.json
│       ├── extraction_X_only/          # Multiple cause-effect pairs
│       │   ├── train.json
│       │   └── test.json
│       ├── extraction_Y_only/          # Single cause-effect pair
│       │   ├── train.json
│       │   └── test.json
│       └── prepare_extraction_data.py
│
├── models/                             # Model implementations
│   ├── bert_detection.py               # BERT detection (BERT, SciBERT, BioBERT, PubMedBERT)
│   ├── llm_finetune.py                 # LoRA fine-tuning for LLMs
│   └── llm_inference.py                # LLM inference engine
│
├── prompts/                            # Prompt templates
│   ├── detection_prompts.py            # 6 strategies for detection
│   └── extraction_prompts.py           # 6 strategies for extraction
│
├── evaluation/                         # Evaluation metrics
│   ├── detection_metrics.py            # P/R/F1/Accuracy
│   └── extraction_metrics.py           # Exact/Token/Cosine F1 + Causality/Sententiality
│
├── tl_data/                            # Transfer learning datasets
│   ├── altlex_train.csv
│   ├── biocause1.csv
│   ├── ctb_train.csv
│   └── fincausal.csv
│
├── ablation/                           # Ablation study
│   ├── sentential_causal.py            # Sententiality analysis
│   ├── ablation.log
│   └── *.csv                           # Results
│
├── error_bucket/                       # Error analysis
│   ├── two_file_match_flags.py         # Compare predictions vs ground truth
│   └── two_file_match_flag_outputs/
│       └── exact_first_error_bucketization_v2.py
│
├── run_detection.py                    # Main script: LLM detection
├── run_extraction.py                   # Main script: LLM extraction
├── run_causal_infer.py                 # Inference on external CSVs
├── run_re_evaluate.py                  # Re-evaluate predictions with new metrics
├── transfer_learning.py                # Cross-dataset transfer learning
├── analyze_results.py                  # Aggregate and analyze results
└── finetune_all_5_models_nohup.sh      # Production fine-tuning script
```

---

## Setup

### Environment

```bash
# Create virtual environment
python3 -m venv env
source env/bin/activate

# Install dependencies
pip install torch transformers datasets peft bitsandbytes accelerate
pip install sentence-transformers scikit-learn pandas tqdm numpy
```

### Directory Setup

```bash
# Create required directories
mkdir -p logs checkpoints results
mkdir -p results/finetuned_models results/detection results/extraction
```

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 1× 16GB | 1-4× A100 40/80GB |
| RAM | 32GB | 64GB+ |
| Storage | 50GB | 200GB+ |

---

## Dataset

### Raw Data Format (30k_train.json / 30k_test.json) found in https://huggingface.co/datasets/jaypee01/PubMedCausal/tree/main

Each sample contains a sentence and up to 16 cause-effect pairs:

```json
{
  "s/n": 6350,
  "Sentence": "Insulin resistance causes hyperglycemia.",
  "Cause 1": "Insulin resistance",
  "Effect 1": "hyperglycemia",
  "Sententiality 1": "Intra",
  "Causality 1": "Explicit",
  "Cause 2": "",
  ...
}
```

### Prepared Data Formats

**Detection** (binary classification):
```json
{
  "s/n": 6350,
  "sentence": "Insulin resistance causes hyperglycemia.",
  "label": 1
}
```

**Extraction** (span + attributes):
```json
{
  "s/n": 17500,
  "sentence": "Insulin resistance causes hyperglycemia.",
  "pairs": [
    {
      "cause": "Insulin resistance",
      "effect": "hyperglycemia",
      "sententiality": "Intra",
      "causality": "Explicit"
    }
  ],
  "num_pairs": 1
}
```

### Dataset Statistics

| Task | Split | Train | Test |
|------|-------|-------|------|
| Detection | - | 15,000 | 15,000 |
| Extraction | Combined | 1,972 | 1,973 |
| Extraction | X_only (multi-pair) | 1,206 | 1,207 |
| Extraction | Y_only (single-pair) | 766 | 766 |

---

## Quick Start

### Run Complete Fine-Tuning Pipeline

```bash
# Run all 5 LLMs across all 3 extraction splits
bash finetune_all_5_models_nohup.sh
```

This script fine-tunes Llama-3B, Llama-8B, Mistral-7B, Qwen-7B, and DeepSeek-7B on all 3 extraction splits, then evaluates them.

### Run BERT Detection Baselines

```bash
CUDA_VISIBLE_DEVICES=0 nohup bash -c 'for model in bert scibert pubmedbert biobert; do
    python models/bert_detection.py \
        --model_name $model \
        --train_file ./data/prepared/detection_train.json \
        --test_file ./data/prepared/detection_test.json \
        --output_dir ./checkpoints/detection_${model}
done' > ./logs/bert_detection.log 2>&1 &
```

---

## Detailed Usage

### 1. Data Preparation

If you need to regenerate prepared data from the raw 30K files:

```bash
# Prepare detection data
python data/prepare_detection_data.py

# Prepare extraction data (all 3 splits)
python data/prepared/prepare_extraction_data.py
```

### 2. BERT Detection Baselines

Train BERT-based models (BERT, SciBERT, BioBERT, PubMedBERT) for binary causal detection:

```bash
# Single model
python models/bert_detection.py \
    --model_name pubmedbert \
    --train_file ./data/prepared/detection_train.json \
    --test_file ./data/prepared/detection_test.json \
    --output_dir ./checkpoints/detection_pubmedbert \
    --num_epochs 3 \
    --batch_size 16 \
    --learning_rate 2e-5

# All 4 models sequentially (background)
CUDA_VISIBLE_DEVICES=0 nohup bash -c 'for model in bert scibert pubmedbert biobert; do
    python models/bert_detection.py \
        --model_name $model \
        --train_file ./data/prepared/detection_train.json \
        --test_file ./data/prepared/detection_test.json \
        --output_dir ./checkpoints/detection_${model}
done' > ./logs/bert_detection_all.log 2>&1 &
```

### 3. LLM Inference (Zero-shot)

#### Detection

```bash
# Single model with multiple strategies
python run_detection.py \
    --test_file ./data/prepared/detection_test.json \
    --models llama-8b \
    --strategies zero-shot few-shot cot cot-fewshot react least-to-most \
    --results_dir ./results/detection

# Large models (32B+, 70B)
CUDA_VISIBLE_DEVICES=0,1 nohup python run_detection.py \
    --test_file ./data/prepared/detection_test.json \
    --models deepseek-r1-distill-qwen-32b meta-llama-3.3-70b deepseek-70b mixtral-8x7b \
    --strategies zero-shot few-shot cot \
    --batch_size 4 \
    --results_dir ./results/detection > ./logs/llm_detection_large.log 2>&1 &
```

#### Extraction

```bash
# Single model, single split
python run_extraction.py \
    --test_file ./data/prepared/extraction_combined/test.json \
    --split_type combined \
    --models llama-8b \
    --strategies zero-shot few-shot cot \
    --results_dir ./results/extraction

# All large models, all splits, all strategies (production)
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup bash -c 'for split in combined X_only Y_only; do
    python run_extraction.py \
        --test_file ./data/prepared/extraction_${split}/test.json \
        --split_type $split \
        --models deepseek-r1-distill-qwen-32b meta-llama-3.3-70b deepseek-70b mixtral-8x7b \
        --strategies zero-shot few-shot cot cot-fewshot react least-to-most \
        --batch_size 2 \
        --results_dir ./results/extraction
done' > ./logs/llm_extraction_all.log 2>&1 &
```

### 4. LLM Fine-Tuning

#### Single Model Fine-tuning

```bash
# Detection
python models/llm_finetune.py \
    --model_name llama-3b \
    --task_type detection \
    --train_file ./data/prepared/detection_train.json \
    --test_file ./data/prepared/detection_test.json \
    --output_dir ./checkpoints/detection_llama3b \
    --num_epochs 3 \
    --batch_size 8

# Extraction
python models/llm_finetune.py \
    --model_name llama-3b \
    --task_type extraction \
    --train_file ./data/prepared/extraction_combined/train.json \
    --test_file ./data/prepared/extraction_combined/test.json \
    --output_dir ./checkpoints/extraction_combined_llama3b \
    --num_epochs 3 \
    --batch_size 8
```

#### Evaluating Fine-tuned Models

```bash
# Test fine-tuned model
python run_extraction.py \
    --test_file ./data/prepared/extraction_combined/test.json \
    --split_type combined \
    --models llama-3b-finetuned \
    --model_path ./checkpoints/extraction_combined_llama3b/final \
    --strategies zero-shot \
    --batch_size 16 \
    --results_dir ./results/finetuned_models
```

#### Fine-tune ALL Models, ALL Splits (Production)

```bash
# Run the complete fine-tuning pipeline
bash finetune_all_5_models_nohup.sh
```

This runs 5 models × 3 splits = 15 fine-tuning + evaluation jobs.

### 5. Transfer Learning

Tests how well models trained on PubMedCausal generalize to other causal datasets:

```bash
# Run full transfer learning pipeline (within + cross-dataset)
CUDA_VISIBLE_DEVICES=0 nohup python transfer_learning.py \
    > ./logs/transfer_learning.log 2>&1 &
```

**Default datasets evaluated**:
- biocause1.csv (biomedical)
- fincausal.csv (financial)
- altlex_train.csv (general)
- ctb_train.csv (CausalTimeBank)
- PubMedCausal detection_train.json (cross-source)

**Default models**:
- bert-base-uncased
- SciBERT
- PubMedBERT
- BioBERT

**Outputs**: `./results/balanced_experiment/`
- `models/<dataset>/<model>/final/` - Trained models
- `predictions/` - Per-dataset predictions
- `metrics_summary.csv` - All metrics

### 6. Ablation Studies

Analyze sententiality (intra vs inter-sentential) impact:

```bash
python ablation/sentential_causal.py \
    --gold_file ablation/test.json \
    --pred_file ablation/extraction_combined_DeepSeek-R1-Distill-Qwen-32B_few-shot_20260308_184255.json \
    --output_dir ./ablation/results
```

### 7. Error Analysis

Bucket errors into categories (exact match, token overlap, cosine similarity):

```bash
# Without cosine (fast)
python error_bucket/two_file_match_flags.py \
    --pred error_bucket/extraction_combined_DeepSeek-R1-Distill-Qwen-32B_few-shot_20260308_184255.json \
    --gold error_bucket/test.json \
    --out_dir error_bucket/two_file_match_flag_outputs

# With BioSentBERT cosine similarity
python error_bucket/two_file_match_flags.py \
    --pred error_bucket/predictions.json \
    --gold error_bucket/test.json \
    --out_dir error_bucket/outputs \
    --cosine \
    --cosine_threshold 0.75
```

### 8. Results Analysis

#### Aggregate Results

```bash
# Compare all results
python analyze_results.py --results_dir ./results

# Outputs aggregated CSV with all model × strategy × split combinations
```

#### Re-evaluate Predictions

If you want to re-run evaluation with different metrics (e.g., add cosine F1 to old runs):

```bash
# Without cosine
python run_re_evaluate.py \
    --pred_dir ./results/extraction \
    --gt_dir ./data/prepared/extraction_combined \
    --out_csv ./results/final_eval_summary.csv

# With BioSentBERT cosine
python run_re_evaluate.py \
    --pred_dir ./results/extraction \
    --gt_dir ./data/prepared/extraction_combined \
    --out_csv ./results/final_eval_summary.csv \
    --cosine
```

#### Test Fine-tuned Models on External CSVs

```bash
# BERT detection on external CSV
python run_causal_infer.py \
    --input_csv tl_data/biocause1.csv \
    --model_dirs ./checkpoints/detection_scibert/final \
    --model_type bert \
    --task detection

# Fine-tuned LLM extraction
python run_causal_infer.py \
    --input_csv tl_data/biocause1.csv \
    --model_dirs ./checkpoints/extraction_combined_llama3b/final \
    --model_type llm \
    --task extraction \
    --strategy zero-shot \
    --batch_size 8 \
    --cosine \
    --cosine_threshold 0.75
```

---

## Evaluation Metrics

### Detection Metrics
- **Accuracy**: Overall correctness
- **Precision/Recall/F1**: Binary (causal class) and Macro (both classes)
- **Confusion Matrix**: TN, FP, FN, TP

### Extraction Metrics (3-tier framework)

**Tier 1: Exact F1** (strict)
- Cause F1, Effect F1, Pair F1
- Requires exact string match after normalization

**Tier 2: Token F1** (primary)
- SQuAD-style token overlap
- Handles minor boundary variations
- Cause F1, Effect F1, Pair F1

**Tier 3: Cosine F1** (semantic)
- BioSentBERT embeddings
- Similarity threshold: 0.75
- Captures paraphrased extractions

**Additional Metrics**:
- Causality F1 (Explicit vs Implicit classification)
- Sententiality F1 (Intra vs Inter-sentential classification)
- Switching Rate (cause/effect swap errors)

---

## Parameters Used

### BERT Detection
| Parameter | Value |
|-----------|-------|
| Learning rate | 2×10⁻⁵ |
| Batch size | 16 |
| Epochs | 3 |
| Max sequence length | 128 |
| Models | BERT, SciBERT, PubMedBERT, BioBERT |

### LLM Fine-tuning (LoRA)
| Parameter | Value |
|-----------|-------|
| LoRA rank (r) | 8 |
| LoRA alpha (α) | 16 |
| Target modules | q_proj, v_proj |
| Dropout | 0.05 |
| Quantization | 8-bit (training) |
| Optimizer | AdamW |
| Learning rate | 2×10⁻⁴ |
| Per-device batch | 4-8 (model-dependent) |
| Gradient accumulation | 8 steps |
| Epochs | 3 |
| Precision | FP16 |
| Warmup steps | 50 |

### LLM Inference
| Parameter | Value |
|-----------|-------|
| Temperature | 0.0 (greedy) |
| Repetition penalty | 1.1 |
| Max tokens (detection) | 32 |
| Max tokens (extraction) | 256 |
| Quantization | 4-bit NF4 (32B+ models) |
| Batch size | 2-16 (model-dependent) |

### Models Supported

**BERT family** (4 models):
- bert-base-uncased
- allenai/scibert_scivocab_uncased
- microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
- dmis-lab/biobert-v1.1

**LLM family - fine-tunable** (5 models):
- meta-llama/Llama-3.2-3B-Instruct
- meta-llama/Llama-3.1-8B-Instruct
- mistralai/Mistral-7B-Instruct-v0.2
- Qwen/Qwen2.5-7B-Instruct
- deepseek-ai/deepseek-llm-7b-chat

**LLM family - zero-shot only** (4 models):
- deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
- meta-llama/Llama-3.3-70B-Instruct
- deepseek-ai/DeepSeek-R1-Distill-Llama-70B
- mistralai/Mixtral-8x7B-Instruct-v0.1

---

## Output Structure

```
results/
├── detection/
│   └── <model>/
│       ├── zero-shot_results.json
│       ├── few-shot_results.json
│       ├── cot_results.json
│       └── ...
├── extraction/
│   └── extraction_<split>_<model>_<strategy>_<timestamp>.json
└── finetuned_models/
    └── <split>/<model>/results.json

checkpoints/
├── detection_<model>/final/
└── extraction_<split>_<model>/final/

logs/
├── bert_detection.log
├── llm_extraction_all.log
├── transfer_learning.log
└── ...
```

#!/usr/bin/env python3
"""
transfer_learning.py
============================
Two-phase pipeline:

  PHASE 1 -- For every dataset x model:
               (a) Split 70% train / 30% test  (stratified where possible)
               (b) Downsample the TRAIN split so both classes are equal
               (c) Fine-tune the base model on the balanced train split
               (d) Save the fine-tuned model

  PHASE 2 -- Evaluation:
               within : every fine-tuned model tested on its OWN 30% test split
               cross  : ONLY the biocausal4k fine-tuned model tested on every
                        OTHER dataset's 30% test split

Base models (hardcoded, no --model_dirs needed):
  bert-base-uncased
  allenai/scibert_scivocab_uncased
  microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
  dmis-lab/biobert-base-cased-v1.2

Datasets (built-in):
  biocause1.csv              labels <- Cause/Effect cols
  fincausal.csv              labels <- Cause/Effect cols
  altlex_train.csv           label_col=pair_label
  train_subtask1.csv         label_col=label
  detection/prepared/detection_train.json  label_col=label  <- CROSS SOURCE

Output layout (under --base_out_dir, default ./results/balanced_experiment):
  models/<dataset>/<model_name>/final/
  predictions/predictions_<dataset>_<model>_within.csv
  predictions/predictions_cross_biocausal4k-to-<dataset>_<model>.csv
  metrics_summary.csv
  metrics_summary.json
  run_config.json

Usage:
  CUDA_VISIBLE_DEVICES=5 nohup python transfer_learning.py \\
      > ./logs/transfer_learning.log 2>&1 &
"""

import argparse
import inspect
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)
from transformers.trainer_utils import EvalPrediction


# ---------------------------------------------------------------------------
# Transformers API compat (evaluation_strategy -> eval_strategy in >= 4.46)
# ---------------------------------------------------------------------------

_TA_PARAMS = inspect.signature(TrainingArguments.__init__).parameters
_USE_NEW_STRATEGY = "eval_strategy" in _TA_PARAMS


def _strategy_kwargs(eval_strat="epoch", save_strat="epoch"):
    if _USE_NEW_STRATEGY:
        return {"eval_strategy": eval_strat, "save_strategy": save_strat}
    return {"evaluation_strategy": eval_strat, "save_strategy": save_strat}


# ---------------------------------------------------------------------------
# Hardcoded base models (starting weights for fine-tuning)
# ---------------------------------------------------------------------------

DEFAULT_MODELS = [
    "bert-base-uncased",
    "allenai/scibert_scivocab_uncased",
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "dmis-lab/biobert-base-cased-v1.2",
]

# ---------------------------------------------------------------------------
# Built-in dataset configs
# ---------------------------------------------------------------------------

DEFAULT_DATASETS = [
    {
        "file":     "tl_data/biocause1.csv",
        "format":   "csv",
        "text_col": "Sentences",
        "name":     "biocause",
    },
    {
        "file":     "tl_data/fincausal.csv",
        "format":   "csv",
        "text_col": "Sentences",
        "name":     "fincausal",
    },
    {
        "file":      "tl_data/altlex_train.csv",
        "format":    "csv",
        "text_col":  "text",
        "name":      "altlex",
        "label_col": "pair_label",
    },
    {
        "file":      "data/prepared/detection_train.json",
        "format":    "json",
        "text_col":  "sentence",
        "name":      "biocausal4k",
        "label_col": "label",
    },
    {
        "file":      "tl_data/ctb_train.csv",
        "format":    "csv",
        "text_col":  "text",
        "name":      "ctb",
        "label_col": "pair_label",
    },
]

# Dataset whose fine-tuned model is used for cross-dataset evaluation.
DEFAULT_CROSS_SOURCE = "biocausal4k"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def has_config(path):
    return os.path.isfile(os.path.join(path, "config.json"))


def resolve_model_dir(path):
    """
    If path does not exist on disk, treat it as a HuggingFace Hub model ID
    (handles both 'bert-base-uncased' and 'org/model' formats).
    Otherwise resolve to the best local checkpoint.
    """
    if not os.path.exists(path):
        return path  # HuggingFace Hub ID, pass through to transformers
    path = os.path.normpath(path)
    if has_config(path):
        return path
    final_dir = os.path.join(path, "final")
    if os.path.isdir(final_dir) and has_config(final_dir):
        return final_dir
    ckpts = sorted(
        [c for c in Path(path).glob("checkpoint-*") if c.is_dir()],
        key=lambda p: int(re.search(r"checkpoint-(\d+)$", str(p)).group(1))
    )
    if not ckpts:
        raise FileNotFoundError("No loadable model found in {}".format(path))
    latest = str(ckpts[-1])
    if not has_config(latest):
        raise FileNotFoundError("Latest checkpoint missing config.json: {}".format(latest))
    return latest


def model_shortname(model_id):
    """
    Return a short filesystem-safe name for a model ID.
    Specific names are checked before generic 'bert' to avoid substring
    collisions (e.g. 'pubmedbert-base-uncased' contains 'bert-base-uncased').
      'bert-base-uncased'                               -> 'bert'
      'allenai/scibert_scivocab_uncased'                -> 'scibert'
      'microsoft/BiomedNLP-PubMedBERT-base-uncased-...' -> 'pubmedbert'
      'dmis-lab/biobert-base-cased-v1.2'                -> 'biobert'
    """
    lower = model_id.lower()
    # Check specific names FIRST to avoid 'bert-base-uncased' matching inside
    # 'pubmedbert-base-uncased' or 'biobert-base-cased'
    if "pubmedbert" in lower or "pubmed" in lower:
        return "pubmedbert"
    if "biobert" in lower:
        return "biobert"
    if "scibert" in lower:
        return "scibert"
    if "bert" in lower:
        return "bert"
    return model_id.split("/")[-1]


def nonempty(x):
    if x is None:
        return False
    if isinstance(x, float) and np.isnan(x):
        return False
    return str(x).strip() != ""


def build_labels_from_cause_effect(df):
    c1 = df.get("Cause1",  pd.Series([None] * len(df)))
    e1 = df.get("Effect1", pd.Series([None] * len(df)))
    c2 = df.get("Cause2",  pd.Series([None] * len(df)))
    e2 = df.get("Effect2", pd.Series([None] * len(df)))
    return np.array(
        [1 if ((nonempty(a) and nonempty(b)) or (nonempty(c) and nonempty(d))) else 0
         for a, b, c, d in zip(c1, e1, c2, e2)],
        dtype=int,
    )


def load_dataset(ds_cfg, csv_encoding=None):
    """Return (df, texts, labels) with text column standardised to 'Sentences'."""
    fmt       = ds_cfg.get("format", "csv").lower()
    filepath  = ds_cfg["file"]
    text_col  = ds_cfg["text_col"]
    label_col = ds_cfg.get("label_col", None)

    if fmt == "json":
        with open(filepath, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        df = pd.DataFrame(raw)
    else:
        read_kw = {"encoding": csv_encoding} if csv_encoding else {}
        df = pd.read_csv(filepath, **read_kw)

    if text_col not in df.columns:
        raise ValueError("text column '{}' not found in {}. Available: {}".format(
            text_col, filepath, df.columns.tolist()))

    if text_col != "Sentences":
        df = df.rename(columns={text_col: "Sentences"})

    texts = df["Sentences"].fillna("").astype(str).tolist()

    if label_col is not None:
        if label_col not in df.columns:
            raise ValueError("label column '{}' not found in {}. Available: {}".format(
                label_col, filepath, df.columns.tolist()))
        labels = df[label_col].astype(int).values
        print("  Labels from '{}': causal={}, non-causal={}".format(
            label_col, int((labels == 1).sum()), int((labels == 0).sum())))
    else:
        labels = build_labels_from_cause_effect(df)
        print("  Labels from Cause/Effect cols: causal={}, non-causal={}".format(
            int((labels == 1).sum()), int((labels == 0).sum())))

    return df, texts, labels


def safe_split(texts, labels, idx, test_size, seed):
    n_classes = len(np.unique(labels))
    strat = labels if n_classes > 1 else None
    if strat is None:
        print("  WARNING: only one class present -- using non-stratified split.")
    return train_test_split(
        texts, labels, idx,
        test_size=test_size, stratify=strat, random_state=seed,
    )


def downsample_to_balance(texts, labels, seed):
    """
    Downsample majority class so both classes are equal in size.
    Only ever called on the TRAIN split, never on the test split.
    """
    texts  = list(texts)
    labels = np.array(labels)
    idx_pos = np.where(labels == 1)[0]
    idx_neg = np.where(labels == 0)[0]
    n_min   = min(len(idx_pos), len(idx_neg))
    rng     = np.random.default_rng(seed)
    if len(idx_pos) > n_min:
        idx_pos = rng.choice(idx_pos, size=n_min, replace=False)
    if len(idx_neg) > n_min:
        idx_neg = rng.choice(idx_neg, size=n_min, replace=False)
    bal_idx = np.concatenate([idx_pos, idx_neg])
    rng.shuffle(bal_idx)
    return [texts[i] for i in bal_idx], labels[bal_idx]


def confusion_counts(y_true, y_pred):
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return tn, fp, fn, tp


def prf_from_counts(tn, fp, fn, tp):
    p   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f   = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    return p, r, f, acc


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class CausalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.encodings = tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# ---------------------------------------------------------------------------
# Trainer metrics callback
# ---------------------------------------------------------------------------

def make_compute_metrics(threshold=0.5):
    def compute_metrics(eval_pred: EvalPrediction):
        logits, labels = eval_pred
        if logits.ndim == 2 and logits.shape[-1] == 2:
            probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
        else:
            probs = torch.sigmoid(torch.tensor(logits)).squeeze(-1).numpy()
        preds = (probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_counts(labels, preds)
        p, r, f, acc = prf_from_counts(tn, fp, fn, tp)
        return {"precision": p, "recall": r, "f1": f, "accuracy": acc}
    return compute_metrics


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

def fine_tune(
    pretrained_dir, train_texts, train_labels, val_texts, val_labels,
    save_dir, epochs, batch_size, lr, max_length, threshold, seed,
):
    """Fine-tune and save best model to save_dir/final/. Returns that path."""
    tokenizer = AutoTokenizer.from_pretrained(pretrained_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_dir, num_labels=2, ignore_mismatched_sizes=True
    )
    train_ds = CausalDataset(train_texts, train_labels, tokenizer, max_length)
    val_ds   = CausalDataset(val_texts,   val_labels,   tokenizer, max_length)
    os.makedirs(save_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        **_strategy_kwargs("epoch", "epoch"),
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        seed=seed,
        report_to="none",
        save_total_limit=1,
    )
    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=make_compute_metrics(threshold),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    final_dir = os.path.join(save_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print("  Saved -> {}".format(final_dir))
    return final_dir


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model_dir, test_texts, test_labels, batch_size, max_length, threshold, device):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    all_probs, all_preds = [], []
    with torch.no_grad():
        for i in range(0, len(test_texts), batch_size):
            batch = test_texts[i: i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            if logits.shape[-1] == 2:
                probs = torch.softmax(logits, dim=-1)[:, 1]
            else:
                probs = torch.sigmoid(logits).squeeze(-1)
            p_cpu = probs.detach().cpu().numpy()
            all_probs.extend(p_cpu.tolist())
            all_preds.extend((p_cpu >= threshold).astype(int).tolist())
    all_probs = np.array(all_probs, dtype=float)
    all_preds = np.array(all_preds, dtype=int)
    tn, fp, fn, tp = confusion_counts(test_labels, all_preds)
    p, r, f, acc = prf_from_counts(tn, fp, fn, tp)
    return {
        "probs": all_probs, "preds": all_preds,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "precision": p, "recall": r, "f1": f, "accuracy": acc,
        "support_total": len(test_labels),
        "support_pos": int((test_labels == 1).sum()),
        "support_neg": int((test_labels == 0).sum()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Balanced BERT training + within/cross-dataset evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--datasets_cfg",  default=None,
                    help="JSON file to override built-in dataset list.")
    ap.add_argument("--base_out_dir",  default="./results/balanced_experiment")
    ap.add_argument("--cross_source",  default=DEFAULT_CROSS_SOURCE,
                    help="Dataset whose fine-tuned model is used for cross eval. "
                         "Default: biocausal4k")
    ap.add_argument("--epochs",        type=int,   default=3)
    ap.add_argument("--batch_size",    type=int,   default=16)
    ap.add_argument("--lr",            type=float, default=2e-5)
    ap.add_argument("--max_length",    type=int,   default=256)
    ap.add_argument("--threshold",     type=float, default=0.5)
    ap.add_argument("--test_size",     type=float, default=0.30)
    ap.add_argument("--seed",          type=int,   default=42)
    ap.add_argument("--csv_encoding",  default=None)
    ap.add_argument("--skip_training", action="store_true",
                    help="Skip Phase 1; reload saved checkpoints (crash recovery).")
    args = ap.parse_args()

    set_seed(args.seed)

    # Use hardcoded base models -- no --model_dirs needed
    model_list = DEFAULT_MODELS

    models_dir      = os.path.join(args.base_out_dir, "models")
    predictions_dir = os.path.join(args.base_out_dir, "predictions")
    os.makedirs(models_dir,      exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)

    with open(os.path.join(args.base_out_dir, "run_config.json"), "w") as fh:
        json.dump(vars(args), fh, indent=2)

    print("Strategy API : {}".format(
        "new (eval_strategy)" if _USE_NEW_STRATEGY else "old (evaluation_strategy)"))
    print("Output root  : {}".format(args.base_out_dir))
    print("Cross source : {}  (this model is tested on all other datasets)".format(
        args.cross_source))

    datasets = json.load(open(args.datasets_cfg)) if args.datasets_cfg else DEFAULT_DATASETS
    ds_names = [d["name"] for d in datasets]

    if args.cross_source not in ds_names:
        raise ValueError("--cross_source '{}' not found in dataset list: {}".format(
            args.cross_source, ds_names))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device  : {}".format(device))
    print("Datasets: {}".format(ds_names))
    print("Models  : {}".format(model_list))
    print("Split   : {}% train / {}% test  +  downsample train to balance".format(
        int((1 - args.test_size) * 100), int(args.test_size * 100)))

    # =========================================================================
    # PHASE 1 -- Split, downsample, fine-tune, save every dataset x model
    # =========================================================================

    print("\n" + "#" * 72)
    print("# PHASE 1 -- Fine-tuning  ({} datasets x {} models)".format(
        len(datasets), len(model_list)))
    print("#" * 72)

    prepared = {}

    for ds_cfg in datasets:
        ds_name = ds_cfg["name"]

        print("\n" + "=" * 72)
        print("DATASET : {}  ({})".format(ds_name, ds_cfg["file"]))
        print("=" * 72)

        try:
            df, texts, labels = load_dataset(ds_cfg, args.csv_encoding)
        except (FileNotFoundError, ValueError) as exc:
            msg = "  ERROR loading {}: {}".format(ds_cfg["file"], exc)
            if ds_name == args.cross_source:
                raise RuntimeError(
                    "{}\n\n"
                    "  The cross-source dataset ('{}') is required but could not be loaded.\n"
                    "  Check that '{}' is correct relative to your working directory.\n"
                    "  Run: find . -name '*.json' | grep -i detection\n"
                    "  Then update DEFAULT_DATASETS in the script with the correct path.".format(
                        msg, args.cross_source, ds_cfg["file"])
                )
            print(msg + ". Skipping.")
            continue

        print("  Total: {}  causal={}  non-causal={}".format(
            len(texts), int((labels == 1).sum()), int((labels == 0).sum())))

        # Step 1 -- 70/30 split
        idx = list(range(len(texts)))
        (
            train_texts, test_texts,
            train_labels, test_labels,
            train_idx, test_idx,
        ) = safe_split(texts, labels, idx, args.test_size, args.seed)
        train_labels = np.array(train_labels)
        test_labels  = np.array(test_labels)

        print("\n  [Step 1] 70/30 split")
        print("    Train : {}  (pos={} neg={})".format(
            len(train_texts), int(train_labels.sum()), int((train_labels == 0).sum())))
        print("    Test  : {}  (pos={} neg={})  <- locked; never touched during training".format(
            len(test_texts), int(test_labels.sum()), int((test_labels == 0).sum())))

        # Step 2 -- Downsample train to balance classes
        bal_texts, bal_labels = downsample_to_balance(train_texts, train_labels, args.seed)

        print("\n  [Step 2] Downsample train to balance")
        print("    Before : pos={} neg={}  total={}".format(
            int(train_labels.sum()), int((train_labels == 0).sum()), len(train_labels)))
        print("    After  : pos={} neg={}  total={}".format(
            int((bal_labels == 1).sum()), int((bal_labels == 0).sum()), len(bal_labels)))

        # Val slice from balanced train (for early stopping only)
        val_frac      = min(0.10 / (1.0 - args.test_size), 0.15)
        n_bal_classes = len(np.unique(bal_labels))
        tr_texts, val_texts, tr_labels, val_labels = train_test_split(
            bal_texts, bal_labels,
            test_size=val_frac,
            stratify=bal_labels if n_bal_classes > 1 else None,
            random_state=args.seed,
        )
        print("    Fit on : {}  Val: {}".format(len(tr_texts), len(val_texts)))

        test_df = df.iloc[list(test_idx)].copy().reset_index(drop=True)

        prepared[ds_name] = {
            "test_texts":  test_texts,
            "test_labels": test_labels,
            "test_df":     test_df,
            "train_bal":   len(bal_texts),
            "train_pos":   int((bal_labels == 1).sum()),
            "train_neg":   int((bal_labels == 0).sum()),
            "ft_paths":    {},
        }

        # Step 3 -- Fine-tune each base model on this dataset
        for model_id in model_list:
            short_name = model_shortname(model_id)
            ft_save_dir  = os.path.join(models_dir, ds_name, short_name)
            ft_final_dir = os.path.join(ft_save_dir, "final")

            print("\n  [Step 3] Fine-tune  model={}  ({})".format(short_name, model_id))

            if args.skip_training and os.path.isdir(ft_final_dir) and has_config(ft_final_dir):
                print("  [skip_training] Reusing {}".format(ft_final_dir))
            else:
                fine_tune(
                    pretrained_dir=model_id,  # HuggingFace Hub ID, downloaded automatically
                    train_texts=list(tr_texts), train_labels=list(tr_labels),
                    val_texts=list(val_texts),   val_labels=list(val_labels),
                    save_dir=ft_save_dir,
                    epochs=args.epochs, batch_size=args.batch_size,
                    lr=args.lr, max_length=args.max_length,
                    threshold=args.threshold, seed=args.seed,
                )

            prepared[ds_name]["ft_paths"][short_name] = ft_final_dir

    # =========================================================================
    # PHASE 2 -- Evaluation
    #   within : each fine-tuned model tested on its own test split
    #   cross  : only the biocausal4k fine-tuned model tested on every other
    # =========================================================================

    print("\n" + "#" * 72)
    print("# PHASE 2 -- Evaluation")
    print("#   Within : each model on its own test split")
    print("#   Cross  : {} model on all other test splits".format(args.cross_source))
    print("#" * 72)

    all_rows = []

    for model_id in model_list:
        short_name = model_shortname(model_id)

        print("\n" + "=" * 72)
        print("MODEL : {}  ({})".format(short_name, model_id))
        print("=" * 72)

        # ---- WITHIN ----
        print("\n  -- WITHIN --")
        for ds_name, info in prepared.items():
            ft_path = info["ft_paths"].get(short_name)
            if not ft_path or not os.path.isdir(ft_path):
                print("  SKIP within {}: no checkpoint found.".format(ds_name))
                continue

            print("  [WITHIN]  train={} -> test={}".format(ds_name, ds_name))
            res = evaluate(
                model_dir=ft_path,
                test_texts=info["test_texts"],
                test_labels=info["test_labels"],
                batch_size=args.batch_size * 2,
                max_length=args.max_length,
                threshold=args.threshold,
                device=device,
            )
            print("  TN={} FP={} FN={} TP={}  "
                  "P={:.4f} R={:.4f} F1={:.4f} Acc={:.4f}".format(
                res["tn"], res["fp"], res["fn"], res["tp"],
                res["precision"], res["recall"], res["f1"], res["accuracy"]))

            pred_df = info["test_df"].copy()
            pred_df["prob_causal"] = res["probs"]
            pred_df["pred_causal"] = res["preds"]
            pred_df["gold_causal"] = info["test_labels"]
            pred_path = os.path.join(
                predictions_dir,
                "predictions_{}_{}_{}.csv".format(ds_name, short_name, "within")
            )
            pred_df.to_csv(pred_path, index=False)

            all_rows.append({
                "eval_type":       "within",
                "train_dataset":   ds_name,
                "test_dataset":    ds_name,
                "model":           short_name,
                "model_id":        model_id,
                "ft_path":         ft_path,
                "train_bal_total": info["train_bal"],
                "train_pos":       info["train_pos"],
                "train_neg":       info["train_neg"],
                "test_total":      res["support_total"],
                "test_pos":        res["support_pos"],
                "test_neg":        res["support_neg"],
                "tn": res["tn"], "fp": res["fp"],
                "fn": res["fn"], "tp": res["tp"],
                "precision":       res["precision"],
                "recall":          res["recall"],
                "f1":              res["f1"],
                "accuracy":        res["accuracy"],
                "predictions_csv": pred_path,
            })

        # ---- CROSS ----
        cross_info = prepared.get(args.cross_source)
        if cross_info is None:
            print("\n  SKIP cross: '{}' was not loaded.".format(args.cross_source))
            continue

        cross_ft_path = cross_info["ft_paths"].get(short_name)
        if not cross_ft_path or not os.path.isdir(cross_ft_path):
            print("\n  SKIP cross: no checkpoint for {} on {}.".format(
                short_name, args.cross_source))
            continue

        print("\n  -- CROSS  (model trained on {}) --".format(args.cross_source))
        for ds_name, info in prepared.items():
            if ds_name == args.cross_source:
                continue  # within already handled above

            print("  [CROSS]  train={} -> test={}".format(args.cross_source, ds_name))
            res = evaluate(
                model_dir=cross_ft_path,
                test_texts=info["test_texts"],
                test_labels=info["test_labels"],
                batch_size=args.batch_size * 2,
                max_length=args.max_length,
                threshold=args.threshold,
                device=device,
            )
            print("  TN={} FP={} FN={} TP={}  "
                  "P={:.4f} R={:.4f} F1={:.4f} Acc={:.4f}".format(
                res["tn"], res["fp"], res["fn"], res["tp"],
                res["precision"], res["recall"], res["f1"], res["accuracy"]))

            pred_df = info["test_df"].copy()
            pred_df["prob_causal"] = res["probs"]
            pred_df["pred_causal"] = res["preds"]
            pred_df["gold_causal"] = info["test_labels"]
            pred_path = os.path.join(
                predictions_dir,
                "predictions_cross_{}-to-{}_{}.csv".format(
                    args.cross_source, ds_name, short_name)
            )
            pred_df.to_csv(pred_path, index=False)

            all_rows.append({
                "eval_type":       "cross",
                "train_dataset":   args.cross_source,
                "test_dataset":    ds_name,
                "model":           short_name,
                "model_id":        model_id,
                "ft_path":         cross_ft_path,
                "train_bal_total": cross_info["train_bal"],
                "train_pos":       cross_info["train_pos"],
                "train_neg":       cross_info["train_neg"],
                "test_total":      res["support_total"],
                "test_pos":        res["support_pos"],
                "test_neg":        res["support_neg"],
                "tn": res["tn"], "fp": res["fp"],
                "fn": res["fn"], "tp": res["tp"],
                "precision":       res["precision"],
                "recall":          res["recall"],
                "f1":              res["f1"],
                "accuracy":        res["accuracy"],
                "predictions_csv": pred_path,
            })

    # ---- Save metrics ----
    metrics_csv = os.path.join(args.base_out_dir, "metrics_summary.csv")
    pd.DataFrame(all_rows).to_csv(metrics_csv, index=False)
    print("\nSaved metrics CSV : {}".format(metrics_csv))

    metrics_json = os.path.join(args.base_out_dir, "metrics_summary.json")
    with open(metrics_json, "w", encoding="utf-8") as fh:
        json.dump(all_rows, fh, indent=2)
    print("Saved metrics JSON: {}".format(metrics_json))

    # ---- Final summary ----
    print("\n" + "=" * 95)
    print("FINAL SUMMARY")
    print("=" * 95)
    print("{:<8} {:<14} {:<14} {:<12} {:>7} {:>7} {:>7} {:>7}".format(
        "Type", "TrainSet", "TestSet", "Model", "P", "R", "F1", "Acc"))
    print("-" * 95)
    for row in all_rows:
        print("{:<8} {:<14} {:<14} {:<12} {:>7.4f} {:>7.4f} {:>7.4f} {:>7.4f}".format(
            row["eval_type"], row["train_dataset"], row["test_dataset"], row["model"],
            row["precision"], row["recall"], row["f1"], row["accuracy"],
        ))
    print("=" * 95)
    print("\nAll outputs saved under: {}".format(args.base_out_dir))
    print("  models/           <- fine-tuned checkpoints  (<dataset>/<model>/final/)")
    print("  predictions/      <- row-level prediction CSVs")
    print("  metrics_summary.csv / .json")
    print("  run_config.json")


if __name__ == "__main__":
    main()
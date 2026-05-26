#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


def normalize_text(text):
    if text is None:
        return ""

    text = str(text).lower().strip()
    text = re.sub(r"^(the|a|an)\s+", "", text)
    text = text.strip('.,;:!?"\'-()[]{}')
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokenize(text):
    return normalize_text(text).split()


def token_f1(pred, gold):
    pred_tokens = tokenize(pred)
    gold_tokens = tokenize(gold)

    if not pred_tokens and not gold_tokens:
        return 1.0

    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counts = defaultdict(int)
    gold_counts = defaultdict(int)

    for token in pred_tokens:
        pred_counts[token] += 1

    for token in gold_tokens:
        gold_counts[token] += 1

    overlap = sum(min(pred_counts[token], gold_counts[token]) for token in pred_counts)

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


def pair_soft_score(pred_pair, gold_pair):
    cause_score = token_f1(
        pred_pair.get("cause", ""),
        gold_pair.get("cause", "")
    )

    effect_score = token_f1(
        pred_pair.get("effect", ""),
        gold_pair.get("effect", "")
    )

    return (cause_score + effect_score) / 2


def exact_pair_match(pred_pair, gold_pair):
    pred_cause = normalize_text(pred_pair.get("cause", ""))
    pred_effect = normalize_text(pred_pair.get("effect", ""))

    gold_cause = normalize_text(gold_pair.get("cause", ""))
    gold_effect = normalize_text(gold_pair.get("effect", ""))

    return pred_cause == gold_cause and pred_effect == gold_effect


def norm_label(label):
    if label is None:
        return ""

    label = str(label).strip().lower()

    if label in {"explicit", "exp", "explict", "explicts", "explitcit"}:
        return "Explicit"

    if label in {"implicit", "imp"}:
        return "Implicit"

    if label in {"intra", "intrasentential", "intra-sentential", "intra sentential"}:
        return "Intra"

    if label in {"inter", "intersentential", "inter-sentential", "inter sentential"}:
        return "Inter"

    return label.title()

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def align_predictions_to_gold(prediction_file, gold_file):
    pred_data = load_json(prediction_file)
    gold_data = load_json(gold_file)

    predictions = pred_data.get("predictions", [])

    gold_by_sn = {
        item.get("s/n"): item
        for item in gold_data
    }

    rows = []

    for sample_index, pred_item in enumerate(predictions):
        sn = pred_item.get("s/n")
        sentence = pred_item.get("sentence", "")
        pred_pairs = pred_item.get("pairs", [])

        if not isinstance(pred_pairs, list):
            pred_pairs = []

        gold_item = gold_by_sn.get(sn, {})
        gold_pairs = gold_item.get("pairs", [])

        if not isinstance(gold_pairs, list):
            gold_pairs = []

        for pred_pair_index, pred_pair in enumerate(pred_pairs):
            best_gold_index = None
            best_gold_pair = None
            best_exact = 0
            best_soft_score = 0.0

            if gold_pairs:
                candidates = []

                for gold_index, gold_pair in enumerate(gold_pairs):
                    exact = int(exact_pair_match(pred_pair, gold_pair))
                    soft_score = pair_soft_score(pred_pair, gold_pair)

                    candidates.append({
                        "gold_index": gold_index,
                        "gold_pair": gold_pair,
                        "exact": exact,
                        "soft_score": soft_score,
                    })

                candidates = sorted(
                    candidates,
                    key=lambda x: (x["exact"], x["soft_score"]),
                    reverse=True
                )

                best = candidates[0]
                best_gold_index = best["gold_index"]
                best_gold_pair = best["gold_pair"]
                best_exact = best["exact"]
                best_soft_score = best["soft_score"]

            rows.append({
                "s/n": sn,
                "sample_index": sample_index,
                "sentence": sentence,
                "pred_pair_index": pred_pair_index,

                "pred_cause": pred_pair.get("cause", ""),
                "pred_effect": pred_pair.get("effect", ""),
                "pred_causality": pred_pair.get("causality", ""),
                "pred_sententiality": pred_pair.get("sententiality", ""),

                "best_gold_pair_index": best_gold_index,
                "gold_cause": best_gold_pair.get("cause", "") if best_gold_pair else "",
                "gold_effect": best_gold_pair.get("effect", "") if best_gold_pair else "",
                "gold_causality": best_gold_pair.get("causality", "") if best_gold_pair else "",
                "gold_sententiality": best_gold_pair.get("sententiality", "") if best_gold_pair else "",

                "exact_match": best_exact,
                "soft_token_overlap_match": int(best_soft_score > 0),
                "soft_pair_score": best_soft_score,
            })

    return pd.DataFrame(rows)


def compute_label_metrics(df, label_task):
    pred_col = f"pred_{label_task}_norm"
    gold_col = f"gold_{label_task}_norm"

    valid_df = df[
        (df[pred_col] != "")
        & (df[gold_col] != "")
    ].copy()

    if len(valid_df) == 0:
        return {
            "Pairs": 0,
            "Correct": 0,
            "Incorrect": 0,
            "Accuracy": 0.0,
            "Macro F1": 0.0,
            "Weighted F1": 0.0,
        }

    labels = sorted(set(valid_df[pred_col]) | set(valid_df[gold_col]))

    f1_scores = []
    supports = []

    for label in labels:
        tp = int(((valid_df[pred_col] == label) & (valid_df[gold_col] == label)).sum())
        fp = int(((valid_df[pred_col] == label) & (valid_df[gold_col] != label)).sum())
        fn = int(((valid_df[pred_col] != label) & (valid_df[gold_col] == label)).sum())
        support = int((valid_df[gold_col] == label).sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        f1_scores.append(f1)
        supports.append(support)

    correct = int((valid_df[pred_col] == valid_df[gold_col]).sum())
    incorrect = int((valid_df[pred_col] != valid_df[gold_col]).sum())

    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    weighted_f1 = float(np.average(f1_scores, weights=supports)) if sum(supports) > 0 else 0.0

    return {
        "Pairs": len(valid_df),
        "Correct": correct,
        "Incorrect": incorrect,
        "Accuracy": correct / len(valid_df),
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pred",
        default="extraction_combined_DeepSeek-R1-Distill-Qwen-32B_few-shot_20260308_184255.json"
    )

    parser.add_argument(
        "--gold",
        default="test.json"
    )

    parser.add_argument(
        "--out_csv",
        default="label_ablation_summary_simple.csv"
    )

    parser.add_argument(
        "--alignment_csv",
        default="deepseek_gold_alignment.csv"
    )

    args = parser.parse_args()

    prediction_file = Path(args.pred)
    gold_file = Path(args.gold)

    df = align_predictions_to_gold(prediction_file, gold_file)

    df["pred_causality_norm"] = df["pred_causality"].apply(norm_label)
    df["gold_causality_norm"] = df["gold_causality"].apply(norm_label)

    df["pred_sententiality_norm"] = df["pred_sententiality"].apply(norm_label)
    df["gold_sententiality_norm"] = df["gold_sententiality"].apply(norm_label)

    exact_df = df[df["exact_match"] == 1].copy()

    partial_df = df[
        (df["exact_match"] == 0)
        & (df["soft_token_overlap_match"] == 1)
    ].copy()

    exact_plus_partial_df = df[
        (df["exact_match"] == 1)
        | (df["soft_token_overlap_match"] == 1)
    ].copy()

    subsets = [
        ("Exact span matches", exact_df),
        ("Partial span matches only", partial_df),
        ("Exact + partial span matches", exact_plus_partial_df),
    ]

    rows = []

    for subset_name, subset_df in subsets:
        for task in ["causality", "sententiality"]:
            metrics = compute_label_metrics(subset_df, task)

            rows.append({
                "Subset": subset_name,
                "Label task": "Causality" if task == "causality" else "Sententiality",
                **metrics,
            })

    result = pd.DataFrame(rows)

    result.to_csv(args.out_csv, index=False)
    df.to_csv(args.alignment_csv, index=False)

    display_df = result.copy()

    for col in ["Accuracy", "Macro F1", "Weighted F1"]:
        display_df[col] = (display_df[col] * 100).map(lambda x: f"{x:.2f}%")

    for col in ["Pairs", "Correct", "Incorrect"]:
        display_df[col] = display_df[col].map(lambda x: f"{x:,}")

    print()
    print(display_df.to_markdown(index=False))
    print()
    print(f"Saved summary CSV: {args.out_csv}")
    print(f"Saved alignment CSV: {args.alignment_csv}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exact_first_error_bucketization_v2.py

Corrected bucketization logic:
1. Exact match
2. Cause-effect switched
3. Effect correct, cause wrong
4. Cause correct, effect wrong
5. Partial match (soft/cosine, no exact)
6. No C-E pair
7. Spurious prediction
8. Others

Why this version exists:
The previous version placed the broad Partial match bucket before side-specific
buckets. That made "Effect correct, cause wrong" and "Cause correct, effect wrong"
become zero because those rows were swallowed by Partial match first.

Inputs:
- prediction_pair_match_flags.csv
- gold_pair_match_flags.csv

Run:
python exact_first_error_bucketization_v2.py \
  --prediction_csv prediction_pair_match_flags.csv \
  --gold_csv gold_pair_match_flags.csv \
  --out_csv exact_first_error_buckets_v2.csv \
  --summary_csv exact_first_error_bucket_summary_v2.csv
"""

import argparse
import re
from typing import List

import numpy as np
import pandas as pd

BUCKETS = [
    "Exact match",
    "Cause-effect switched",
    "Effect correct, cause wrong",
    "Cause correct, effect wrong",
    "Partial match (soft/cosine, no exact)",
    "No C-E pair",
    "Spurious prediction",
    "Others",
]


def normalize_text(text) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"^(the|a|an)\s+", "", text)
    text = text.strip('.,;:!?"\'-()[]{}')
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def has_text(text) -> bool:
    return normalize_text(text) != ""


def safe_int(value, default=0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def safe_float(value, default=np.nan) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def get_first(row, candidates: List[str], default=""):
    for col in candidates:
        if col in row.index:
            val = row[col]
            if not pd.isna(val):
                return val
    return default


def clean_col_name(text: str) -> str:
    text = text.lower()
    text = text.replace("/", "_")
    text = text.replace("-", "_")
    text = text.replace(",", "")
    text = text.replace("(", "")
    text = text.replace(")", "")
    text = re.sub(r"\s+", "_", text)
    return text


def get_sn(row):
    return get_first(row, ["s/n", "sn", "sample_id", "id", "index"])


def get_sentence(row):
    return get_first(row, ["sentence", "text", "abstract", "input_sentence"])


def get_pred_cause(row):
    return get_first(row, [
        "pred_cause", "predicted_cause", "prediction_cause",
        "best_pred_cause", "matched_pred_cause"
    ])


def get_pred_effect(row):
    return get_first(row, [
        "pred_effect", "predicted_effect", "prediction_effect",
        "best_pred_effect", "matched_pred_effect"
    ])


def get_gold_cause(row):
    return get_first(row, [
        "gold_cause", "best_gold_cause", "matched_gold_cause",
        "best_gold_cause_soft", "best_gold_cause_cosine",
        "true_cause", "reference_cause"
    ])


def get_gold_effect(row):
    return get_first(row, [
        "gold_effect", "best_gold_effect", "matched_gold_effect",
        "best_gold_effect_soft", "best_gold_effect_cosine",
        "true_effect", "reference_effect"
    ])


def get_flag(row, candidates: List[str], default=0) -> int:
    for col in candidates:
        if col in row.index and not pd.isna(row[col]):
            return safe_int(row[col], default)
    return default


def get_score(row, candidates: List[str], default=np.nan) -> float:
    for col in candidates:
        if col in row.index and not pd.isna(row[col]):
            return safe_float(row[col], default)
    return default


def exact_match(row) -> int:
    return get_flag(row, ["exact_match", "pair_exact_match", "exact_pair_match"], 0)


def soft_match(row) -> int:
    # Uses the binary flag if it already exists.
    return get_flag(row, [
        "soft_token_overlap_match", "soft_match", "soft_pair_match", "token_overlap_match"
    ], 0)


def cosine_match(row, cosine_threshold: float) -> int:
    # Prefer existing binary flag.
    for col in ["cosine_similarity_match", "cosine_match", "cosine_pair_match"]:
        if col in row.index and not pd.isna(row[col]):
            return safe_int(row[col], 0)

    # Fallback to score if binary flag is not available.
    score = get_score(row, [
        "cosine_pair_score", "best_cosine_pair_score", "pair_cosine_score", "cos_pair_score"
    ])
    if np.isnan(score):
        return 0
    return int(score >= cosine_threshold)


def token_f1(a, b) -> float:
    a = normalize_text(a)
    b = normalize_text(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    a_tokens = a.split()
    b_tokens = b.split()
    if not a_tokens or not b_tokens:
        return 0.0
    ac = {}
    bc = {}
    for t in a_tokens:
        ac[t] = ac.get(t, 0) + 1
    for t in b_tokens:
        bc[t] = bc.get(t, 0) + 1
    overlap = sum(min(ac.get(t, 0), bc.get(t, 0)) for t in ac)
    if overlap == 0:
        return 0.0
    p = overlap / len(a_tokens)
    r = overlap / len(b_tokens)
    return 2 * p * r / (p + r)


def side_score(row, side: str) -> float:
    """Return the best available side score, preferring explicit score columns."""
    if side == "cause":
        score = get_score(row, [
            "soft_cause_score", "cause_soft_score", "token_cause_score",
            "cosine_cause_score", "cause_cosine_score", "cos_cause_score"
        ])
        if not np.isnan(score):
            return score
        return token_f1(get_pred_cause(row), get_gold_cause(row))

    score = get_score(row, [
        "soft_effect_score", "effect_soft_score", "token_effect_score",
        "cosine_effect_score", "effect_cosine_score", "cos_effect_score"
    ])
    if not np.isnan(score):
        return score
    return token_f1(get_pred_effect(row), get_gold_effect(row))


def side_correct(row, side: str, side_threshold: float) -> bool:
    """
    A side is correct if:
    - a side exact flag exists and is 1; OR
    - normalized pred side equals normalized gold side; OR
    - side score >= threshold.
    """
    if side == "cause":
        exact_cols = ["cause_exact_match", "exact_cause_match", "cause_correct_flag"]
        pred = get_pred_cause(row)
        gold = get_gold_cause(row)
    else:
        exact_cols = ["effect_exact_match", "exact_effect_match", "effect_correct_flag"]
        pred = get_pred_effect(row)
        gold = get_gold_effect(row)

    for col in exact_cols:
        if col in row.index and not pd.isna(row[col]) and safe_int(row[col], 0) == 1:
            return True

    if has_text(pred) and has_text(gold) and normalize_text(pred) == normalize_text(gold):
        return True

    return side_score(row, side) >= side_threshold


def switched_match(row, switched_threshold: float) -> bool:
    for col in [
        "cause_effect_switched", "cause_effect_switched_flag", "switched_match",
        "cosine_switched_match", "bucket_07_cause_effect_switched",
    ]:
        if col in row.index and not pd.isna(row[col]):
            return safe_int(row[col], 0) == 1

    pred_cause = get_pred_cause(row)
    pred_effect = get_pred_effect(row)
    gold_cause = get_gold_cause(row)
    gold_effect = get_gold_effect(row)

    if not (has_text(pred_cause) and has_text(pred_effect) and has_text(gold_cause) and has_text(gold_effect)):
        return False

    pc_ge = token_f1(pred_cause, gold_effect)
    pe_gc = token_f1(pred_effect, gold_cause)
    return pc_ge >= switched_threshold and pe_gc >= switched_threshold


def classify_prediction_row(row, cosine_threshold: float, side_threshold: float, switched_threshold: float) -> dict:
    exact = exact_match(row)
    soft = soft_match(row)
    cos = cosine_match(row, cosine_threshold)
    partial_no_exact = int(exact == 0 and (soft == 1 or cos == 1))

    pred_has_cause = has_text(get_pred_cause(row))
    pred_has_effect = has_text(get_pred_effect(row))

    cause_ok = side_correct(row, "cause", side_threshold)
    effect_ok = side_correct(row, "effect", side_threshold)
    switched = switched_match(row, switched_threshold)

    # Corrected hierarchy: broad Partial match comes AFTER specific side-error buckets.
    if exact == 1:
        bucket = "Exact match"
    elif switched:
        bucket = "Cause-effect switched"
    elif effect_ok and not cause_ok:
        bucket = "Effect correct, cause wrong"
    elif cause_ok and not effect_ok:
        bucket = "Cause correct, effect wrong"
    elif partial_no_exact == 1:
        bucket = "Partial match (soft/cosine, no exact)"
    elif pred_has_cause and pred_has_effect:
        bucket = "Spurious prediction"
    else:
        bucket = "Others"

    if exact == 1:
        partial_source = "not_partial_exact_match"
    elif soft == 1 and cos == 1:
        partial_source = "soft_and_cosine"
    elif soft == 1:
        partial_source = "soft_only"
    elif cos == 1:
        partial_source = "cosine_only"
    else:
        partial_source = "none"

    return {
        "main_bucket": bucket,
        "exact_match": exact,
        "soft_token_overlap_match": soft,
        "cosine_similarity_match": cos,
        "partial_match_no_exact": partial_no_exact,
        "partial_match_source": partial_source,
        "cause_side_score": side_score(row, "cause"),
        "effect_side_score": side_score(row, "effect"),
        "cause_correct_side": int(cause_ok),
        "effect_correct_side": int(effect_ok),
        "cause_effect_switched": int(switched),
    }


def build_no_ce_rows(gold_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    pred_sns_with_pairs = set()
    for _, row in pred_df.iterrows():
        sn = get_sn(row)
        if pd.isna(sn):
            continue
        if has_text(get_pred_cause(row)) or has_text(get_pred_effect(row)):
            pred_sns_with_pairs.add(str(sn))

    rows = []
    for _, row in gold_df.iterrows():
        sn = get_sn(row)
        if pd.isna(sn):
            continue
        if str(sn) not in pred_sns_with_pairs:
            rows.append({
                "row_source": "gold_no_prediction",
                "s/n": sn,
                "sentence": get_sentence(row),
                "pred_cause": "",
                "pred_effect": "",
                "gold_cause": get_gold_cause(row),
                "gold_effect": get_gold_effect(row),
                "exact_match": 0,
                "soft_token_overlap_match": 0,
                "cosine_similarity_match": 0,
                "partial_match_no_exact": 0,
                "partial_match_source": "none",
                "cause_side_score": 0.0,
                "effect_side_score": 0.0,
                "cause_correct_side": 0,
                "effect_correct_side": 0,
                "cause_effect_switched": 0,
                "main_bucket": "No C-E pair",
            })
    return pd.DataFrame(rows)


def add_bucket_flags(df: pd.DataFrame) -> pd.DataFrame:
    for bucket in BUCKETS:
        col = "bucket_" + clean_col_name(bucket)
        df[col] = (df["main_bucket"] == bucket).astype(int)
    bucket_cols = ["bucket_" + clean_col_name(bucket) for bucket in BUCKETS]
    df["bucket_flag_sum"] = df[bucket_cols].sum(axis=1)
    bad = df[df["bucket_flag_sum"] != 1]
    if len(bad) > 0:
        raise AssertionError(f"{len(bad)} rows do not have exactly one bucket.")
    return df


def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = df["main_bucket"].value_counts().rename_axis("error_bucket").reset_index(name="count")
    all_buckets = pd.DataFrame({"error_bucket": BUCKETS})
    summary = all_buckets.merge(summary, on="error_bucket", how="left")
    summary["count"] = summary["count"].fillna(0).astype(int)
    total = summary["count"].sum()
    summary["percentage"] = (summary["count"] / total * 100).round(2) if total > 0 else 0.0
    return summary


def main():
    parser = argparse.ArgumentParser(description="Corrected exact-first bucketization with side-specific buckets before general partial match.")
    parser.add_argument("--prediction_csv", required=True, help="prediction_pair_match_flags.csv")
    parser.add_argument("--gold_csv", required=True, help="gold_pair_match_flags.csv")
    parser.add_argument("--out_csv", default="exact_first_error_buckets_v2.csv")
    parser.add_argument("--summary_csv", default="exact_first_error_bucket_summary_v2.csv")
    parser.add_argument("--non_exact_summary_csv", default="non_exact_error_bucket_summary_v2.csv")
    parser.add_argument("--partial_summary_csv", default="partial_match_summary_v2.csv")
    parser.add_argument("--cosine_threshold", type=float, default=0.75)
    parser.add_argument("--side_threshold", type=float, default=0.75, help="Threshold for cause/effect side correctness.")
    parser.add_argument("--switched_threshold", type=float, default=0.75)
    args = parser.parse_args()

    pred_df = pd.read_csv(args.prediction_csv)
    gold_df = pd.read_csv(args.gold_csv)

    pred_records = []
    for _, row in pred_df.iterrows():
        rec = row.to_dict()
        rec["row_source"] = "prediction"
        rec["s/n"] = get_sn(row)
        rec["sentence"] = get_sentence(row)
        rec["pred_cause"] = get_pred_cause(row)
        rec["pred_effect"] = get_pred_effect(row)
        rec["gold_cause"] = get_gold_cause(row)
        rec["gold_effect"] = get_gold_effect(row)
        rec.update(classify_prediction_row(row, args.cosine_threshold, args.side_threshold, args.switched_threshold))
        pred_records.append(rec)

    pred_bucketed = pd.DataFrame(pred_records)
    no_ce_df = build_no_ce_rows(gold_df, pred_df)
    combined = pd.concat([pred_bucketed, no_ce_df], ignore_index=True, sort=False)
    combined = add_bucket_flags(combined)

    full_summary = make_summary(combined)
    non_exact = combined[combined["main_bucket"] != "Exact match"].copy()
    non_exact_summary = make_summary(non_exact)
    non_exact_summary = non_exact_summary[non_exact_summary["error_bucket"] != "Exact match"].reset_index(drop=True)

    partial_df = combined[combined["partial_match_no_exact"] == 1].copy()
    if len(partial_df) > 0:
        partial_summary = partial_df["partial_match_source"].value_counts().rename_axis("partial_match_source").reset_index(name="count")
        partial_summary["percentage"] = (partial_summary["count"] / len(partial_df) * 100).round(2)
    else:
        partial_summary = pd.DataFrame(columns=["partial_match_source", "count", "percentage"])

    combined.to_csv(args.out_csv, index=False)
    full_summary.to_csv(args.summary_csv, index=False)
    non_exact_summary.to_csv(args.non_exact_summary_csv, index=False)
    partial_summary.to_csv(args.partial_summary_csv, index=False)

    assert len(combined) == full_summary["count"].sum()
    assert len(non_exact) == non_exact_summary["count"].sum()

    print("Done.")
    print(f"Prediction rows read: {len(pred_df)}")
    print(f"Gold rows read: {len(gold_df)}")
    print(f"Rows bucketized including exact + No C-E: {len(combined)}")
    print(f"Side threshold: {args.side_threshold}")
    print()
    print("Full summary including Exact match:")
    print(full_summary.to_string(index=False))
    print()
    print("Non-exact summary excluding Exact match:")
    print(non_exact_summary.to_string(index=False))
    print()
    print("Partial match flag breakdown, regardless of main bucket:")
    print(partial_summary.to_string(index=False))


if __name__ == "__main__":
    main()

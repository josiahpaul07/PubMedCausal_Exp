# -*- coding: utf-8 -*-
"""
run_re_evaluate.py
==================
Wrapper script that:
  1. Scans results/extraction/ for all prediction JSON files
  2. Groups by (model, strategy) - picks LATEST timestamp per group
  3. Runs soft F1 + cosine evaluation on the deduplicated set
  4. Saves a single unified summary CSV and JSON

Usage:
  # Without cosine (fast)
  python run_re_evaluate.py \
      --pred_dir  ./results/extraction \
      --gt_dir    ./data/prepared/extraction_combined \
      --out_csv   ./results/final_eval_summary.csv

  # With BioSentBERT cosine similarity
  python run_re_evaluate.py \
      --pred_dir  ./results/extraction \
      --gt_dir    ./data/prepared/extraction_combined \
      --out_csv   ./results/final_eval_summary.csv \
      --cosine
"""

import os
import re
import sys
import json
import glob
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np


# -------------------------------------------------------------
# Filename parser
# Handles both patterns:
#   extraction_combined_DeepSeek-7B_cot_20260213_165959.json
#   extraction_combined_Llama-3B_zero-shot_20260213_120000.json
# -------------------------------------------------------------

STRATEGIES = [
    'zero-shot', 'few-shot', 'cot-fewshot', 'cot',
    'least-to-most', 'react'
]
# Sort longest first to prevent 'cot' matching inside 'cot-fewshot'
STRATEGIES_SORTED = sorted(STRATEGIES, key=len, reverse=True)


def parse_filename(fname: str) -> Optional[Tuple[str, str, str, str]]:
    """
    Parse prediction filename into (split, model, strategy, timestamp).
    Returns None if pattern doesn't match.

    Pattern: extraction_{split}_{model}_{strategy}_{YYYYMMDD}_{HHMMSS}.json
    """
    base = os.path.splitext(os.path.basename(fname))[0]

    # Must start with extraction_
    if not base.startswith('extraction_'):
        return None

    # Try each known strategy to split the name
    for strategy in STRATEGIES_SORTED:
        # Build pattern: extraction_{split}_{model}_{strategy}_{timestamp}
        pattern = rf'^extraction_(.+?)_(.+?)_{re.escape(strategy)}_(\d{{8}}_\d{{6}})$'
        m = re.match(pattern, base)
        if m:
            split     = m.group(1)
            model     = m.group(2)
            timestamp = m.group(3)
            return split, model, strategy, timestamp

    # Fallback: generic pattern - last two _ groups are timestamp
    parts = base.split('_')
    if len(parts) >= 4:
        # Last two parts are timestamp: YYYYMMDD_HHMMSS
        try:
            ts = '_'.join(parts[-2:])
            datetime.strptime(ts, '%Y%m%d_%H%M%S')  # validate
            # Strategy is the part before timestamp
            # Find which strategy matches
            remaining = '_'.join(parts[:-2])  # everything before timestamp
            for strategy in STRATEGIES_SORTED:
                if remaining.endswith('_' + strategy):
                    prefix = remaining[:-(len(strategy)+1)]
                    # prefix = extraction_{split}_{model}
                    inner = prefix[len('extraction_'):]
                    # Try to find split
                    for split in ['combined', 'X_only', 'Y_only']:
                        if inner.startswith(split + '_'):
                            model = inner[len(split)+1:]
                            return split, model, strategy, ts
            return None
        except ValueError:
            return None

    return None


def pick_latest(file_list: List[str]) -> str:
    """Given a list of files with same (model, strategy), pick the latest timestamp."""
    def get_ts(f):
        result = parse_filename(f)
        return result[3] if result else '00000000_000000'
    return max(file_list, key=get_ts)


def find_gt_file(gt_dir: str, split: str) -> Optional[str]:
    """Find ground truth test.json for a given split"""
    # Try exact match first
    candidates = [
        os.path.join(gt_dir, 'test.json'),                      # if gt_dir is already the split dir
        os.path.join(gt_dir, split, 'test.json'),               # e.g. extraction_combined/test.json
        os.path.join(gt_dir, f'extraction_{split}', 'test.json'),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c

    # Recursive search
    for root, dirs, files in os.walk(gt_dir):
        if 'test.json' in files and split in root:
            return os.path.join(root, 'test.json')

    return None


# -------------------------------------------------------------
# Copy of core evaluation logic from re_evaluate.py
# (so this script is self-contained)
# -------------------------------------------------------------

_cosine_model      = None
_cosine_tokenizer  = None

def load_cosine_model(model_name='pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb'):
    global _cosine_model, _cosine_tokenizer
    if _cosine_model is not None:
        return
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
        print(f"  Loading BioSentBERT: {model_name}...")
        _cosine_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _cosine_model     = AutoModel.from_pretrained(model_name)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _cosine_model.to(device)
        _cosine_model.eval()
        print(f"  BioSentBERT loaded on {device}")
    except Exception as e:
        print(f"  WARNING: BioSentBERT not loaded ({e}). Cosine will be skipped.")
        _cosine_model = None


def _normalize(text) -> str:
    if isinstance(text, list):
        text = text[0] if text else ''
    if text is None:
        return ''
    text = str(text).lower().strip()
    text = re.sub(r'^(the|a|an)\s+', '', text)
    text = text.strip('.,;:!?"\'-()[]{}')
    return text.strip()


def _tokenize(text: str) -> List[str]:
    return _normalize(text).split()


def _token_f1(pred: str, gold: str) -> float:
    p_tok = _tokenize(pred)
    g_tok = _tokenize(gold)
    if not p_tok and not g_tok:
        return 1.0
    if not p_tok or not g_tok:
        return 0.0
    pc = defaultdict(int)
    gc = defaultdict(int)
    for t in p_tok: pc[t] += 1
    for t in g_tok: gc[t] += 1
    overlap = sum(min(pc[t], gc[t]) for t in pc)
    if overlap == 0:
        return 0.0
    p = overlap / len(p_tok)
    r = overlap / len(g_tok)
    return 2*p*r/(p+r)


def _best_tf1(pred: str, golds: List[str]) -> float:
    if not golds: return 0.0
    return max(_token_f1(pred, g) for g in golds)


def _get_embeddings(texts: list):
    if _cosine_model is None or not texts:
        return None
    import torch
    device = next(_cosine_model.parameters()).device
    enc = _cosine_tokenizer(
        texts, padding=True, truncation=True,
        max_length=128, return_tensors='pt'
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = _cosine_model(**enc)
    tok_emb = out.last_hidden_state
    mask    = enc['attention_mask'].unsqueeze(-1).expand(tok_emb.size()).float()
    pooled  = torch.sum(tok_emb * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
    norms   = pooled.norm(dim=1, keepdim=True).clamp(min=1e-9)
    return (pooled / norms).cpu().numpy()


def _best_cosine(pred: str, golds: List[str]) -> float:
    if _cosine_model is None or not golds:
        return -1.0
    texts = [_normalize(pred)] + [_normalize(g) for g in golds]
    embs  = _get_embeddings(texts)
    if embs is None:
        return -1.0
    return float(max(np.dot(embs[0], embs[i+1]) for i in range(len(golds))))


def evaluate_file(pred_file: str, gt_file: str, cosine_threshold: float = 0.75) -> Dict:
    """Full evaluation of a single prediction file"""

    # Load predictions
    with open(pred_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    predictions = data.get('predictions', [])

    # Build s/n-keyed pred dict for proper alignment
    pred_by_sn = {}
    for item in predictions:
        sn = item.get('s/n', None)
        pairs = item.get('pairs', [])
        if not isinstance(pairs, list):
            pairs = []
        if sn is not None:
            pred_by_sn[sn] = pairs
    use_sn_align = bool(pred_by_sn)

    # Positional fallback
    if not use_sn_align:
        pred_pairs_positional = [item.get('pairs', []) if isinstance(item.get('pairs', []), list)
                                 else [] for item in predictions]

    # Load ground truth
    with open(gt_file, 'r', encoding='utf-8') as f:
        gt_raw = json.load(f)
    gt_list_raw = [{'pairs': item.get('pairs', []), 's/n': item.get('s/n', None)} for item in gt_raw]

    # Align by s/n if available
    if use_sn_align:
        pred_pairs_list = []
        gt_list         = []
        missing = 0
        for gt_item in gt_list_raw:
            sn = gt_item.get('s/n', None)
            if sn is not None and sn in pred_by_sn:
                pred_pairs_list.append(pred_by_sn[sn])
            else:
                pred_pairs_list.append([])   # no prediction -> penalizes recall
                missing += 1
            gt_list.append(gt_item)
        if missing:
            print(f"    NOTE: {missing} gt samples had no matching prediction (s/n not found)")
    else:
        # Positional fallback
        pred_pairs_list = pred_pairs_positional
        gt_list         = gt_list_raw
        n = min(len(pred_pairs_list), len(gt_list))
        if len(pred_pairs_list) != len(gt_list):
            print(f"    WARNING: pred={len(pred_pairs_list)} gt={len(gt_list)}, using first {n}")
        pred_pairs_list = pred_pairs_list[:n]
        gt_list         = gt_list[:n]

    # Accumulate scores
    cause_p_all,  cause_r_all  = [], []
    effect_p_all, effect_r_all = [], []
    pair_p_all,   pair_r_all   = [], []

    cos_cause_p_all  = []   # precision: pred -> gold
    cos_cause_r_all  = []   # recall:    gold -> pred
    cos_effect_p_all = []
    cos_effect_r_all = []
    cos_pair_p_all   = []
    cos_pair_r_all   = []

    exact_cause_hits = exact_effect_hits = exact_pair_hits = 0
    total_pred_causes = total_gold_causes = 0
    total_pred_effects = total_gold_effects = 0
    total_pred_pairs = total_gold_pairs = 0

    for pred_pairs, gold_item in zip(pred_pairs_list, gt_list):
        gold_pairs = gold_item.get('pairs', [])
        if not isinstance(gold_pairs, list):
            gold_pairs = []
        if not isinstance(pred_pairs, list):
            pred_pairs = []

        gc_list = [_normalize(g.get('cause',  '')) for g in gold_pairs if g.get('cause')]
        ge_list = [_normalize(g.get('effect', '')) for g in gold_pairs if g.get('effect')]
        gp_list = [(_normalize(g.get('cause','')), _normalize(g.get('effect','')))
                   for g in gold_pairs if g.get('cause') and g.get('effect')]

        pc_list = [_normalize(p.get('cause',  '')) for p in pred_pairs if p.get('cause')]
        pe_list = [_normalize(p.get('effect', '')) for p in pred_pairs if p.get('effect')]
        pp_list = [(_normalize(p.get('cause','')), _normalize(p.get('effect','')))
                   for p in pred_pairs if p.get('cause') and p.get('effect')]

        # Token F1 precision (pred -> gold)
        for pc in pc_list:
            cause_p_all.append(_best_tf1(pc, gc_list) if gc_list else 0.0)
            if any(_normalize(pc) == gc for gc in gc_list):
                exact_cause_hits += 1
        for gc in gc_list:
            cause_r_all.append(_best_tf1(gc, pc_list) if pc_list else 0.0)

        for pe in pe_list:
            effect_p_all.append(_best_tf1(pe, ge_list) if ge_list else 0.0)
            if any(_normalize(pe) == ge for ge in ge_list):
                exact_effect_hits += 1
        for ge in ge_list:
            effect_r_all.append(_best_tf1(ge, pe_list) if pe_list else 0.0)

        for pc, pe in pp_list:
            s = max((_token_f1(pc,gc)+_token_f1(pe,ge))/2 for gc,ge in gp_list) if gp_list else 0.0
            pair_p_all.append(s)
            if any(pc==gc and pe==ge for gc,ge in gp_list):
                exact_pair_hits += 1
        for gc, ge in gp_list:
            s = max((_token_f1(pc,gc)+_token_f1(pe,ge))/2 for pc,pe in pp_list) if pp_list else 0.0
            pair_r_all.append(s)

        # Cosine similarity - proper P/R/F1 (mirrors token F1 logic)
        if _cosine_model is not None:
            # Cause precision: for each predicted cause, best cosine match in gold causes
            for pc in pc_list:
                cos_cause_p_all.append(_best_cosine(pc, gc_list) if gc_list else 0.0)
            # Cause recall: for each gold cause, best cosine match in predicted causes
            for gc in gc_list:
                cos_cause_r_all.append(_best_cosine(gc, pc_list) if pc_list else 0.0)

            # Effect precision
            for pe in pe_list:
                cos_effect_p_all.append(_best_cosine(pe, ge_list) if ge_list else 0.0)
            # Effect recall
            for ge in ge_list:
                cos_effect_r_all.append(_best_cosine(ge, pe_list) if pe_list else 0.0)

            # Pair precision: for each pred pair, best average cosine over gold pairs
            for pc, pe in pp_list:
                if gp_list:
                    s = max((_best_cosine(pc,[gc]) + _best_cosine(pe,[ge])) / 2 for gc, ge in gp_list)
                else:
                    s = 0.0
                cos_pair_p_all.append(s)
            # Pair recall: for each gold pair, best average cosine over pred pairs
            for gc, ge in gp_list:
                if pp_list:
                    s = max((_best_cosine(gc,[pc]) + _best_cosine(ge,[pe])) / 2 for pc, pe in pp_list)
                else:
                    s = 0.0
                cos_pair_r_all.append(s)

        total_pred_causes  += len(pc_list);  total_gold_causes  += len(gc_list)
        total_pred_effects += len(pe_list);  total_gold_effects += len(ge_list)
        total_pred_pairs   += len(pp_list);  total_gold_pairs   += len(gp_list)

    def prf(ps, rs):
        p = float(np.mean(ps)) if ps else 0.0
        r = float(np.mean(rs)) if rs else 0.0
        f = 2*p*r/(p+r) if (p+r) > 0 else 0.0
        return p, r, f

    def ef1(p, r):
        return 2*p*r/(p+r) if (p+r) > 0 else 0.0

    cp, cr, cf = prf(cause_p_all,  cause_r_all)
    ep, er, ef = prf(effect_p_all, effect_r_all)
    pp, pr, pf = prf(pair_p_all,   pair_r_all)

    ecp = exact_cause_hits  / total_pred_causes  if total_pred_causes  else 0.0
    ecr = exact_cause_hits  / total_gold_causes  if total_gold_causes  else 0.0
    eep = exact_effect_hits / total_pred_effects if total_pred_effects else 0.0
    eer = exact_effect_hits / total_gold_effects if total_gold_effects else 0.0
    epp = exact_pair_hits   / total_pred_pairs   if total_pred_pairs   else 0.0
    epr = exact_pair_hits   / total_gold_pairs   if total_gold_pairs   else 0.0

    def cos_prf(ps, rs):
        p = float(np.mean(ps)) if ps else 0.0
        r = float(np.mean(rs)) if rs else 0.0
        f = 2*p*r/(p+r) if (p+r) > 0 else 0.0
        return p, r, f

    cos_cp, cos_cr, cos_cf = cos_prf(cos_cause_p_all,  cos_cause_r_all)
    cos_ep, cos_er, cos_ef = cos_prf(cos_effect_p_all, cos_effect_r_all)
    cos_pp, cos_pr, cos_pf = cos_prf(cos_pair_p_all,   cos_pair_r_all)

    # % of precision scores above threshold (semantic precision rate)
    cos_cause_abv  = float(np.mean([s >= cosine_threshold for s in cos_cause_p_all]))  if cos_cause_p_all  else -1.0
    cos_effect_abv = float(np.mean([s >= cosine_threshold for s in cos_effect_p_all])) if cos_effect_p_all else -1.0
    cos_pair_abv   = float(np.mean([s >= cosine_threshold for s in cos_pair_p_all]))   if cos_pair_p_all   else -1.0

    return {
        'soft_cause_p':  cp,  'soft_cause_r':  cr,  'soft_cause_f1':  cf,
        'soft_effect_p': ep,  'soft_effect_r': er,  'soft_effect_f1': ef,
        'soft_pair_p':   pp,  'soft_pair_r':   pr,  'soft_pair_f1':   pf,
        'exact_cause_f1':  ef1(ecp, ecr),
        'exact_effect_f1': ef1(eep, eer),
        'exact_pair_f1':   ef1(epp, epr),
        'cosine_cause_p':      cos_cp,
        'cosine_cause_r':      cos_cr,
        'cosine_cause_f1':     cos_cf,
        'cosine_effect_p':     cos_ep,
        'cosine_effect_r':     cos_er,
        'cosine_effect_f1':    cos_ef,
        'cosine_pair_p':       cos_pp,
        'cosine_pair_r':       cos_pr,
        'cosine_pair_f1':      cos_pf,
        'cosine_cause_above':  cos_cause_abv,
        'cosine_effect_above': cos_effect_abv,
        'cosine_pair_above':   cos_pair_abv,
        'cosine_threshold':    cosine_threshold,
        'total_pred_pairs':    total_pred_pairs,
        'total_gold_pairs':    total_gold_pairs,
        'total_samples':       len(gt_list),
    }


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description='Batch re-evaluate extraction results with deduplication by model+strategy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic (Token F1 + Exact F1)
  python run_re_evaluate.py \\
      --pred_dir ./results/extraction \\
      --gt_dir   ./data/prepared/extraction_combined \\
      --out_csv  ./results/final_eval_summary.csv

  # With BioSentBERT cosine similarity
  python run_re_evaluate.py \\
      --pred_dir ./results/extraction \\
      --gt_dir   ./data/prepared/extraction_combined \\
      --out_csv  ./results/final_eval_summary.csv \\
      --cosine
        """
    )
    ap.add_argument('--pred_dir', required=True,
                    help='Directory containing prediction JSON files')
    ap.add_argument('--gt_dir',   required=True,
                    help='Ground truth directory (contains test.json)')
    ap.add_argument('--out_csv',  default='./results/final_eval_summary.csv',
                    help='Output summary CSV')
    ap.add_argument('--out_json', default=None,
                    help='Output full metrics JSON (optional)')
    ap.add_argument('--cosine',   action='store_true',
                    help='Enable BioSentBERT cosine similarity')
    ap.add_argument('--cosine_model', default='pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb',
                    help='HuggingFace model for cosine similarity')
    ap.add_argument('--cosine_threshold', type=float, default=0.75,
                    help='Cosine match threshold (default: 0.75)')
    ap.add_argument('--split', default='combined',
                    choices=['combined', 'X_only', 'Y_only'],
                    help='Which split to evaluate (default: combined)')
    args = ap.parse_args()

    # Load BioSentBERT if requested
    if args.cosine:
        load_cosine_model(args.cosine_model)

    # Find ground truth file
    gt_file = find_gt_file(args.gt_dir, args.split)
    if not gt_file:
        print(f"ERROR: Could not find test.json in {args.gt_dir} for split '{args.split}'")
        sys.exit(1)
    print(f"Ground truth: {gt_file}")

    # -- Scan and parse all prediction files ------------------
    all_files = glob.glob(os.path.join(args.pred_dir, '*.json'))
    all_files = [f for f in all_files if 'summary' not in os.path.basename(f).lower()]
    print(f"Found {len(all_files)} prediction files in {args.pred_dir}")

    # Group by (split, model, strategy)
    groups = defaultdict(list)
    skipped = []
    for f in all_files:
        result = parse_filename(f)
        if result:
            split, model, strategy, ts = result
            if split == args.split:
                groups[(model, strategy)].append(f)
        else:
            skipped.append(os.path.basename(f))

    if skipped:
        print(f"Skipped {len(skipped)} files (unrecognised pattern):")
        for s in skipped[:5]:
            print(f"  {s}")
        if len(skipped) > 5:
            print(f"  ... and {len(skipped)-5} more")

    if not groups:
        print(f"ERROR: No matching files found for split='{args.split}'. Check --split value.")
        print("Available files:")
        for f in all_files[:10]:
            print(f"  {os.path.basename(f)}")
        sys.exit(1)

    # Pick latest timestamp per (model, strategy)
    selected = {}
    for (model, strategy), files in groups.items():
        chosen = pick_latest(files)
        selected[(model, strategy)] = chosen
        if len(files) > 1:
            print(f"  [{model}|{strategy}] {len(files)} files -> kept: {os.path.basename(chosen)}")

    print(f"\nEvaluating {len(selected)} unique (model, strategy) combinations...")
    print(f"Split: {args.split}  |  Cosine: {'ON' if args.cosine else 'OFF'}\n")

    # -- Evaluate each selected file ---------------------------
    rows = []
    full_results = []

    for (model, strategy), pred_file in sorted(selected.items()):
        print(f"  {model:<30} {strategy:<15} -> {os.path.basename(pred_file)}")
        try:
            metrics = evaluate_file(pred_file, gt_file, args.cosine_threshold)

            # Print mini-report
            cos_str = ""
            if args.cosine and metrics['cosine_pair_f1'] >= 0:
                cos_str = f"  Cos.Pair.F1={metrics['cosine_pair_f1']:.4f} (P={metrics['cosine_pair_p']:.4f} R={metrics['cosine_pair_r']:.4f})"
            print(f"    Soft P-F1={metrics['soft_pair_f1']:.4f}  "
                  f"Soft C-F1={metrics['soft_cause_f1']:.4f}  "
                  f"Soft E-F1={metrics['soft_effect_f1']:.4f}  "
                  f"Exact P-F1={metrics['exact_pair_f1']:.4f}{cos_str}")

            row = {
                'model':    model,
                'strategy': strategy,
                'split':    args.split,
                'file':     os.path.basename(pred_file),
                **metrics
            }
            rows.append(row)
            full_results.append({'model': model, 'strategy': strategy,
                                  'file': pred_file, 'metrics': metrics})

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback; traceback.print_exc()
            continue

    if not rows:
        print("No results to save.")
        sys.exit(1)

    # -- Save outputs ------------------------------------------
    df = pd.DataFrame(rows)

    # Sort by soft pair F1 descending
    df = df.sort_values('soft_pair_f1', ascending=False)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved summary CSV: {args.out_csv}")

    if args.out_json:
        with open(args.out_json, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, indent=2)
        print(f"Saved full JSON  : {args.out_json}")

    # -- Final summary table -----------------------------------
    print(f"\n{'='*115}")
    print(f"FINAL RESULTS -- Split: {args.split.upper()}")
    print("="*115)
    if args.cosine:
        print(f"{'Model':<30} {'Strategy':<15} {'Soft C-F1':>10} {'Soft E-F1':>10} {'Soft P-F1':>10} | {'Ex.P-F1':>9} | {'Cos.C-F1':>10} {'Cos.P-F1':>10}")
        print("-"*115)
        for _, row in df.iterrows():
            cc = f"{row['cosine_cause_f1']:.4f}" if row['cosine_cause_f1'] >= 0 else "  N/A  "
            cp = f"{row['cosine_pair_f1']:.4f}"  if row['cosine_pair_f1']  >= 0 else "  N/A  "
            print(f"{row['model']:<30} {row['strategy']:<15} "
                  f"{row['soft_cause_f1']:>10.4f} {row['soft_effect_f1']:>10.4f} {row['soft_pair_f1']:>10.4f} | "
                  f"{row['exact_pair_f1']:>9.4f} | "
                  f"{cc:>10} {cp:>10}")
    else:
        print(f"{'Model':<30} {'Strategy':<15} {'Soft C-F1':>10} {'Soft E-F1':>10} {'Soft P-F1':>10} | {'Ex.C-F1':>9} {'Ex.E-F1':>9} {'Ex.P-F1':>9}")
        print("-"*115)
        for _, row in df.iterrows():
            print(f"{row['model']:<30} {row['strategy']:<15} "
                  f"{row['soft_cause_f1']:>10.4f} {row['soft_effect_f1']:>10.4f} {row['soft_pair_f1']:>10.4f} | "
                  f"{row['exact_cause_f1']:>9.4f} {row['exact_effect_f1']:>9.4f} {row['exact_pair_f1']:>9.4f}")
    print("="*115)
    print(f"\nTotal combinations evaluated: {len(df)}")


if __name__ == "__main__":
    main()
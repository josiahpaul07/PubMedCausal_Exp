# -*- coding: utf-8 -*-
"""
two_file_match_flags_previous_logic.py
======================================
Compare ONLY two files:
  1) extraction_combined_DeepSeek-R1-Distill-Qwen-32B_few-shot_20260308_184255.json
  2) test.json

This version keeps the metric logic aligned with your original run_re_evaluate.py:
  - Soft token overlap F1 has NO threshold. Precision/recall/F1 are computed
    from mean raw token-F1 scores.
  - Cosine F1 is computed from mean raw cosine scores, NOT from the 0/1 thresholded flag.
  - cosine_threshold=0.75 is only used for row-level cosine_similarity_match and
    cosine_*_above rates, just like your previous script.

It also creates the requested row-level binary columns:
  - exact_match
  - soft_token_overlap_match
  - cosine_similarity_match

Run without cosine:
  python two_file_match_flags_previous_logic.py \
    --pred extraction_combined_DeepSeek-R1-Distill-Qwen-32B_few-shot_20260308_184255.json \
    --gold test.json \
    --out_dir two_file_previous_logic_outputs

Run with cosine:
  pip install transformers torch pandas numpy

  python two_file_match_flags_previous_logic.py \
    --pred extraction_combined_DeepSeek-R1-Distill-Qwen-32B_few-shot_20260308_184255.json \
    --gold test.json \
    --out_dir two_file_previous_logic_outputs \
    --cosine \
    --cosine_threshold 0.75
"""

import os
import re
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Same normalization and token-F1 logic as your previous script
# ---------------------------------------------------------------------

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
    for t in p_tok:
        pc[t] += 1
    for t in g_tok:
        gc[t] += 1

    overlap = sum(min(pc[t], gc[t]) for t in pc)
    if overlap == 0:
        return 0.0

    p = overlap / len(p_tok)
    r = overlap / len(g_tok)
    return 2 * p * r / (p + r)


def _best_tf1(pred: str, golds: List[str]) -> float:
    if not golds:
        return 0.0
    return max(_token_f1(pred, g) for g in golds)


# ---------------------------------------------------------------------
# Same BioBERT/BioSentBERT mean-pooling cosine logic as your script
# ---------------------------------------------------------------------

_cosine_model = None
_cosine_tokenizer = None


def load_cosine_model(model_name='pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb'):
    """Load cosine model exactly in the style of the previous script."""
    global _cosine_model, _cosine_tokenizer
    if _cosine_model is not None:
        return
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
        print(f"Loading BioSentBERT: {model_name}...")
        _cosine_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _cosine_model = AutoModel.from_pretrained(model_name)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _cosine_model.to(device)
        _cosine_model.eval()
        print(f"BioSentBERT loaded on {device}")
    except Exception as e:
        print(f"WARNING: BioSentBERT not loaded ({e}). Cosine will be skipped.")
        _cosine_model = None


def _get_embeddings(texts: List[str]):
    if _cosine_model is None or not texts:
        return None
    import torch
    device = next(_cosine_model.parameters()).device
    enc = _cosine_tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = _cosine_model(**enc)
    tok_emb = out.last_hidden_state
    mask = enc['attention_mask'].unsqueeze(-1).expand(tok_emb.size()).float()
    pooled = torch.sum(tok_emb * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
    norms = pooled.norm(dim=1, keepdim=True).clamp(min=1e-9)
    return (pooled / norms).cpu().numpy()


def _best_cosine(pred: str, golds: List[str]) -> float:
    """Same as your previous _best_cosine: normalize texts, embed together, return max dot product."""
    if _cosine_model is None or not golds:
        return -1.0
    texts = [_normalize(pred)] + [_normalize(g) for g in golds]
    embs = _get_embeddings(texts)
    if embs is None:
        return -1.0
    return float(max(np.dot(embs[0], embs[i + 1]) for i in range(len(golds))))


def _pair_cosine_against_one(pred_cause: str, pred_effect: str, gold_cause: str, gold_effect: str) -> float:
    if _cosine_model is None:
        return -1.0
    return (_best_cosine(pred_cause, [gold_cause]) + _best_cosine(pred_effect, [gold_effect])) / 2


# ---------------------------------------------------------------------
# Loading/alignment for the two files only
# ---------------------------------------------------------------------

def load_predictions(pred_file: str) -> List[Dict[str, Any]]:
    with open(pred_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'predictions' in data:
        return data.get('predictions', [])
    if isinstance(data, list):
        return data
    raise ValueError("Prediction file must be a list or a dict with key 'predictions'.")


def load_gold(gold_file: str) -> List[Dict[str, Any]]:
    with open(gold_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Gold file must be a list, like test.json.")
    return data


def aligned_pairs(pred_file: str, gold_file: str):
    predictions = load_predictions(pred_file)
    gt_raw = load_gold(gold_file)

    pred_by_sn: Dict[Any, List[Dict[str, Any]]] = {}
    pred_sentence_by_sn: Dict[Any, str] = {}
    for item in predictions:
        sn = item.get('s/n', None)
        pairs = item.get('pairs', [])
        if not isinstance(pairs, list):
            pairs = []
        if sn is not None:
            pred_by_sn[sn] = pairs
            pred_sentence_by_sn[sn] = item.get('sentence', '')

    use_sn_align = bool(pred_by_sn)

    if use_sn_align:
        rows = []
        missing = 0
        for sample_index, gt_item in enumerate(gt_raw):
            sn = gt_item.get('s/n', None)
            if sn is not None and sn in pred_by_sn:
                pred_pairs = pred_by_sn[sn]
            else:
                pred_pairs = []
                missing += 1
            gold_pairs = gt_item.get('pairs', [])
            if not isinstance(gold_pairs, list):
                gold_pairs = []
            rows.append({
                'sample_index': sample_index,
                's/n': sn,
                'sentence': gt_item.get('sentence', pred_sentence_by_sn.get(sn, '')),
                'gold_pairs': gold_pairs,
                'pred_pairs': pred_pairs,
            })
        if missing:
            print(f"NOTE: {missing} gold samples had no matching prediction by s/n.")
        return rows

    # Positional fallback, same spirit as your previous code
    rows = []
    n = min(len(predictions), len(gt_raw))
    if len(predictions) != len(gt_raw):
        print(f"WARNING: pred={len(predictions)} gt={len(gt_raw)}, using first {n}")
    for i in range(n):
        pred_pairs = predictions[i].get('pairs', [])
        gold_pairs = gt_raw[i].get('pairs', [])
        if not isinstance(pred_pairs, list):
            pred_pairs = []
        if not isinstance(gold_pairs, list):
            gold_pairs = []
        rows.append({
            'sample_index': i,
            's/n': gt_raw[i].get('s/n', None),
            'sentence': gt_raw[i].get('sentence', predictions[i].get('sentence', '')),
            'gold_pairs': gold_pairs,
            'pred_pairs': pred_pairs,
        })
    return rows


def pair_tuple(pair: Dict[str, Any]) -> Tuple[str, str]:
    return _normalize(pair.get('cause', '')), _normalize(pair.get('effect', ''))


def valid_pair(pair: Dict[str, Any]) -> bool:
    return bool(pair.get('cause')) and bool(pair.get('effect'))


# ---------------------------------------------------------------------
# EXACT previous-style metrics
# ---------------------------------------------------------------------

def prf(ps: List[float], rs: List[float]) -> Tuple[float, float, float]:
    p = float(np.mean(ps)) if ps else 0.0
    r = float(np.mean(rs)) if rs else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def ef1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def evaluate_previous_style(rows: List[Dict[str, Any]], cosine_threshold: float) -> Dict[str, Any]:
    """This mirrors evaluate_file() from your previous script, but for only these two files."""
    cause_p_all, cause_r_all = [], []
    effect_p_all, effect_r_all = [], []
    pair_p_all, pair_r_all = [], []

    cos_cause_p_all, cos_cause_r_all = [], []
    cos_effect_p_all, cos_effect_r_all = [], []
    cos_pair_p_all, cos_pair_r_all = [], []

    exact_cause_hits = exact_effect_hits = exact_pair_hits = 0
    total_pred_causes = total_gold_causes = 0
    total_pred_effects = total_gold_effects = 0
    total_pred_pairs = total_gold_pairs = 0

    for row in rows:
        pred_pairs = row['pred_pairs'] if isinstance(row['pred_pairs'], list) else []
        gold_pairs = row['gold_pairs'] if isinstance(row['gold_pairs'], list) else []

        gc_list = [_normalize(g.get('cause', '')) for g in gold_pairs if g.get('cause')]
        ge_list = [_normalize(g.get('effect', '')) for g in gold_pairs if g.get('effect')]
        gp_list = [pair_tuple(g) for g in gold_pairs if valid_pair(g)]

        pc_list = [_normalize(p.get('cause', '')) for p in pred_pairs if p.get('cause')]
        pe_list = [_normalize(p.get('effect', '')) for p in pred_pairs if p.get('effect')]
        pp_list = [pair_tuple(p) for p in pred_pairs if valid_pair(p)]

        # Soft token precision/recall, exactly as previous script
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
            s = max((_token_f1(pc, gc) + _token_f1(pe, ge)) / 2 for gc, ge in gp_list) if gp_list else 0.0
            pair_p_all.append(s)
            if any(pc == gc and pe == ge for gc, ge in gp_list):
                exact_pair_hits += 1
        for gc, ge in gp_list:
            s = max((_token_f1(pc, gc) + _token_f1(pe, ge)) / 2 for pc, pe in pp_list) if pp_list else 0.0
            pair_r_all.append(s)

        # Cosine precision/recall: raw-score averaging, exactly like previous script
        if _cosine_model is not None:
            for pc in pc_list:
                cos_cause_p_all.append(_best_cosine(pc, gc_list) if gc_list else 0.0)
            for gc in gc_list:
                cos_cause_r_all.append(_best_cosine(gc, pc_list) if pc_list else 0.0)

            for pe in pe_list:
                cos_effect_p_all.append(_best_cosine(pe, ge_list) if ge_list else 0.0)
            for ge in ge_list:
                cos_effect_r_all.append(_best_cosine(ge, pe_list) if pe_list else 0.0)

            for pc, pe in pp_list:
                if gp_list:
                    s = max((_best_cosine(pc, [gc]) + _best_cosine(pe, [ge])) / 2 for gc, ge in gp_list)
                else:
                    s = 0.0
                cos_pair_p_all.append(s)

            for gc, ge in gp_list:
                if pp_list:
                    s = max((_best_cosine(gc, [pc]) + _best_cosine(ge, [pe])) / 2 for pc, pe in pp_list)
                else:
                    s = 0.0
                cos_pair_r_all.append(s)

        total_pred_causes += len(pc_list)
        total_gold_causes += len(gc_list)
        total_pred_effects += len(pe_list)
        total_gold_effects += len(ge_list)
        total_pred_pairs += len(pp_list)
        total_gold_pairs += len(gp_list)

    cp, cr, cf = prf(cause_p_all, cause_r_all)
    ep, er, ef = prf(effect_p_all, effect_r_all)
    pp, pr, pf = prf(pair_p_all, pair_r_all)

    ecp = exact_cause_hits / total_pred_causes if total_pred_causes else 0.0
    ecr = exact_cause_hits / total_gold_causes if total_gold_causes else 0.0
    eep = exact_effect_hits / total_pred_effects if total_pred_effects else 0.0
    eer = exact_effect_hits / total_gold_effects if total_gold_effects else 0.0
    epp = exact_pair_hits / total_pred_pairs if total_pred_pairs else 0.0
    epr = exact_pair_hits / total_gold_pairs if total_gold_pairs else 0.0

    cos_cp, cos_cr, cos_cf = prf(cos_cause_p_all, cos_cause_r_all)
    cos_ep, cos_er, cos_ef = prf(cos_effect_p_all, cos_effect_r_all)
    cos_pp, cos_pr, cos_pf = prf(cos_pair_p_all, cos_pair_r_all)

    cos_cause_abv = float(np.mean([s >= cosine_threshold for s in cos_cause_p_all])) if cos_cause_p_all else -1.0
    cos_effect_abv = float(np.mean([s >= cosine_threshold for s in cos_effect_p_all])) if cos_effect_p_all else -1.0
    cos_pair_abv = float(np.mean([s >= cosine_threshold for s in cos_pair_p_all])) if cos_pair_p_all else -1.0

    return {
        'soft_cause_p': cp, 'soft_cause_r': cr, 'soft_cause_f1': cf,
        'soft_effect_p': ep, 'soft_effect_r': er, 'soft_effect_f1': ef,
        'soft_pair_p': pp, 'soft_pair_r': pr, 'soft_pair_f1': pf,
        'exact_cause_f1': ef1(ecp, ecr),
        'exact_effect_f1': ef1(eep, eer),
        'exact_pair_f1': ef1(epp, epr),
        'cosine_cause_p': cos_cp,
        'cosine_cause_r': cos_cr,
        'cosine_cause_f1': cos_cf,
        'cosine_effect_p': cos_ep,
        'cosine_effect_r': cos_er,
        'cosine_effect_f1': cos_ef,
        'cosine_pair_p': cos_pp,
        'cosine_pair_r': cos_pr,
        'cosine_pair_f1': cos_pf,
        'cosine_cause_above': cos_cause_abv,
        'cosine_effect_above': cos_effect_abv,
        'cosine_pair_above': cos_pair_abv,
        'cosine_threshold': cosine_threshold,
        'total_pred_pairs': total_pred_pairs,
        'total_gold_pairs': total_gold_pairs,
        'total_samples': len(rows),
        'metric_note': 'cosine_pair_f1 is raw-score F1 exactly like the previous evaluator; cosine_pair_above is the thresholded precision-side rate.',
    }


# ---------------------------------------------------------------------
# Requested row-level match flags
# ---------------------------------------------------------------------

def exact_pair_match(pred_pair: Dict[str, Any], gold_pair: Dict[str, Any]) -> int:
    pc, pe = pair_tuple(pred_pair)
    gc, ge = pair_tuple(gold_pair)
    return int(pc == gc and pe == ge)


def pair_token_score(pred_pair: Dict[str, Any], gold_pair: Dict[str, Any]) -> float:
    pc, pe = pair_tuple(pred_pair)
    gc, ge = pair_tuple(gold_pair)
    return (_token_f1(pc, gc) + _token_f1(pe, ge)) / 2


def pair_cosine_score(pred_pair: Dict[str, Any], gold_pair: Dict[str, Any]) -> float:
    if _cosine_model is None:
        return -1.0
    pc, pe = pair_tuple(pred_pair)
    gc, ge = pair_tuple(gold_pair)
    return (_best_cosine(pc, [gc]) + _best_cosine(pe, [ge])) / 2


def best_for_gold(gold_pair: Dict[str, Any], pred_pairs: List[Dict[str, Any]], cosine_threshold: float) -> Dict[str, Any]:
    pred_pairs = [p for p in pred_pairs if valid_pair(p)]
    if not pred_pairs:
        return {
            'exact_match': 0,
            'soft_token_overlap_match': 0,
            'cosine_similarity_match': 0 if _cosine_model is not None else '',
            'soft_pair_score': 0.0,
            'cosine_pair_score': -1.0 if _cosine_model is not None else '',
            'best_pred_pair_index': '',
            'best_pred_cause': '',
            'best_pred_effect': '',
        }

    exact = int(any(exact_pair_match(p, gold_pair) for p in pred_pairs))
    soft_scores = [pair_token_score(p, gold_pair) for p in pred_pairs]
    best_idx = int(np.argmax(soft_scores))
    best_soft = float(soft_scores[best_idx])

    out = {
        'exact_match': exact,
        'soft_token_overlap_match': int(best_soft > 0.0),
        'soft_pair_score': best_soft,
        'best_pred_pair_index': best_idx,
        'best_pred_cause': pred_pairs[best_idx].get('cause', ''),
        'best_pred_effect': pred_pairs[best_idx].get('effect', ''),
    }

    if _cosine_model is not None:
        cos_scores = [pair_cosine_score(p, gold_pair) for p in pred_pairs]
        cos_idx = int(np.argmax(cos_scores))
        best_cos = float(cos_scores[cos_idx])
        out.update({
            'cosine_similarity_match': int(best_cos >= cosine_threshold),
            'cosine_pair_score': best_cos,
            'best_pred_pair_index_cosine': cos_idx,
            'best_pred_cause_cosine': pred_pairs[cos_idx].get('cause', ''),
            'best_pred_effect_cosine': pred_pairs[cos_idx].get('effect', ''),
        })
    else:
        out.update({
            'cosine_similarity_match': '',
            'cosine_pair_score': '',
            'best_pred_pair_index_cosine': '',
            'best_pred_cause_cosine': '',
            'best_pred_effect_cosine': '',
        })

    return out


def best_for_prediction(pred_pair: Dict[str, Any], gold_pairs: List[Dict[str, Any]], cosine_threshold: float) -> Dict[str, Any]:
    gold_pairs = [g for g in gold_pairs if valid_pair(g)]
    if not gold_pairs:
        return {
            'exact_match': 0,
            'soft_token_overlap_match': 0,
            'cosine_similarity_match': 0 if _cosine_model is not None else '',
            'soft_pair_score': 0.0,
            'cosine_pair_score': -1.0 if _cosine_model is not None else '',
            'best_gold_pair_index': '',
            'best_gold_cause': '',
            'best_gold_effect': '',
        }

    exact = int(any(exact_pair_match(pred_pair, g) for g in gold_pairs))
    soft_scores = [pair_token_score(pred_pair, g) for g in gold_pairs]
    best_idx = int(np.argmax(soft_scores))
    best_soft = float(soft_scores[best_idx])

    out = {
        'exact_match': exact,
        'soft_token_overlap_match': int(best_soft > 0.0),
        'soft_pair_score': best_soft,
        'best_gold_pair_index': best_idx,
        'best_gold_cause': gold_pairs[best_idx].get('cause', ''),
        'best_gold_effect': gold_pairs[best_idx].get('effect', ''),
    }

    if _cosine_model is not None:
        cos_scores = [pair_cosine_score(pred_pair, g) for g in gold_pairs]
        cos_idx = int(np.argmax(cos_scores))
        best_cos = float(cos_scores[cos_idx])
        out.update({
            'cosine_similarity_match': int(best_cos >= cosine_threshold),
            'cosine_pair_score': best_cos,
            'best_gold_pair_index_cosine': cos_idx,
            'best_gold_cause_cosine': gold_pairs[cos_idx].get('cause', ''),
            'best_gold_effect_cosine': gold_pairs[cos_idx].get('effect', ''),
        })
    else:
        out.update({
            'cosine_similarity_match': '',
            'cosine_pair_score': '',
            'best_gold_pair_index_cosine': '',
            'best_gold_cause_cosine': '',
            'best_gold_effect_cosine': '',
        })

    return out


def build_flag_tables(rows: List[Dict[str, Any]], out_dir: str, cosine_threshold: float):
    gold_rows, pred_rows, sample_rows = [], [], []

    for row in rows:
        sn = row['s/n']
        sample_index = row['sample_index']
        sentence = row['sentence']
        gold_pairs = [g for g in row['gold_pairs'] if valid_pair(g)]
        pred_pairs = [p for p in row['pred_pairs'] if valid_pair(p)]

        gold_flags = []
        for gi, g in enumerate(gold_pairs):
            flags = best_for_gold(g, pred_pairs, cosine_threshold)
            gold_flags.append(flags)
            gold_rows.append({
                's/n': sn,
                'sample_index': sample_index,
                'sentence': sentence,
                'gold_pair_index': gi,
                'gold_cause': g.get('cause', ''),
                'gold_effect': g.get('effect', ''),
                'gold_causality': g.get('causality', ''),
                'gold_sententiality': g.get('sententiality', ''),
                **flags,
            })

        pred_flags = []
        for pi, p in enumerate(pred_pairs):
            flags = best_for_prediction(p, gold_pairs, cosine_threshold)
            pred_flags.append(flags)
            pred_rows.append({
                's/n': sn,
                'sample_index': sample_index,
                'sentence': sentence,
                'pred_pair_index': pi,
                'pred_cause': p.get('cause', ''),
                'pred_effect': p.get('effect', ''),
                'pred_causality': p.get('causality', ''),
                'pred_sententiality': p.get('sententiality', ''),
                **flags,
            })

        sample_rows.append({
            's/n': sn,
            'sample_index': sample_index,
            'sentence': sentence,
            'num_gold_pairs': len(gold_pairs),
            'num_pred_pairs': len(pred_pairs),
            'exact_match': int(any(x.get('exact_match', 0) for x in gold_flags)),
            'soft_token_overlap_match': int(any(x.get('soft_token_overlap_match', 0) for x in gold_flags)),
            'cosine_similarity_match': (
                int(any(x.get('cosine_similarity_match', 0) for x in gold_flags))
                if _cosine_model is not None else ''
            ),
            'best_soft_pair_score': max([x.get('soft_pair_score', 0.0) for x in gold_flags], default=0.0),
            'best_cosine_pair_score': (
                max([x.get('cosine_pair_score', -1.0) for x in gold_flags], default=-1.0)
                if _cosine_model is not None else ''
            ),
        })

    gold_df = pd.DataFrame(gold_rows)
    pred_df = pd.DataFrame(pred_rows)
    sample_df = pd.DataFrame(sample_rows)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    gold_df.to_csv(os.path.join(out_dir, 'gold_pair_match_flags.csv'), index=False)
    pred_df.to_csv(os.path.join(out_dir, 'prediction_pair_match_flags.csv'), index=False)
    sample_df.to_csv(os.path.join(out_dir, 'sample_match_flags.csv'), index=False)
    return gold_df, pred_df, sample_df


def main():
    ap = argparse.ArgumentParser(description='Two-file comparison using the same metric logic as your previous reevaluation script.')
    ap.add_argument('--pred', required=True, help='Prediction JSON file')
    ap.add_argument('--gold', required=True, help='Gold/reference test.json file')
    ap.add_argument('--out_dir', default='two_file_previous_logic_outputs', help='Output directory')
    ap.add_argument('--cosine', action='store_true', help='Enable cosine with BioBERT/BioSentBERT')
    ap.add_argument('--cosine_model', default='pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')
    ap.add_argument('--cosine_threshold', type=float, default=0.75)
    args = ap.parse_args()

    if args.cosine:
        load_cosine_model(args.cosine_model)

    rows = aligned_pairs(args.pred, args.gold)
    gold_df, pred_df, sample_df = build_flag_tables(rows, args.out_dir, args.cosine_threshold)
    metrics = evaluate_previous_style(rows, args.cosine_threshold)

    metrics.update({
        'prediction_file': os.path.basename(args.pred),
        'gold_file': os.path.basename(args.gold),
        'cosine_enabled_requested': bool(args.cosine),
        'cosine_model_loaded': _cosine_model is not None,
        'cosine_model': args.cosine_model if args.cosine else None,
        'gold_pair_match_flags_csv': os.path.join(args.out_dir, 'gold_pair_match_flags.csv'),
        'prediction_pair_match_flags_csv': os.path.join(args.out_dir, 'prediction_pair_match_flags.csv'),
        'sample_match_flags_csv': os.path.join(args.out_dir, 'sample_match_flags.csv'),
    })

    summary_json = os.path.join(args.out_dir, 'previous_logic_summary.json')
    summary_csv = os.path.join(args.out_dir, 'previous_logic_summary.csv')
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame([metrics]).to_csv(summary_csv, index=False)

    print('\nSaved outputs:')
    print(f"  {os.path.join(args.out_dir, 'gold_pair_match_flags.csv')}")
    print(f"  {os.path.join(args.out_dir, 'prediction_pair_match_flags.csv')}")
    print(f"  {os.path.join(args.out_dir, 'sample_match_flags.csv')}")
    print(f"  {summary_json}")
    print(f"  {summary_csv}")

    print('\nPrevious-style metric summary:')
    for k in [
        'total_samples', 'total_pred_pairs', 'total_gold_pairs',
        'soft_pair_p', 'soft_pair_r', 'soft_pair_f1',
        'exact_pair_f1',
        'cosine_pair_p', 'cosine_pair_r', 'cosine_pair_f1', 'cosine_pair_above'
    ]:
        print(f"  {k}: {metrics.get(k)}")


if __name__ == '__main__':
    main()

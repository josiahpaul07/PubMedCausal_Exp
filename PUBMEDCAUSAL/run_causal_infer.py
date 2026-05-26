#!/usr/bin/env python3
# coding: utf-8
"""
run_causal_infer_v2.py
======================
Test BERT models AND fine-tuned LLM models on external CSV datasets.

Supported model types:
  --model_type bert       -> AutoModelForSequenceClassification (detection only)
  --model_type llm        -> AutoModelForCausalLM (detection or extraction)

Tasks:
  --task detection        -> outputs binary label (0/1), computes P/R/F1
  --task extraction       -> outputs cause/effect pairs, computes Exact, Token F1,
                             and (optionally) BioSentBERT Cosine F1

Extraction metrics (3 tiers):
  Exact F1     -- strict string match after normalisation
  Token F1     -- SQuAD-style partial word overlap (soft)
  Cosine F1    -- BioSentBERT semantic similarity F1 (requires --cosine)

Usage examples:
  # BERT detection (original behaviour, unchanged)
  python run_causal_infer_v2.py \
      --input_csv data.csv \
      --model_dirs ./checkpoints/detection_scibert/final \
      --model_type bert \
      --task detection

  # Fine-tuned LLM extraction with all metrics
  python run_causal_infer_v2.py \
      --input_csv data.csv \
      --model_dirs ./checkpoints/extraction_combined_llama3b/final \
      --model_type llm \
      --task extraction \
      --strategy zero-shot \
      --batch_size 8 \
      --cosine \
      --cosine_threshold 0.75

  # Multiple LLM models at once
  python run_causal_infer_v2.py \
      --input_csv data.csv \
      --model_dirs ./checkpoints/detection_llama3b/final ./checkpoints/detection_mistral7b/final \
      --model_type llm \
      --task detection \
      --strategy zero-shot
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)

# =============================================================
# Shared utilities
# =============================================================

def has_config(path):
    return os.path.isfile(os.path.join(path, "config.json"))

def is_peft_adapter(path):
    return os.path.isfile(os.path.join(path, "adapter_config.json"))

def get_peft_base_model(adapter_path):
    with open(os.path.join(adapter_path, "adapter_config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    base = cfg.get("base_model_name_or_path", "")
    if not base:
        raise ValueError(f"adapter_config.json at {adapter_path} missing 'base_model_name_or_path'.")
    return base

def resolve_model_dir(path):
    # If it looks like a HuggingFace model ID (e.g. "meta-llama/Llama-3.2-3B-Instruct"),
    # pass it through directly -- transformers will handle the download/cache
    if not os.path.exists(path) and '/' in path and not path.startswith('./') and not path.startswith('/'):
        return path
    path = os.path.normpath(path)
    if has_config(path): return path
    if is_peft_adapter(path): return path
    final_dir = os.path.join(path, "final")
    if os.path.isdir(final_dir) and (has_config(final_dir) or is_peft_adapter(final_dir)):
        return final_dir
    ckpts = [c for c in glob.glob(os.path.join(path, "checkpoint-*")) if os.path.isdir(c)]
    if not ckpts:
        raise FileNotFoundError(f"No loadable model found in {path}.")
    def step_num(p):
        m = re.search(r"checkpoint-(\d+)$", p)
        return int(m.group(1)) if m else -1
    latest = sorted(ckpts, key=step_num)[-1]
    if not has_config(latest) and not is_peft_adapter(latest):
        raise FileNotFoundError(f"Latest checkpoint missing config: {latest}")
    return latest

def pick_text_col(df, requested):
    if requested != "auto":
        if requested not in df.columns:
            raise ValueError(f"Text column '{requested}' not found.")
        return requested
    for c in ["Sentences", "Sentence", "text", "Text", "content", "Content"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if df[c].dtype == "object":
            return c
    raise ValueError("Could not auto-detect a text column. Use --text_col <name>.")

def nonempty(x):
    if x is None: return False
    if isinstance(x, float) and np.isnan(x): return False
    return str(x).strip() != ""

def build_gold_detection(df):
    c1 = df["Cause1"] if "Cause1" in df.columns else pd.Series([None]*len(df))
    e1 = df["Effect1"] if "Effect1" in df.columns else pd.Series([None]*len(df))
    c2 = df["Cause2"] if "Cause2" in df.columns else pd.Series([None]*len(df))
    e2 = df["Effect2"] if "Effect2" in df.columns else pd.Series([None]*len(df))
    y = [1 if ((nonempty(a) and nonempty(b)) or (nonempty(c) and nonempty(d))) else 0
         for a, b, c, d in zip(c1, e1, c2, e2)]
    return np.array(y, dtype=int)

def build_gold_extraction(df):
    gold = []
    for _, row in df.iterrows():
        pairs = []
        for i in ["1", "2"]:
            c, e = row.get(f"Cause{i}"), row.get(f"Effect{i}")
            if nonempty(c) and nonempty(e):
                pairs.append({"cause": str(c).strip(), "effect": str(e).strip()})
        gold.append({"pairs": pairs})
    return gold

def confusion_counts(y_true, y_pred):
    tn = int(((y_true==0)&(y_pred==0)).sum())
    fp = int(((y_true==0)&(y_pred==1)).sum())
    fn = int(((y_true==1)&(y_pred==0)).sum())
    tp = int(((y_true==1)&(y_pred==1)).sum())
    return tn, fp, fn, tp

def prf_from_counts(tn, fp, fn, tp):
    p = tp/(tp+fp) if (tp+fp)>0 else 0.0
    r = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f = 2*p*r/(p+r) if (p+r)>0 else 0.0
    return p, r, f


# =============================================================
# Token F1 helpers (SQuAD-style)
# =============================================================

def _normalize_span(text):
    if isinstance(text, list): text = text[0] if text else ''
    if text is None: return ''
    text = str(text).lower().strip()
    text = re.sub(r'^(the|a|an)\s+', '', text)
    text = text.strip('.,;:!?"\'-()')
    return text.strip()

def _tokenize(text):
    return _normalize_span(text).split()

def _token_f1(pred, gold):
    p_tok = _tokenize(pred)
    g_tok = _tokenize(gold)
    if not p_tok and not g_tok: return 1.0
    if not p_tok or not g_tok:  return 0.0
    pc, gc = defaultdict(int), defaultdict(int)
    for t in p_tok: pc[t] += 1
    for t in g_tok: gc[t] += 1
    overlap = sum(min(pc[t], gc[t]) for t in pc)
    if overlap == 0: return 0.0
    p = overlap / len(p_tok)
    r = overlap / len(g_tok)
    return 2*p*r/(p+r)

def _best_tf1(pred, golds):
    if not golds: return 0.0
    return max(_token_f1(pred, g) for g in golds)


# =============================================================
# BioSentBERT cosine similarity helpers
# =============================================================

_cosine_model     = None
_cosine_tokenizer = None

def load_cosine_model(model_name='pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb'):
    global _cosine_model, _cosine_tokenizer
    if _cosine_model is not None: return
    try:
        from transformers import AutoModel
        print(f"  Loading BioSentBERT: {model_name}...")
        _cosine_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _cosine_model     = AutoModel.from_pretrained(model_name)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _cosine_model.to(device)
        _cosine_model.eval()
        print(f"  BioSentBERT loaded on {device}")
    except Exception as e:
        print(f"  WARNING: Could not load BioSentBERT ({e}). Cosine scores will be skipped.")
        _cosine_model = None

def _get_embeddings(texts):
    if _cosine_model is None or not texts: return None
    device = next(_cosine_model.parameters()).device
    enc = _cosine_tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = _cosine_model(**enc)
    tok_emb = out.last_hidden_state
    mask    = enc['attention_mask'].unsqueeze(-1).expand(tok_emb.size()).float()
    pooled  = torch.sum(tok_emb * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
    norms   = pooled.norm(dim=1, keepdim=True).clamp(min=1e-9)
    return (pooled / norms).cpu().numpy()

def _best_cosine(pred, golds):
    if _cosine_model is None or not golds: return -1.0
    texts = [_normalize_span(pred)] + [_normalize_span(g) for g in golds]
    embs  = _get_embeddings(texts)
    if embs is None: return -1.0
    return float(max(np.dot(embs[0], embs[i+1]) for i in range(len(golds))))


# =============================================================
# BERT inference
# =============================================================

def infer_bert(model_dir, texts, batch_size, max_length, threshold, device):
    used_dir  = resolve_model_dir(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(used_dir, use_fast=True)
    model     = AutoModelForSequenceClassification.from_pretrained(used_dir)
    model.to(device); model.eval()
    probs, preds = [], []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(texts[i:i+batch_size], padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            if   logits.shape[-1] == 2: p = torch.softmax(logits, dim=-1)[:,1]
            elif logits.shape[-1] == 1: p = torch.sigmoid(logits).squeeze(-1)
            else: raise ValueError(f"Unexpected num_labels={logits.shape[-1]}")
            p_cpu = p.detach().cpu().numpy()
            probs.extend(p_cpu.tolist())
            preds.extend((p_cpu >= threshold).astype(int).tolist())
    return np.array(probs, dtype=float), np.array(preds, dtype=int), used_dir


# =============================================================
# Prompt builders
# =============================================================

def detection_prompt(strategy, sentence):
    return (
        "You are an expert in causal reasoning. "
        "Determine whether the following sentence expresses a causal relationship.\n"
        "Answer with only 'Yes' or 'No'.\n\n"
        f"Sentence: {sentence}\n\nAnswer:"
    )

def extraction_prompt(strategy, sentence):
    return (
        "You are an expert in causal information extraction.\n"
        "Extract all cause-effect pairs from the sentence below.\n"
        "Return ONLY a JSON object with this exact format:\n"
        '{"pairs": [{"cause": "...", "effect": "..."}]}\n'
        'If no causal relationship exists, return: {"pairs": []}\n\n'
        f"Sentence: {sentence}\n\nJSON:"
    )


# =============================================================
# Response parsers
# =============================================================

def parse_detection_response(response):
    text = response.strip().lower()
    if re.search(r'\byes\b', text): return 1
    if re.search(r'\bno\b',  text): return 0
    first = text.split()[0] if text.split() else ''
    return 1 if first.startswith('y') else 0

def normalize_field(value):
    if isinstance(value, list): value = value[0] if value else ''
    if value is None: return ''
    return str(value).strip().lower()

def parse_extraction_response(response):
    text = response.strip()
    text = re.sub(r'^```(?:json)?', '', text, flags=re.MULTILINE).strip()
    text = re.sub(r'```$', '',        text, flags=re.MULTILINE).strip()
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match: return []
    try:
        data  = json.loads(match.group())
        pairs = data.get('pairs', [])
        if not isinstance(pairs, list): return []
        result = []
        for p in pairs:
            if isinstance(p, dict):
                c = normalize_field(p.get('cause', ''))
                e = normalize_field(p.get('effect',''))
                if c and e: result.append({'cause': c, 'effect': e})
        return result
    except (json.JSONDecodeError, TypeError):
        return []


# =============================================================
# LLM inference
# =============================================================

def infer_llm(model_dir, texts, task, strategy, batch_size, max_length, device):
    used_dir  = resolve_model_dir(model_dir)
    peft_mode = is_peft_adapter(used_dir)

    if peft_mode:
        base_model_path = get_peft_base_model(used_dir)
        print(f"  Detected LoRA adapter. Base model: {base_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(used_dir)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    if peft_mode:
        try:
            from peft import PeftModel
        except ImportError:
            raise ImportError("Install peft: pip install peft --break-system-packages")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16 if device=="cuda" else torch.float32,
            device_map="auto" if device=="cuda" else None,
        )
        model = PeftModel.from_pretrained(base_model, used_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            used_dir,
            torch_dtype=torch.float16 if device=="cuda" else torch.float32,
            device_map="auto" if device=="cuda" else None,
        )
    model.eval()

    max_new_tokens = 32 if task == 'detection' else 256
    outputs = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            prompts = ([detection_prompt(strategy, t) for t in batch_texts]
                       if task == 'detection' else
                       [extraction_prompt(strategy, t) for t in batch_texts])
            enc = tokenizer(prompts, return_tensors="pt", padding=True,
                            truncation=True, max_length=max_length)
            enc = {k: v.to(device) for k, v in enc.items()}
            gen = model.generate(
                **enc, max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False, use_cache=True, repetition_penalty=1.1,
            )
            for j, out_ids in enumerate(gen):
                input_len  = enc['input_ids'][j].shape[0]
                response   = tokenizer.decode(out_ids[input_len:], skip_special_tokens=True)
                outputs.append(parse_detection_response(response) if task=='detection'
                               else parse_extraction_response(response))
            if device == "cuda": torch.cuda.empty_cache()
            print(f"    Processed {min(i+batch_size, len(texts))}/{len(texts)}", end='\r')

    print()
    return outputs, used_dir


# =============================================================
# Extraction metrics: Exact F1 + Token F1 + Cosine F1
# =============================================================

def compute_extraction_metrics(pred_list, gold_list, cosine_threshold=0.75):
    """
    Three-tier extraction evaluation:
      Exact F1  -- strict normalised string match (corpus-level set overlap)
      Token F1  -- SQuAD-style partial word overlap (sample-level average)
      Cosine F1 -- BioSentBERT semantic similarity  (sample-level average)

    All three report proper P/R/F1:
      Precision: pred -> gold  (how many predictions are correct)
      Recall:    gold -> pred  (how many gold spans were found)
    """
    # Exact accumulators (corpus-level sets)
    all_pred_causes  = set(); all_gold_causes  = set()
    all_pred_effects = set(); all_gold_effects = set()
    all_pred_pairs   = set(); all_gold_pairs   = set()

    # Token F1 accumulators (sample-level lists)
    tok_cause_p_all  = []; tok_cause_r_all  = []
    tok_effect_p_all = []; tok_effect_r_all = []
    tok_pair_p_all   = []; tok_pair_r_all   = []

    # Cosine accumulators (sample-level lists)
    cos_cause_p_all  = []; cos_cause_r_all  = []
    cos_effect_p_all = []; cos_effect_r_all = []
    cos_pair_p_all   = []; cos_pair_r_all   = []

    use_cosine = _cosine_model is not None

    for pred_pairs, gold_item in zip(pred_list, gold_list):
        gold_pairs = gold_item.get('pairs', []) if isinstance(gold_item, dict) else []
        if not isinstance(gold_pairs, list): gold_pairs = []
        if not isinstance(pred_pairs, list): pred_pairs = []

        gc = [_normalize_span(g.get('cause', ''))  for g in gold_pairs if g.get('cause')]
        ge = [_normalize_span(g.get('effect',''))  for g in gold_pairs if g.get('effect')]
        gp = [(_normalize_span(g.get('cause','')), _normalize_span(g.get('effect','')))
              for g in gold_pairs if g.get('cause') and g.get('effect')]

        pc = [_normalize_span(p.get('cause', ''))  for p in pred_pairs if p.get('cause')]
        pe = [_normalize_span(p.get('effect',''))  for p in pred_pairs if p.get('effect')]
        pp = [(_normalize_span(p.get('cause','')), _normalize_span(p.get('effect','')))
              for p in pred_pairs if p.get('cause') and p.get('effect')]

        # Exact sets
        for x in pc: all_pred_causes.add(x)
        for x in gc: all_gold_causes.add(x)
        for x in pe: all_pred_effects.add(x)
        for x in ge: all_gold_effects.add(x)
        for x in pp: all_pred_pairs.add(x)
        for x in gp: all_gold_pairs.add(x)

        # Token F1 -- Cause
        for x in pc: tok_cause_p_all.append(_best_tf1(x, gc) if gc else 0.0)
        for x in gc: tok_cause_r_all.append(_best_tf1(x, pc) if pc else 0.0)
        # Token F1 -- Effect
        for x in pe: tok_effect_p_all.append(_best_tf1(x, ge) if ge else 0.0)
        for x in ge: tok_effect_r_all.append(_best_tf1(x, pe) if pe else 0.0)
        # Token F1 -- Pair
        for a, b in pp:
            s = max((_token_f1(a,ga)+_token_f1(b,gb))/2 for ga,gb in gp) if gp else 0.0
            tok_pair_p_all.append(s)
        for ga, gb in gp:
            s = max((_token_f1(a,ga)+_token_f1(b,gb))/2 for a,b in pp) if pp else 0.0
            tok_pair_r_all.append(s)

        # Cosine F1 -- Cause
        if use_cosine:
            for x in pc: cos_cause_p_all.append(_best_cosine(x, gc) if gc else 0.0)
            for x in gc: cos_cause_r_all.append(_best_cosine(x, pc) if pc else 0.0)
            # Cosine F1 -- Effect
            for x in pe: cos_effect_p_all.append(_best_cosine(x, ge) if ge else 0.0)
            for x in ge: cos_effect_r_all.append(_best_cosine(x, pe) if pe else 0.0)
            # Cosine F1 -- Pair
            for a, b in pp:
                s = max((_best_cosine(a,[ga])+_best_cosine(b,[gb]))/2 for ga,gb in gp) if gp else 0.0
                cos_pair_p_all.append(s)
            for ga, gb in gp:
                s = max((_best_cosine(ga,[a])+_best_cosine(gb,[b]))/2 for a,b in pp) if pp else 0.0
                cos_pair_r_all.append(s)

    # Aggregate helpers
    def set_prf(ps, gs):
        if not ps and not gs: return 1.0, 1.0, 1.0
        if not ps or  not gs: return 0.0, 0.0, 0.0
        cor = len(ps & gs)
        p = cor/len(ps); r = cor/len(gs)
        return p, r, 2*p*r/(p+r) if (p+r)>0 else 0.0

    def avg_prf(ps, rs):
        p = float(np.mean(ps)) if ps else 0.0
        r = float(np.mean(rs)) if rs else 0.0
        return p, r, 2*p*r/(p+r) if (p+r)>0 else 0.0

    ex_cp, ex_cr, ex_cf = set_prf(all_pred_causes,  all_gold_causes)
    ex_ep, ex_er, ex_ef = set_prf(all_pred_effects, all_gold_effects)
    ex_pp, ex_pr, ex_pf = set_prf(all_pred_pairs,   all_gold_pairs)

    tok_cp, tok_cr, tok_cf = avg_prf(tok_cause_p_all,  tok_cause_r_all)
    tok_ep, tok_er, tok_ef = avg_prf(tok_effect_p_all, tok_effect_r_all)
    tok_pp, tok_pr, tok_pf = avg_prf(tok_pair_p_all,   tok_pair_r_all)

    if use_cosine:
        cos_cp, cos_cr, cos_cf = avg_prf(cos_cause_p_all,  cos_cause_r_all)
        cos_ep, cos_er, cos_ef = avg_prf(cos_effect_p_all, cos_effect_r_all)
        cos_pp, cos_pr, cos_pf = avg_prf(cos_pair_p_all,   cos_pair_r_all)
        cos_c_abv = float(np.mean([s>=cosine_threshold for s in cos_cause_p_all]))  if cos_cause_p_all  else -1.0
        cos_e_abv = float(np.mean([s>=cosine_threshold for s in cos_effect_p_all])) if cos_effect_p_all else -1.0
        cos_p_abv = float(np.mean([s>=cosine_threshold for s in cos_pair_p_all]))   if cos_pair_p_all   else -1.0
    else:
        cos_cp=cos_cr=cos_cf=cos_ep=cos_er=cos_ef=cos_pp=cos_pr=cos_pf=-1.0
        cos_c_abv=cos_e_abv=cos_p_abv=-1.0

    return {
        'exact_cause_p':  ex_cp,  'exact_cause_r':  ex_cr,  'exact_cause_f1':  ex_cf,
        'exact_effect_p': ex_ep,  'exact_effect_r': ex_er,  'exact_effect_f1': ex_ef,
        'exact_pair_p':   ex_pp,  'exact_pair_r':   ex_pr,  'exact_pair_f1':   ex_pf,
        'token_cause_p':  tok_cp, 'token_cause_r':  tok_cr, 'token_cause_f1':  tok_cf,
        'token_effect_p': tok_ep, 'token_effect_r': tok_er, 'token_effect_f1': tok_ef,
        'token_pair_p':   tok_pp, 'token_pair_r':   tok_pr, 'token_pair_f1':   tok_pf,
        'cosine_cause_p': cos_cp, 'cosine_cause_r': cos_cr, 'cosine_cause_f1': cos_cf,
        'cosine_effect_p':cos_ep, 'cosine_effect_r':cos_er, 'cosine_effect_f1':cos_ef,
        'cosine_pair_p':  cos_pp, 'cosine_pair_r':  cos_pr, 'cosine_pair_f1':  cos_pf,
        'cosine_cause_above_thresh':  cos_c_abv,
        'cosine_effect_above_thresh': cos_e_abv,
        'cosine_pair_above_thresh':   cos_p_abv,
        'cosine_threshold': cosine_threshold,
        'pred_causes':  len(all_pred_causes),  'gold_causes':  len(all_gold_causes),
        'pred_effects': len(all_pred_effects), 'gold_effects': len(all_gold_effects),
        'pred_pairs':   len(all_pred_pairs),   'gold_pairs':   len(all_gold_pairs),
        'total_samples': len(gold_list),
    }


# =============================================================
# Main
# =============================================================

def main():
    ap = argparse.ArgumentParser(
        description='Test BERT or fine-tuned LLM models on external CSV datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input_csv",          required=True)
    ap.add_argument("--text_col",           default="auto")
    ap.add_argument("--csv_encoding",       default=None)
    ap.add_argument("--model_dirs",         nargs="+", required=True)
    ap.add_argument("--model_type",         choices=["bert","llm"], default="bert")
    ap.add_argument("--task",               choices=["detection","extraction"], default="detection")
    ap.add_argument("--strategy",           nargs="+", default=["zero-shot"],
                    choices=["zero-shot","few-shot","cot","cot-fewshot","react","least-to-most"],
                    help="One or more prompting strategies (default: zero-shot)")
    ap.add_argument("--batch_size",         type=int,   default=32)
    ap.add_argument("--max_length",         type=int,   default=256)
    ap.add_argument("--threshold",          type=float, default=0.5)
    ap.add_argument("--cosine",             action='store_true',
                    help="Enable BioSentBERT cosine F1 for extraction")
    ap.add_argument("--cosine_model",       default='pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')
    ap.add_argument("--cosine_threshold",   type=float, default=0.75)
    ap.add_argument("--out_csv",            default="predictions.csv")
    ap.add_argument("--save_metrics_csv",   default="metrics_summary.csv")
    ap.add_argument("--save_metrics_json",  default=None)
    args = ap.parse_args()

    if args.model_type == "bert" and args.task == "extraction":
        ap.error("BERT only supports --task detection.")

    if args.cosine and args.task == "extraction":
        load_cosine_model(args.cosine_model)

    df = (pd.read_csv(args.input_csv, encoding=args.csv_encoding)
          if args.csv_encoding else pd.read_csv(args.input_csv))
    text_col = pick_text_col(df, args.text_col)
    texts    = df[text_col].fillna("").astype(str).tolist()
    print(f"Loaded {len(texts)} rows from {args.input_csv} (text col: '{text_col}')")

    y_true_detection  = build_gold_detection(df)
    y_true_extraction = build_gold_extraction(df) if args.task == "extraction" else None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Model type: {args.model_type} | Task: {args.task}" +
          (f" | Strategy: {args.strategy}" if args.model_type == "llm" else ""))

    out  = df.copy()
    rows = []

    for model_dir in args.model_dirs:
        base = os.path.basename(os.path.normpath(model_dir))
        print(f"\n{'='*60}\nModel dir : {model_dir}")

        if args.model_type == "bert":
            probs, y_pred, used_dir = infer_bert(model_dir, texts, args.batch_size,
                                                  args.max_length, args.threshold, device)
            used_leaf  = os.path.basename(os.path.normpath(used_dir))
            model_name = f"{base}/{used_leaf}" if used_leaf.startswith("checkpoint-") else base
            out[base+"_prob_causal"] = probs
            out[base+"_pred_causal"] = y_pred
            tn, fp, fn, tp = confusion_counts(y_true_detection, y_pred)
            p, r, f = prf_from_counts(tn, fp, fn, tp)
            acc = (tp+tn)/max(tp+tn+fp+fn,1)
            print(f"Loaded: {used_dir}")
            print(f"TN={tn} FP={fp} FN={fn} TP={tp}")
            print(f"P={p:.4f} R={r:.4f} F1={f:.4f} Acc={acc:.4f}")
            rows.append({"model":model_name,"model_type":"bert","task":"detection","strategy":"n/a",
                         "loaded_path":used_dir,"threshold":args.threshold,
                         "tn":tn,"fp":fp,"fn":fn,"tp":tp,
                         "precision":p,"recall":r,"f1":f,"accuracy":acc,
                         "support_total":len(y_true_detection),
                         "support_pos":int((y_true_detection==1).sum()),
                         "support_neg":int((y_true_detection==0).sum())})

        else:
            # Load model ONCE, then loop over all strategies
            used_dir  = resolve_model_dir(model_dir)
            peft_mode = is_peft_adapter(used_dir)

            if peft_mode:
                base_model_path = get_peft_base_model(used_dir)
                print(f"  Detected LoRA adapter. Base model: {base_model_path}")
                tokenizer = AutoTokenizer.from_pretrained(base_model_path)
            else:
                tokenizer = AutoTokenizer.from_pretrained(used_dir)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'left'

            if peft_mode:
                try:
                    from peft import PeftModel
                except ImportError:
                    raise ImportError("Install peft: pip install peft --break-system-packages")
                base_mdl = AutoModelForCausalLM.from_pretrained(
                    base_model_path,
                    torch_dtype=torch.float16 if device=="cuda" else torch.float32,
                    device_map="auto" if device=="cuda" else None,
                )
                llm_model = PeftModel.from_pretrained(base_mdl, used_dir)
            else:
                llm_model = AutoModelForCausalLM.from_pretrained(
                    used_dir,
                    torch_dtype=torch.float16 if device=="cuda" else torch.float32,
                    device_map="auto" if device=="cuda" else None,
                )
            llm_model.eval()
            print(f"Loaded: {used_dir}")

            used_leaf  = os.path.basename(os.path.normpath(used_dir))
            model_name = f"{base}/{used_leaf}" if used_leaf.startswith("checkpoint-") else base
            max_new_tokens = 32 if args.task == 'detection' else 256

            for strategy in args.strategy:
                print(f"\n  -- Strategy: {strategy} --")
                raw_outputs = []

                with torch.no_grad():
                    for i in range(0, len(texts), args.batch_size):
                        batch_texts = texts[i:i+args.batch_size]
                        prompts = ([detection_prompt(strategy, t) for t in batch_texts]
                                   if args.task == 'detection' else
                                   [extraction_prompt(strategy, t) for t in batch_texts])
                        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                                        truncation=True, max_length=args.max_length)
                        enc = {k: v.to(device) for k, v in enc.items()}
                        gen = llm_model.generate(
                            **enc, max_new_tokens=max_new_tokens,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            do_sample=False, use_cache=True, repetition_penalty=1.1,
                        )
                        for j, out_ids in enumerate(gen):
                            input_len = enc['input_ids'][j].shape[0]
                            response  = tokenizer.decode(out_ids[input_len:], skip_special_tokens=True)
                            raw_outputs.append(parse_detection_response(response) if args.task=='detection'
                                               else parse_extraction_response(response))
                        if device == "cuda": torch.cuda.empty_cache()
                        print(f"    Processed {min(i+args.batch_size, len(texts))}/{len(texts)}", end='\r')
                print()

                if args.task == "detection":
                    y_pred = np.array(raw_outputs, dtype=int)
                    out[f"{base}_{strategy}_pred_causal"] = y_pred
                    tn, fp, fn, tp = confusion_counts(y_true_detection, y_pred)
                    p, r, f = prf_from_counts(tn, fp, fn, tp)
                    acc = (tp+tn)/max(tp+tn+fp+fn,1)
                    print(f"  TN={tn} FP={fp} FN={fn} TP={tp}")
                    print(f"  P={p:.4f} R={r:.4f} F1={f:.4f} Acc={acc:.4f}")
                    rows.append({"model":model_name,"model_type":"llm","task":"detection",
                                 "strategy":strategy,"loaded_path":used_dir,"threshold":"n/a",
                                 "tn":tn,"fp":fp,"fn":fn,"tp":tp,
                                 "precision":p,"recall":r,"f1":f,"accuracy":acc,
                                 "support_total":len(y_true_detection),
                                 "support_pos":int((y_true_detection==1).sum()),
                                 "support_neg":int((y_true_detection==0).sum())})
                else:
                    out[f"{base}_{strategy}_pred_cause"]  = ["; ".join(p["cause"]  for p in ps) if ps else "" for ps in raw_outputs]
                    out[f"{base}_{strategy}_pred_effect"] = ["; ".join(p["effect"] for p in ps) if ps else "" for ps in raw_outputs]

                    m = compute_extraction_metrics(raw_outputs, y_true_extraction, args.cosine_threshold)

                    W = 10
                    print(f"\n  {'Metric':<22} {'P':>{W}} {'R':>{W}} {'F1':>{W}}")
                    print(f"  {'-'*54}")
                    print(f"  {'Exact  Cause':<22} {m['exact_cause_p']:>{W}.4f} {m['exact_cause_r']:>{W}.4f} {m['exact_cause_f1']:>{W}.4f}")
                    print(f"  {'Exact  Effect':<22} {m['exact_effect_p']:>{W}.4f} {m['exact_effect_r']:>{W}.4f} {m['exact_effect_f1']:>{W}.4f}")
                    print(f"  {'Exact  Pair':<22} {m['exact_pair_p']:>{W}.4f} {m['exact_pair_r']:>{W}.4f} {m['exact_pair_f1']:>{W}.4f}")
                    print(f"  {'-'*54}")
                    print(f"  {'Token  Cause':<22} {m['token_cause_p']:>{W}.4f} {m['token_cause_r']:>{W}.4f} {m['token_cause_f1']:>{W}.4f}")
                    print(f"  {'Token  Effect':<22} {m['token_effect_p']:>{W}.4f} {m['token_effect_r']:>{W}.4f} {m['token_effect_f1']:>{W}.4f}")
                    print(f"  {'Token  Pair':<22} {m['token_pair_p']:>{W}.4f} {m['token_pair_r']:>{W}.4f} {m['token_pair_f1']:>{W}.4f}")
                    if _cosine_model is not None:
                        print(f"  {'-'*54}")
                        print(f"  {'Cosine Cause':<22} {m['cosine_cause_p']:>{W}.4f} {m['cosine_cause_r']:>{W}.4f} {m['cosine_cause_f1']:>{W}.4f}")
                        print(f"  {'Cosine Effect':<22} {m['cosine_effect_p']:>{W}.4f} {m['cosine_effect_r']:>{W}.4f} {m['cosine_effect_f1']:>{W}.4f}")
                        print(f"  {'Cosine Pair':<22} {m['cosine_pair_p']:>{W}.4f} {m['cosine_pair_r']:>{W}.4f} {m['cosine_pair_f1']:>{W}.4f}")
                        print(f"  Threshold={args.cosine_threshold}  "
                              f"Cause>={args.cosine_threshold}: {m['cosine_cause_above_thresh']:.2%}  "
                              f"Pair>={args.cosine_threshold}: {m['cosine_pair_above_thresh']:.2%}")

                    rows.append({"model":model_name,"model_type":"llm","task":"extraction",
                                 "strategy":strategy,"loaded_path":used_dir, **m})

    # Save outputs
    out.to_csv(args.out_csv, index=False)
    print(f"\nSaved predictions : {args.out_csv}")
    pd.DataFrame(rows).to_csv(args.save_metrics_csv, index=False)
    print(f"Saved metrics     : {args.save_metrics_csv}")
    if args.save_metrics_json:
        with open(args.save_metrics_json, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print(f"Saved metrics JSON: {args.save_metrics_json}")

    # Final summary
    print("\n" + "="*80 + "\nFINAL SUMMARY\n" + "="*80)
    for row in rows:
        print(f"\nModel    : {row['model']}")
        print(f"Type     : {row['model_type']} | Task: {row['task']} | Strategy: {row.get('strategy','n/a')}")
        if row['task'] == 'detection':
            print(f"P={row['precision']:.4f}  R={row['recall']:.4f}  F1={row['f1']:.4f}  Acc={row['accuracy']:.4f}")
        else:
            print(f"  Exact  -- Cause F1:{row['exact_cause_f1']:.4f}  Effect F1:{row['exact_effect_f1']:.4f}  Pair F1:{row['exact_pair_f1']:.4f}")
            print(f"  Token  -- Cause F1:{row['token_cause_f1']:.4f}  Effect F1:{row['token_effect_f1']:.4f}  Pair F1:{row['token_pair_f1']:.4f}")
            if row.get('cosine_pair_f1', -1.0) >= 0:
                print(f"  Cosine -- Cause F1:{row['cosine_cause_f1']:.4f}  Effect F1:{row['cosine_effect_f1']:.4f}  Pair F1:{row['cosine_pair_f1']:.4f}")
    print("="*80)


if __name__ == "__main__":
    main()
"""
Evaluation Metrics for Causal Extraction Task
Metrics:
1. Cause Extraction: Precision, Recall, F1
2. Effect Extraction: Precision, Recall, F1
3. Pair Matching: Precision, Recall, F1
4. Switching Rate
5. Causality Classification: Precision, Recall, F1 (Implicit vs Explicit)
6. Sententiality Classification: Precision, Recall, F1 (Intra vs Inter)
FIXED: Handles list values returned by Mistral and other models
"""

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import json


class ExtractionEvaluator:
    """Evaluate causal extraction performance"""
    
    def __init__(self):
        self.results = {}
    
    @staticmethod
    def normalize_text(text) -> str:
        """Normalize text for comparison - handles strings, lists, and other types"""
        # Handle list - take first element
        if isinstance(text, list):
            text = text[0] if text else ''
        # Handle None
        if text is None:
            return ''
        # Handle non-string
        if not isinstance(text, str):
            text = str(text)
        return text.strip().lower()
    
    def extract_causes_effects(self, pairs: List[Dict]) -> Tuple[Set[str], Set[str]]:
        """Extract causes and effects from pairs - handles list values"""
        causes = set()
        effects = set()
        
        for pair in pairs:
            if 'cause' in pair and pair['cause']:
                cause = self.normalize_text(pair['cause'])
                if cause:
                    causes.add(cause)
            if 'effect' in pair and pair['effect']:
                effect = self.normalize_text(pair['effect'])
                if effect:
                    effects.add(effect)
        
        return causes, effects
    
    def extract_pairs(self, pairs: List[Dict]) -> Set[Tuple[str, str]]:
        """Extract cause-effect pairs as tuples - handles list values"""
        pair_tuples = set()
        
        for pair in pairs:
            if 'cause' in pair and 'effect' in pair and pair['cause'] and pair['effect']:
                cause = self.normalize_text(pair['cause'])
                effect = self.normalize_text(pair['effect'])
                if cause and effect:
                    pair_tuples.add((cause, effect))
        
        return pair_tuples
    
    def check_switching(self, pred_pairs: List[Dict], gt_pairs: List[Dict]) -> int:
        """Check if cause and effect are switched. Returns count of switched pairs."""
        gt_set = self.extract_pairs(gt_pairs)
        
        switched_count = 0
        for pair in pred_pairs:
            if 'cause' in pair and 'effect' in pair and pair['cause'] and pair['effect']:
                cause = self.normalize_text(pair['cause'])
                effect = self.normalize_text(pair['effect'])
                
                if cause and effect and (effect, cause) in gt_set:
                    switched_count += 1
        
        return switched_count
    
    def calculate_prf(self, predicted: Set, ground_truth: Set) -> Dict:
        """Calculate Precision, Recall, F1"""
        if len(predicted) == 0 and len(ground_truth) == 0:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
        
        if len(predicted) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        if len(ground_truth) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        correct = len(predicted.intersection(ground_truth))
        
        precision = correct / len(predicted) if len(predicted) > 0 else 0.0
        recall = correct / len(ground_truth) if len(ground_truth) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'correct': correct,
            'predicted': len(predicted),
            'ground_truth': len(ground_truth)
        }
    
    def evaluate_classification(self, predictions: List[Dict], ground_truth: List[Dict], 
                               field: str) -> Dict:
        """Evaluate classification accuracy for causality or sententiality"""
        pred_labels = []
        gt_labels = []
        
        for pred_pair in predictions:
            if field not in pred_pair or not pred_pair[field]:
                continue
            
            pred_cause = self.normalize_text(pred_pair.get('cause', ''))
            pred_effect = self.normalize_text(pred_pair.get('effect', ''))
            
            for gt_pair in ground_truth:
                gt_cause = self.normalize_text(gt_pair.get('cause', ''))
                gt_effect = self.normalize_text(gt_pair.get('effect', ''))
                
                if pred_cause == gt_cause and pred_effect == gt_effect:
                    pred_labels.append(self.normalize_text(pred_pair[field]))
                    gt_labels.append(self.normalize_text(gt_pair.get(field, '')))
                    break
        
        if len(pred_labels) == 0:
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'per_class': {}}
        
        accuracy = accuracy_score(gt_labels, pred_labels)
        unique_labels = list(set(gt_labels + pred_labels))
        
        precision, recall, f1, support = precision_recall_fscore_support(
            gt_labels, pred_labels, labels=unique_labels, average=None, zero_division=0
        )
        
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            gt_labels, pred_labels, average='macro', zero_division=0
        )
        
        per_class = {}
        for i, label in enumerate(unique_labels):
            per_class[label] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i])
            }
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision_macro),
            'recall': float(recall_macro),
            'f1': float(f1_macro),
            'per_class': per_class,
            'total_matched': len(pred_labels)
        }
    
    def evaluate(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
        """
        Main evaluation function
        
        Args:
            predictions: List of predicted extraction results
                Each item: {'s/n': ..., 'sentence': ..., 'pairs': [...]}
            ground_truth: List of ground truth extraction results
        
        Returns:
            Dictionary of all metrics
        """
        all_pred_causes = set()
        all_gt_causes = set()
        all_pred_effects = set()
        all_gt_effects = set()
        all_pred_pairs = set()
        all_gt_pairs = set()
        
        total_switched = 0
        total_samples = len(predictions)
        
        causality_preds = []
        causality_gts = []
        sententiality_preds = []
        sententiality_gts = []
        
        for pred, gt in zip(predictions, ground_truth):
            pred_pairs = pred.get('pairs', [])
            gt_pairs = gt.get('pairs', [])
            
            # Ensure pairs are lists
            if not isinstance(pred_pairs, list):
                pred_pairs = []
            if not isinstance(gt_pairs, list):
                gt_pairs = []
            
            # 1. Extract causes and effects
            pred_causes, pred_effects = self.extract_causes_effects(pred_pairs)
            gt_causes, gt_effects = self.extract_causes_effects(gt_pairs)
            
            all_pred_causes.update(pred_causes)
            all_gt_causes.update(gt_causes)
            all_pred_effects.update(pred_effects)
            all_gt_effects.update(gt_effects)
            
            # 2. Extract pairs
            pred_pair_set = self.extract_pairs(pred_pairs)
            gt_pair_set = self.extract_pairs(gt_pairs)
            
            all_pred_pairs.update(pred_pair_set)
            all_gt_pairs.update(gt_pair_set)
            
            # 3. Check switching
            switched = self.check_switching(pred_pairs, gt_pairs)
            total_switched += switched
            
            # 4. Collect classification labels
            for pred_pair in pred_pairs:
                pred_cause = self.normalize_text(pred_pair.get('cause', ''))
                pred_effect = self.normalize_text(pred_pair.get('effect', ''))
                
                if not pred_cause or not pred_effect:
                    continue
                
                for gt_pair in gt_pairs:
                    gt_cause = self.normalize_text(gt_pair.get('cause', ''))
                    gt_effect = self.normalize_text(gt_pair.get('effect', ''))
                    
                    if pred_cause == gt_cause and pred_effect == gt_effect:
                        # Causality
                        if 'causality' in pred_pair and 'causality' in gt_pair:
                            pred_caus = self.normalize_text(pred_pair['causality'])
                            gt_caus = self.normalize_text(gt_pair['causality'])
                            if pred_caus and gt_caus:
                                causality_preds.append(pred_caus)
                                causality_gts.append(gt_caus)
                        
                        # Sententiality
                        if 'sententiality' in pred_pair and 'sententiality' in gt_pair:
                            pred_sent = self.normalize_text(pred_pair['sententiality'])
                            gt_sent = self.normalize_text(gt_pair['sententiality'])
                            if pred_sent and gt_sent:
                                sententiality_preds.append(pred_sent)
                                sententiality_gts.append(gt_sent)
                        break
        
        results = {
            'cause_extraction': self.calculate_prf(all_pred_causes, all_gt_causes),
            'effect_extraction': self.calculate_prf(all_pred_effects, all_gt_effects),
            'pair_matching': self.calculate_prf(all_pred_pairs, all_gt_pairs),
            'switching': {
                'total_switched': total_switched,
                'switching_rate': total_switched / max(len(all_pred_pairs), 1)
            },
            'causality_classification': self._calculate_classification_metrics(
                causality_preds, causality_gts
            ) if causality_preds else {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
            'sententiality_classification': self._calculate_classification_metrics(
                sententiality_preds, sententiality_gts
            ) if sententiality_preds else {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
            'summary': {
                'total_samples': total_samples,
                'total_predicted_pairs': len(all_pred_pairs),
                'total_ground_truth_pairs': len(all_gt_pairs)
            }
        }
        
        self.results = results
        return results
    
    def _calculate_classification_metrics(self, predictions: List[str], 
                                         ground_truth: List[str]) -> Dict:
        """Calculate classification metrics"""
        if len(predictions) == 0:
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        accuracy = accuracy_score(ground_truth, predictions)
        unique_labels = list(set(ground_truth + predictions))
        
        precision, recall, f1, support = precision_recall_fscore_support(
            ground_truth, predictions, labels=unique_labels, average='macro', zero_division=0
        )
        
        precision_pc, recall_pc, f1_pc, support_pc = precision_recall_fscore_support(
            ground_truth, predictions, labels=unique_labels, average=None, zero_division=0
        )
        
        per_class = {}
        for i, label in enumerate(unique_labels):
            per_class[label] = {
                'precision': float(precision_pc[i]),
                'recall': float(recall_pc[i]),
                'f1': float(f1_pc[i]),
                'support': int(support_pc[i])
            }
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'per_class': per_class
        }
    
    def print_report(self):
        """Print detailed evaluation report"""
        if not self.results:
            print("No evaluation results available. Run evaluate() first.")
            return
        
        r = self.results
        
        print("\n" + "="*70)
        print("EXTRACTION EVALUATION REPORT")
        print("="*70)
        
        print("\n--- 1. Cause Extraction ---")
        ce = r['cause_extraction']
        print(f"Precision: {ce['precision']:.4f}")
        print(f"Recall:    {ce['recall']:.4f}")
        print(f"F1-Score:  {ce['f1']:.4f}")
        print(f"Correct:   {ce.get('correct', 0)}/{ce.get('predicted', 0)} predicted, {ce.get('ground_truth', 0)} actual")
        
        print("\n--- 2. Effect Extraction ---")
        ee = r['effect_extraction']
        print(f"Precision: {ee['precision']:.4f}")
        print(f"Recall:    {ee['recall']:.4f}")
        print(f"F1-Score:  {ee['f1']:.4f}")
        print(f"Correct:   {ee.get('correct', 0)}/{ee.get('predicted', 0)} predicted, {ee.get('ground_truth', 0)} actual")
        
        print("\n--- 3. Cause-Effect Pair Matching ---")
        pm = r['pair_matching']
        print(f"Precision: {pm['precision']:.4f}")
        print(f"Recall:    {pm['recall']:.4f}")
        print(f"F1-Score:  {pm['f1']:.4f}")
        print(f"Correct:   {pm.get('correct', 0)}/{pm.get('predicted', 0)} predicted, {pm.get('ground_truth', 0)} actual")
        
        print("\n--- 4. Cause-Effect Switching ---")
        sw = r['switching']
        print(f"Switched Pairs:   {sw['total_switched']}")
        print(f"Switching Rate:   {sw['switching_rate']:.4f}")
        
        print("\n--- 5. Causality Classification (Implicit vs Explicit) ---")
        cc = r['causality_classification']
        print(f"Accuracy:  {cc['accuracy']:.4f}")
        print(f"Precision: {cc['precision']:.4f}")
        print(f"Recall:    {cc['recall']:.4f}")
        print(f"F1-Score:  {cc['f1']:.4f}")
        if 'per_class' in cc and cc['per_class']:
            for label, metrics in cc['per_class'].items():
                print(f"  {label}: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
        
        print("\n--- 6. Sententiality Classification (Intra vs Inter) ---")
        sc = r['sententiality_classification']
        print(f"Accuracy:  {sc['accuracy']:.4f}")
        print(f"Precision: {sc['precision']:.4f}")
        print(f"Recall:    {sc['recall']:.4f}")
        print(f"F1-Score:  {sc['f1']:.4f}")
        if 'per_class' in sc and sc['per_class']:
            for label, metrics in sc['per_class'].items():
                print(f"  {label}: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
        
        print("\n--- Summary ---")
        s = r['summary']
        print(f"Total Samples:           {s['total_samples']}")
        print(f"Predicted Pairs:         {s['total_predicted_pairs']}")
        print(f"Ground Truth Pairs:      {s['total_ground_truth_pairs']}")
        
        print("="*70 + "\n")
    
    def save_results(self, filepath: str):
        """Save results to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {filepath}")
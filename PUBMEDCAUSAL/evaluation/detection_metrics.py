"""
Evaluation Metrics for Causal Detection Task
Metrics: Accuracy, Precision, Recall, F1-score
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from typing import List, Dict
import json


class DetectionEvaluator:
    """Evaluate causal detection performance"""
    
    def __init__(self):
        self.results = {}
    
    def evaluate(self, predictions: List[str], ground_truth: List[int]) -> Dict:
        """
        Evaluate detection predictions
        
        Args:
            predictions: List of "Yes"/"No" predictions
            ground_truth: List of binary labels (1=causal, 0=non-causal)
        
        Returns:
            Dictionary of metrics
        """
        # Convert predictions to binary
        pred_binary = []
        for pred in predictions:
            if isinstance(pred, str):
                pred = pred.strip().lower()
                if 'yes' in pred:
                    pred_binary.append(1)
                elif 'no' in pred:
                    pred_binary.append(0)
                else:
                    pred_binary.append(0)  # Default to no causality
            else:
                pred_binary.append(int(pred))
        
        # Calculate metrics
        accuracy = accuracy_score(ground_truth, pred_binary)
        
        # Precision, Recall, F1 (macro and per-class)
        precision, recall, f1, support = precision_recall_fscore_support(
            ground_truth,
            pred_binary,
            average=None,
            labels=[0, 1]
        )
        
        # Macro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            ground_truth,
            pred_binary,
            average='macro'
        )
        
        # Confusion matrix
        cm = confusion_matrix(ground_truth, pred_binary, labels=[0, 1])
        
        # Compile results
        results = {
            'accuracy': float(accuracy),
            
            # Per-class metrics
            'non_causal': {
                'precision': float(precision[0]),
                'recall': float(recall[0]),
                'f1': float(f1[0]),
                'support': int(support[0])
            },
            'causal': {
                'precision': float(precision[1]),
                'recall': float(recall[1]),
                'f1': float(f1[1]),
                'support': int(support[1])
            },
            
            # Macro averages
            'macro': {
                'precision': float(precision_macro),
                'recall': float(recall_macro),
                'f1': float(f1_macro)
            },
            
            # Confusion matrix
            'confusion_matrix': {
                'true_negative': int(cm[0, 0]),
                'false_positive': int(cm[0, 1]),
                'false_negative': int(cm[1, 0]),
                'true_positive': int(cm[1, 1])
            },
            
            # Additional info
            'total_samples': len(ground_truth),
            'total_predicted_causal': sum(pred_binary),
            'total_actual_causal': sum(ground_truth)
        }
        
        self.results = results
        return results
    
    def print_report(self):
        """Print detailed evaluation report"""
        if not self.results:
            print("No evaluation results available. Run evaluate() first.")
            return
        
        r = self.results
        
        print("\n" + "="*60)
        print("DETECTION EVALUATION REPORT")
        print("="*60)
        
        print(f"\nOverall Accuracy: {r['accuracy']:.4f}")
        
        print("\n--- Per-Class Metrics ---")
        print(f"Non-Causal (Class 0):")
        print(f"  Precision: {r['non_causal']['precision']:.4f}")
        print(f"  Recall:    {r['non_causal']['recall']:.4f}")
        print(f"  F1-Score:  {r['non_causal']['f1']:.4f}")
        print(f"  Support:   {r['non_causal']['support']}")
        
        print(f"\nCausal (Class 1):")
        print(f"  Precision: {r['causal']['precision']:.4f}")
        print(f"  Recall:    {r['causal']['recall']:.4f}")
        print(f"  F1-Score:  {r['causal']['f1']:.4f}")
        print(f"  Support:   {r['causal']['support']}")
        
        print("\n--- Macro Averages ---")
        print(f"Precision: {r['macro']['precision']:.4f}")
        print(f"Recall:    {r['macro']['recall']:.4f}")
        print(f"F1-Score:  {r['macro']['f1']:.4f}")
        
        print("\n--- Confusion Matrix ---")
        cm = r['confusion_matrix']
        print(f"                 Predicted")
        print(f"                 No    Yes")
        print(f"Actual  No      {cm['true_negative']:4d}  {cm['false_positive']:4d}")
        print(f"        Yes     {cm['false_negative']:4d}  {cm['true_positive']:4d}")
        
        print(f"\n--- Summary ---")
        print(f"Total Samples:           {r['total_samples']}")
        print(f"Predicted Causal:        {r['total_predicted_causal']}")
        print(f"Actual Causal:           {r['total_actual_causal']}")
        
        print("="*60 + "\n")
    
    def save_results(self, filepath: str):
        """Save results to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {filepath}")


def test_evaluator():
    """Test the evaluator"""
    # Sample data
    ground_truth = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
    predictions = ["Yes", "No", "Yes", "No", "No", "Yes", "No", "Yes", "Yes", "No"]
    
    evaluator = DetectionEvaluator()
    results = evaluator.evaluate(predictions, ground_truth)
    evaluator.print_report()


if __name__ == "__main__":
    test_evaluator()
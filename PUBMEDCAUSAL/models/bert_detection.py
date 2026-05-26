"""
BERT Detection - Simplified Version

Models: BERT, SciBERT, BioBERT, PubMedBERT
Uses the provided train file for training.
Uses the provided test file only for final evaluation.

Usage:
    python bert_detection.py --model_name bert --train_file train.json \
                             --test_file test.json --output_dir ./checkpoints/
"""

import os
import json
import torch
import argparse
import numpy as np

from dataclasses import dataclass
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


@dataclass
class Config:
    model_name: str = ''
    train_file: str = ''
    test_file: str = ''
    output_dir: str = ''
    num_epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    max_length: int = 128


MODELS = {
    'bert': 'bert-base-uncased',
    'scibert': 'allenai/scibert_scivocab_uncased',
    'pubmedbert': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
    'biobert': 'dmis-lab/biobert-v1.1',
}


class Detector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.mp = MODELS.get(cfg.model_name.lower(), cfg.model_name)

        print(f"Model: {cfg.model_name} | Device: {self.dev}")

    def load_model(self):
        self.tok = AutoTokenizer.from_pretrained(self.mp)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.mp,
            num_labels=2
        )

        print(f"Loaded ({sum(p.numel() for p in self.model.parameters()):,} params)")

    def tokenize(self, ex):
        return self.tok(
            ex['text'],
            padding='max_length',
            truncation=True,
            max_length=self.cfg.max_length
        )

    def metrics(self, ep: EvalPrediction):
        preds = np.argmax(ep.predictions, axis=1)
        labels = ep.label_ids

        p, r, f1, _ = precision_recall_fscore_support(
            labels,
            preds,
            average='binary',
            zero_division=0
        )

        acc = accuracy_score(labels, preds)

        return {
            'accuracy': acc,
            'precision': p,
            'recall': r,
            'f1': f1,
        }

    def train(self):
        with open(self.cfg.train_file, 'r', encoding='utf-8') as f:
            train_data = json.load(f)

        with open(self.cfg.test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)

        print(f"\nLoaded {len(train_data)} training samples")
        print(f"Loaded {len(test_data)} test samples")

        train_ds = Dataset.from_dict({
            'text': [x['sentence'] for x in train_data],
            'label': [x['label'] for x in train_data]
        })

        test_ds = Dataset.from_dict({
            'text': [x['sentence'] for x in test_data],
            'label': [x['label'] for x in test_data]
        })

        self.load_model()

        print("Tokenizing...")

        train_ds = train_ds.map(
            self.tokenize,
            batched=True,
            remove_columns=['text']
        )

        test_ds = test_ds.map(
            self.tokenize,
            batched=True,
            remove_columns=['text']
        )

        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(
                output_dir=self.cfg.output_dir,
                num_train_epochs=self.cfg.num_epochs,
                per_device_train_batch_size=self.cfg.batch_size,
                per_device_eval_batch_size=self.cfg.batch_size,
                learning_rate=self.cfg.learning_rate,
                warmup_steps=100,
                logging_steps=50,
                eval_strategy="no",
                save_strategy="epoch",
                save_total_limit=2,
                load_best_model_at_end=False,
                report_to=[]
            ),
            train_dataset=train_ds,
            compute_metrics=self.metrics
        )

        print("\nTraining...")
        trainer.train()

        print("\n" + "=" * 60)
        print("FINAL TEST EVALUATION")
        print("=" * 60)

        test_results = trainer.predict(test_ds)

        preds = np.argmax(test_results.predictions, axis=1)
        labels = test_ds['label']

        p, r, f1, _ = precision_recall_fscore_support(
            labels,
            preds,
            average='binary',
            zero_division=0
        )

        acc = accuracy_score(labels, preds)

        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {p:.4f}")
        print(f"Recall:    {r:.4f}")
        print(f"F1 Score:  {f1:.4f}")

        final_dir = os.path.join(self.cfg.output_dir, "final")
        trainer.save_model(final_dir)
        self.tok.save_pretrained(final_dir)

        results = {
            'model': self.cfg.model_name,
            'train_samples': len(train_data),
            'test_samples': len(test_data),
            'test_metrics': {
                'accuracy': float(acc),
                'precision': float(p),
                'recall': float(r),
                'f1': float(f1)
            }
        }

        with open(os.path.join(self.cfg.output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nModel saved to: {final_dir}")
        print(f"Results saved to: {os.path.join(self.cfg.output_dir, 'results.json')}")


def main():
    p = argparse.ArgumentParser(description='BERT Detection - Train/Test Only')

    p.add_argument(
        '--model_name',
        required=True,
        choices=list(MODELS.keys()),
        help='Model to train'
    )

    p.add_argument(
        '--train_file',
        required=True,
        help='Training data JSON file'
    )

    p.add_argument(
        '--test_file',
        required=True,
        help='Test data JSON file'
    )

    p.add_argument(
        '--output_dir',
        required=True,
        help='Output directory for checkpoints'
    )

    p.add_argument(
        '--num_epochs',
        type=int,
        default=3,
        help='Number of epochs'
    )

    p.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size'
    )

    p.add_argument(
        '--learning_rate',
        type=float,
        default=2e-5,
        help='Learning rate'
    )

    p.add_argument(
        '--max_length',
        type=int,
        default=128,
        help='Max sequence length'
    )

    args = p.parse_args()

    cfg = Config(
        model_name=args.model_name,
        train_file=args.train_file,
        test_file=args.test_file,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length
    )

    os.makedirs(cfg.output_dir, exist_ok=True)

    Detector(cfg).train()


if __name__ == "__main__":
    main()
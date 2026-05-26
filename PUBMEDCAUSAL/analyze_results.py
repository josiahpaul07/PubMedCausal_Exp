"""
Results Analysis and Comparison Utility
Aggregates and compares results across models, strategies, and splits
"""

import json
import argparse
from pathlib import Path
import pandas as pd


class ResultsAnalyzer:
    """Analyze and compare benchmark results"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
    
    def load_detection_results(self) -> pd.DataFrame:
        """Load all detection results into DataFrame"""
        results = []
        
        for json_file in self.results_dir.glob('detection/**/*.json'):
            if 'summary' not in json_file.name:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        
                        result = {
                            'model': data['model'],
                            'strategy': data['prompt_strategy'],
                            'timestamp': data['timestamp'],
                            'accuracy': data['metrics']['accuracy'],
                            'precision': data['metrics']['macro']['precision'],
                            'recall': data['metrics']['macro']['recall'],
                            'f1': data['metrics']['macro']['f1'],
                            'causal_f1': data['metrics']['causal']['f1'],
                            'non_causal_f1': data['metrics']['non_causal']['f1'],
                            'file': str(json_file)
                        }
                        results.append(result)
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")
        
        return pd.DataFrame(results)
    
    def load_extraction_results(self) -> pd.DataFrame:
        """Load all extraction results into DataFrame"""
        results = []
        
        for json_file in self.results_dir.glob('extraction/**/*.json'):
            if 'summary' not in json_file.name:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        
                        result = {
                            'model': data['model'],
                            'strategy': data['prompt_strategy'],
                            'split': data['split_type'],
                            'timestamp': data['timestamp'],
                            'cause_f1': data['metrics']['cause_extraction']['f1'],
                            'effect_f1': data['metrics']['effect_extraction']['f1'],
                            'pair_f1': data['metrics']['pair_matching']['f1'],
                            'switching_rate': data['metrics']['switching']['switching_rate'],
                            'causality_f1': data['metrics']['causality_classification']['f1'],
                            'sententiality_f1': data['metrics']['sententiality_classification']['f1'],
                            'file': str(json_file)
                        }
                        results.append(result)
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")
        
        return pd.DataFrame(results)
    
    def compare_models_detection(self, strategy: str = 'zero-shot') -> pd.DataFrame:
        """Compare models on detection task for a specific strategy"""
        df = self.load_detection_results()
        
        if df.empty:
            print("No detection results found!")
            return pd.DataFrame()
        
        # Filter by strategy
        df_filtered = df[df['strategy'] == strategy].copy()
        
        # Sort by F1
        df_filtered = df_filtered.sort_values('f1', ascending=False)
        
        # Select key columns
        columns = ['model', 'accuracy', 'precision', 'recall', 'f1', 'causal_f1']
        return df_filtered[columns]
    
    def compare_strategies_detection(self, model: str) -> pd.DataFrame:
        """Compare strategies for a specific model on detection"""
        df = self.load_detection_results()
        
        if df.empty:
            print("No detection results found!")
            return pd.DataFrame()
        
        # Filter by model
        df_filtered = df[df['model'] == model].copy()
        
        # Sort by F1
        df_filtered = df_filtered.sort_values('f1', ascending=False)
        
        columns = ['strategy', 'accuracy', 'precision', 'recall', 'f1']
        return df_filtered[columns]
    
    def compare_models_extraction(self, split: str = 'combined', 
                                  strategy: str = 'zero-shot') -> pd.DataFrame:
        """Compare models on extraction task"""
        df = self.load_extraction_results()
        
        if df.empty:
            print("No extraction results found!")
            return pd.DataFrame()
        
        # Filter
        df_filtered = df[(df['split'] == split) & (df['strategy'] == strategy)].copy()
        
        # Sort by pair F1
        df_filtered = df_filtered.sort_values('pair_f1', ascending=False)
        
        columns = ['model', 'pair_f1', 'cause_f1', 'effect_f1', 
                  'switching_rate', 'causality_f1', 'sententiality_f1']
        return df_filtered[columns]
    
    def compare_splits_extraction(self, model: str, strategy: str = 'zero-shot') -> pd.DataFrame:
        """Compare performance across splits"""
        df = self.load_extraction_results()
        
        if df.empty:
            print("No extraction results found!")
            return pd.DataFrame()
        
        # Filter
        df_filtered = df[(df['model'] == model) & (df['strategy'] == strategy)].copy()
        
        columns = ['split', 'pair_f1', 'cause_f1', 'effect_f1', 
                  'switching_rate', 'causality_f1']
        return df_filtered[columns]
    
    def best_performance_summary(self):
        """Generate summary of best performance across all experiments"""
        print("\n" + "="*80)
        print("BEST PERFORMANCE SUMMARY")
        print("="*80)
        
        # Detection
        df_det = self.load_detection_results()
        if not df_det.empty:
            best_det = df_det.loc[df_det['f1'].idxmax()]
            print("\nBest Detection Performance:")
            print(f"  Model: {best_det['model']}")
            print(f"  Strategy: {best_det['strategy']}")
            print(f"  F1-Score: {best_det['f1']:.4f}")
            print(f"  Accuracy: {best_det['accuracy']:.4f}")
        
        # Extraction
        df_ext = self.load_extraction_results()
        if not df_ext.empty:
            for split in ['X_only', 'Y_only', 'combined']:
                df_split = df_ext[df_ext['split'] == split]
                if not df_split.empty:
                    best_ext = df_split.loc[df_split['pair_f1'].idxmax()]
                    print(f"\nBest Extraction Performance ({split}):")
                    print(f"  Model: {best_ext['model']}")
                    print(f"  Strategy: {best_ext['strategy']}")
                    print(f"  Pair F1: {best_ext['pair_f1']:.4f}")
                    print(f"  Causality F1: {best_ext['causality_f1']:.4f}")
        
        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze benchmark results')
    
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='Results directory')
    
    subparsers = parser.add_subparsers(dest='command', help='Analysis command')
    
    # Compare models
    compare_models = subparsers.add_parser('compare-models', help='Compare models')
    compare_models.add_argument('--task', choices=['detection', 'extraction'], required=True)
    compare_models.add_argument('--strategy', default='zero-shot')
    compare_models.add_argument('--split', default='combined')
    
    # Compare strategies
    compare_strategies = subparsers.add_parser('compare-strategies', help='Compare strategies')
    compare_strategies.add_argument('--task', choices=['detection', 'extraction'], required=True)
    compare_strategies.add_argument('--model', required=True)
    
    # Best performance
    subparsers.add_parser('best', help='Show best performance summary')
    
    args = parser.parse_args()
    
    analyzer = ResultsAnalyzer(args.results_dir)
    
    if args.command == 'compare-models':
        if args.task == 'detection':
            df = analyzer.compare_models_detection(args.strategy)
        else:
            df = analyzer.compare_models_extraction(args.split, args.strategy)
        
        if not df.empty:
            print("\n" + df.to_string(index=False))
    
    elif args.command == 'compare-strategies':
        if args.task == 'detection':
            df = analyzer.compare_strategies_detection(args.model)
        else:
            df = analyzer.compare_splits_extraction(args.model)
        
        if not df.empty:
            print("\n" + df.to_string(index=False))
    
    elif args.command == 'best':
        analyzer.best_performance_summary()
    
    else:
        print("No command specified. Use --help for usage.")
        print("\nAvailable commands:")
        print("  compare-models    - Compare different models")
        print("  compare-strategies - Compare prompting strategies")
        print("  best              - Show best performance summary")


if __name__ == "__main__":
    main()
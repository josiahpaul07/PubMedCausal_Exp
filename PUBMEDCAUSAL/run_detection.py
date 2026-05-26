"""
Main Script for Causal Detection Benchmarking
Runs LLM models with different prompting strategies
OPTIMIZED: Passes task='detection' to batch_generate for faster inference (32 tokens vs 512)
FIXED: Model loaded once per model not per strategy
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm


def run_detection_benchmark(args):
    """Run detection benchmarking"""
    
    # Import modules
    sys.path.append(str(Path(__file__).parent))
    
    from prompts.detection_prompts import DetectionPrompts
    from models.llm_inference import LLMInference, ModelConfig, get_model_configs
    from evaluation.detection_metrics import DetectionEvaluator
    
    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test data
    with open(args.test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"Loaded {len(test_data)} test samples")
    
    # Get model configs
    all_configs = get_model_configs()
    
    # Determine which models to run
    if args.models:
        model_names = args.models
    elif args.mode == 'prompt-only':
        model_names = ['deepseek-r1-distill-qwen-32b', 'meta-llama-3.3-70b', 
                      'deepseek-70b', 'mixtral-8x7b']
    else:
        model_names = list(all_configs.keys())
    
    # Determine strategies
    strategies = args.strategies if args.strategies else [
        'zero-shot', 'cot', 'few-shot', 'cot-fewshot', 'react', 'least-to-most'
    ]
    
    # Summary results
    summary = []
    
    # Run each model
    for model_name in model_names:
        if model_name in all_configs:
            config = all_configs[model_name]
        else:
            config = ModelConfig(
                name=model_name,
                model_path=args.model_path if args.model_path else model_name,
                model_type='huggingface',
                batch_size=args.batch_size if args.batch_size else 8
            )
        
        # Override batch size if provided via CLI
        if args.batch_size:
            config.batch_size = args.batch_size

        # OPTIMIZATION: Load model ONCE per model (not per strategy - saves huge time!)
        print(f"\nLoading model: {config.name}")
        inference = LLMInference(config)

        for strategy in strategies:
            try:
                print(f"\n{'='*70}")
                print(f"Model: {config.name}")
                print(f"Strategy: {strategy}")
                print(f"{'='*70}")
                
                # Generate prompts
                print("Generating prompts...")
                prompts = [DetectionPrompts.get_prompt(strategy, item['sentence']) 
                          for item in test_data]
                
                # Run inference - OPTIMIZED: task='detection' uses only 32 max_new_tokens
                print("Running inference...")
                responses = inference.batch_generate(prompts, task='detection')
                
                # Parse responses
                print("Parsing responses...")
                predictions = [DetectionPrompts.parse_response(r) for r in responses]
                
                # Get ground truth
                ground_truth = [item['label'] for item in test_data]
                
                # Evaluate
                print("Evaluating...")
                evaluator = DetectionEvaluator()
                results = evaluator.evaluate(predictions, ground_truth)
                evaluator.print_report()
                
                # Save results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                result_file = results_dir / f"detection_{config.name}_{strategy}_{timestamp}.json"
                
                output = {
                    'model': config.name,
                    'prompt_strategy': strategy,
                    'timestamp': timestamp,
                    'metrics': results,
                    'predictions': predictions,
                    'config': {
                        'test_file': str(args.test_file),
                        'num_samples': len(test_data)
                    }
                }
                
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(output, f, indent=2)
                
                print(f"? Results saved to {result_file}")
                
                summary.append({
                    'model': config.name,
                    'strategy': strategy,
                    'accuracy': results['accuracy'],
                    'f1': results['macro']['f1'],
                    'precision': results['macro']['precision'],
                    'recall': results['macro']['recall']
                })
                
            except Exception as e:
                print(f"? Error with {config.name} + {strategy}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
    
    # Save summary
    summary_file = results_dir / f"detection_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n? All benchmarks complete! Summary saved to {summary_file}")
    
    # Print summary table
    print("\n" + "="*90)
    print("DETECTION BENCHMARK SUMMARY")
    print("="*90)
    print(f"{'Model':<35} {'Strategy':<15} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8}")
    print("-"*90)
    
    for item in summary:
        print(f"{item['model']:<35} {item['strategy']:<15} "
              f"{item['accuracy']:<8.4f} {item['precision']:<8.4f} "
              f"{item['recall']:<8.4f} {item['f1']:<8.4f}")
    
    print("="*90 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Run causal detection benchmarking')
    
    # Data
    parser.add_argument('--test_file', type=str, required=True,
                       help='Path to test data (detection_test.json)')
    
    # Models
    parser.add_argument('--mode', type=str, choices=['prompt-only', 'fine-tuned', 'all'],
                       default='prompt-only',
                       help='Which models to run')
    parser.add_argument('--models', nargs='+',
                       help='Specific models to run (overrides mode)')
    parser.add_argument('--model_path', type=str,
                       help='Path to fine-tuned model checkpoint')
    parser.add_argument('--models_base_dir', type=str,
                       help='Base directory for local models (e.g., ./30k_biocausal_exp)')
    
    # Strategies
    parser.add_argument('--strategies', nargs='+',
                       choices=['zero-shot', 'cot', 'few-shot', 'cot-fewshot', 'react', 'least-to-most'],
                       help='Prompting strategies to use (default: all)')
    
    # Other
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size for inference (overrides model default)')
    parser.add_argument('--results_dir', type=str, default='./results/detection',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    run_detection_benchmark(args)


if __name__ == "__main__":
    main()
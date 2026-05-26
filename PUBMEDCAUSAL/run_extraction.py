"""
Main Script for Causal Extraction Benchmarking
Runs LLM models with different prompting strategies on 3 experimental splits
Supports 6 prompting strategies: zero-shot, few-shot, cot, cot-fewshot, least-to-most, react
OPTIMIZED: Passes task='extraction' to batch_generate for faster inference (256 tokens vs 512)
FIXED: Model loaded once per model not per strategy
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add project directories to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'prompts'))
sys.path.insert(0, str(project_root / 'models'))
sys.path.insert(0, str(project_root / 'evaluation'))


# Local folder name mapping (used when models_base_dir is provided)
LOCAL_FOLDER_MAP = {
    'llama-3b': 'llama-3.2-3b',
    'llama-8b': 'llama-3.1-8b',
    'mistral-7b': 'mistral-7b-instruct-v0.2',
    'qwen-7b': 'qwen2.5-7b-instruct',
    'deepseek-7b': 'deepseek-llm-7b-chat',
    'deepseek-r1-distill-qwen-32b': 'deepseek-r1-distill-qwen-32b',
    'meta-llama-3.3-70b': 'meta-llama-3.3-70b-instruct',
    'deepseek-70b': 'DeepSeek-R1-Distill-Llama-70B',
    'mixtral-8x7b': 'mixtral-8x7b-instruct-v0.1',
}

# HuggingFace repo IDs (correct IDs used as fallback)
HF_REPO_MAP = {
    'llama-3b': 'meta-llama/Llama-3.2-3B-Instruct',
    'llama-8b': 'meta-llama/Llama-3.1-8B-Instruct',
    'mistral-7b': 'mistralai/Mistral-7B-Instruct-v0.2',
    'qwen-7b': 'Qwen/Qwen2.5-7B-Instruct',
    'deepseek-7b': 'deepseek-ai/deepseek-llm-7b-chat',
    'deepseek-r1-distill-qwen-32b': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
    'meta-llama-3.3-70b': 'meta-llama/Llama-3.3-70B-Instruct',
    'deepseek-70b': 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
    'mixtral-8x7b': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
}


def get_model_path(model_name, models_base_dir=None):
    """Get model path - checks local first, falls back to correct HuggingFace ID"""
    if '-finetuned' in model_name:
        return model_name
    
    # Check local folder first
    if models_base_dir:
        local_folder = LOCAL_FOLDER_MAP.get(model_name, model_name)
        local_path = os.path.join(models_base_dir, local_folder)
        
        if os.path.exists(local_path):
            print(f"? Using local model: {local_path}")
            return local_path
        else:
            print(f"? Local model not found at {local_path}, using HuggingFace")
    
    # Fall back to correct HuggingFace repo ID
    hf_path = HF_REPO_MAP.get(model_name, model_name)
    print(f"? Using HuggingFace model: {hf_path}")
    return hf_path


def run_extraction_benchmark(args):
    """Run extraction benchmarking"""
    
    # Import modules
    from extraction_prompts import ExtractionPrompts
    from llm_inference import LLMInference, ModelConfig, get_model_configs
    from extraction_metrics import ExtractionEvaluator
    
    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test data
    with open(args.test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"Loaded {len(test_data)} test samples")
    print(f"Split type: {args.split_type}")
    
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
        'zero-shot', 'few-shot', 'cot', 'cot-fewshot', 'least-to-most', 'react'
    ]
    
    # Summary results
    summary = []
    
    # Run each model
    for model_name in model_names:
        # Get model path (local or HuggingFace)
        if args.model_path:
            model_path = args.model_path
        else:
            model_path = get_model_path(model_name, args.models_base_dir)
        
        # Create config
        if model_name in all_configs:
            config = all_configs[model_name]
            config.model_path = model_path
        else:
            config = ModelConfig(
                name=model_name,
                model_path=model_path,
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
                print(f"Split: {args.split_type}")
                print(f"{'='*70}")
                
                # Generate prompts
                print("Generating prompts...")
                prompts = [ExtractionPrompts.get_prompt(strategy, item['sentence'])
                          for item in test_data]
                
                # Run inference - OPTIMIZED: task='extraction' uses 256 max_new_tokens
                print("Running inference...")
                responses = inference.batch_generate(prompts, task='extraction')
                
                # Parse responses
                print("Parsing responses...")
                predictions = []
                for i, response in enumerate(responses):
                    parsed = ExtractionPrompts.parse_response(response)
                    predictions.append({
                        's/n': test_data[i].get('s/n', i),
                        'sentence': test_data[i]['sentence'],
                        'pairs': parsed.get('pairs', [])
                    })
                
                # Evaluate
                print("Evaluating...")
                evaluator = ExtractionEvaluator()
                results = evaluator.evaluate(predictions, test_data)
                evaluator.print_report()
                
                # Save results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                result_file = results_dir / f"extraction_{args.split_type}_{config.name}_{strategy}_{timestamp}.json"
                
                output = {
                    'model': config.name,
                    'model_path': model_path,
                    'prompt_strategy': strategy,
                    'split_type': args.split_type,
                    'timestamp': timestamp,
                    'metrics': results,
                    'predictions': predictions,
                    'config': {
                        'test_file': str(args.test_file),
                        'num_samples': len(test_data),
                        'batch_size': config.batch_size
                    }
                }
                
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(output, f, indent=2)
                
                print(f"? Results saved to {result_file}")
                
                summary.append({
                    'model': config.name,
                    'strategy': strategy,
                    'split': args.split_type,
                    'cause_f1': results['cause_extraction']['f1'],
                    'effect_f1': results['effect_extraction']['f1'],
                    'pair_f1': results['pair_matching']['f1'],
                    'switching_rate': results['switching']['switching_rate'],
                    'causality_f1': results['causality_classification']['f1'],
                    'sententiality_f1': results['sententiality_classification']['f1']
                })
                
            except Exception as e:
                print(f"? Error with {config.name} + {strategy}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
    
    # Save summary
    summary_file = results_dir / f"extraction_{args.split_type}_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n? All benchmarks complete! Summary saved to {summary_file}")
    
    # Print summary table
    print("\n" + "="*120)
    print(f"EXTRACTION BENCHMARK SUMMARY - {args.split_type.upper()}")
    print("="*120)
    print(f"{'Model':<30} {'Strategy':<15} {'Cause-F1':<10} {'Effect-F1':<10} {'Pair-F1':<10} {'Switch%':<9} {'Caus-F1':<9} {'Sent-F1':<9}")
    print("-"*120)
    
    for item in summary:
        print(f"{item['model']:<30} {item['strategy']:<15} "
              f"{item['cause_f1']:<10.4f} {item['effect_f1']:<10.4f} "
              f"{item['pair_f1']:<10.4f} {item['switching_rate']:<9.4f} "
              f"{item['causality_f1']:<9.4f} {item['sententiality_f1']:<9.4f}")
    
    print("="*120 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run causal extraction benchmarking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prompt-only
  python run_extraction.py \\
      --test_file ./data/prepared/extraction_combined/test.json \\
      --split_type combined \\
      --models llama-3b llama-8b mistral-7b \\
      --strategies zero-shot few-shot cot \\
      --batch_size 16

  # Fine-tuned model
  python run_extraction.py \\
      --test_file ./data/prepared/extraction_combined/test.json \\
      --split_type combined \\
      --models llama-3b-finetuned \\
      --model_path ./checkpoints/extraction_combined_llama3b/final \\
      --strategies zero-shot few-shot cot \\
      --batch_size 16
        """
    )
    
    # Data
    parser.add_argument('--test_file', type=str, required=True,
                       help='Path to test data (extraction split)')
    parser.add_argument('--split_type', type=str, required=True,
                       choices=['X_only', 'Y_only', 'combined'],
                       help='Which experimental split to evaluate')
    
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
                       choices=['zero-shot', 'few-shot', 'cot', 'cot-fewshot', 'least-to-most', 'react'],
                       help='Prompting strategies to use (default: all 6)')
    
    # Other
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size for inference (overrides model default)')
    parser.add_argument('--results_dir', type=str, default='./results/extraction',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    run_extraction_benchmark(args)


if __name__ == "__main__":
    main()
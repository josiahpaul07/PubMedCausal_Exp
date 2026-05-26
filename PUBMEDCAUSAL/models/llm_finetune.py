"""
LoRA Fine-tuning for Causal Detection and Extraction
FIXED: Updated for new quantization API + local model support
"""

import os
import json
import torch
from dataclasses import dataclass
from typing import List, Dict
import sys
from pathlib import Path

# Add project directories to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'prompts'))
sys.path.insert(0, str(project_root / 'models'))
sys.path.insert(0, str(project_root / 'evaluation'))

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig  # Import for new quantization API
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import argparse


@dataclass
class FineTuneConfig:
    """Configuration for fine-tuning"""
    model_name: str
    task_type: str  # 'detection' or 'extraction'
    train_file: str
    test_file: str
    output_dir: str
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    lora_r: int = 8
    lora_alpha: int = 16


def get_model_path(model_name: str, base_dir: str = ".") -> str:
    """
    Get local model path or HuggingFace ID
    
    Args:
        model_name: Short name like 'llama-3b'
        base_dir: Base directory where models are stored
    
    Returns:
        Local path or HuggingFace ID
    """
    model_map = {
        'llama-3b': 'llama-3.2-3b',
        'llama-8b': 'llama-3.1-8b-instruct',
        'mistral-7b': 'mistral-7b',
        'qwen-7b': 'qwen2.5-7b-instruct-fp16',
        'deepseek-7b': 'deepseek-llm-7b-chat-fp16',
    }
    
    # Try local path first
    local_name = model_map.get(model_name.lower(), model_name)
    local_path = os.path.join(base_dir, local_name)
    
    if os.path.exists(local_path):
        print(f"? Using local model: {local_path}")
        return local_path
    
    # Fallback to HuggingFace
    print(f"? Local model not found: {local_path}")
    print(f"  Using HuggingFace ID")
    
    hf_map = {
        'llama-3b': 'meta-llama/Llama-3.2-3B-Instruct',
        'llama-8b': 'meta-llama/Llama-3.1-8B-Instruct',
        'mistral-7b': 'mistralai/Mistral-7B-Instruct-v0.2',
        'qwen-7b': 'Qwen/Qwen2.5-7B-Instruct',
        'deepseek-7b': 'deepseek-ai/deepseek-llm-7b-chat',
    }
    return hf_map.get(model_name.lower(), model_name)


class DataPreprocessor:
    """Handles data loading and formatting for fine-tuning"""
    
    def __init__(self, tokenizer, task_type: str):
        self.tokenizer = tokenizer
        self.task_type = task_type
    
    def format_detection_sample(self, sample: Dict) -> str:
        """Format a detection sample for training"""
        try:
            from detection_prompts import DetectionPrompts
            
            sentence = sample['sentence']
            label = sample['label']
            
            # Use zero-shot prompt format
            prompt = DetectionPrompts.get_prompt('zero-shot', sentence)
            response = str(label)
            
            return f"{prompt}\n{response}"
        except Exception as e:
            print(f"Error formatting detection sample: {e}")
            sentence = sample.get('sentence', '')
            label = sample.get('label', 0)
            return f"Detect causality in: {sentence}\nAnswer: {label}"
    
    def format_extraction_sample(self, sample: Dict) -> str:
        """Format an extraction sample for training"""
        try:
            from extraction_prompts import ExtractionPrompts
            
            # Handle different sample formats
            if isinstance(sample, dict):
                sentence = sample.get('sentence', '')
                pairs = sample.get('pairs', [])
            else:
                # If sample is a string (from tokenize_function)
                return sample
            
            # Create prompt
            prompt = ExtractionPrompts.get_prompt('zero-shot', sentence)
            
            # Create response
            response_pairs = []
            for pair in pairs:
                response_pairs.append({
                    'cause': pair.get('cause', ''),
                    'effect': pair.get('effect', ''),
                    'causality': pair.get('causality', 'Explicit'),
                    'sententiality': pair.get('sententiality', 'Intra')
                })
            
            response = json.dumps({'pairs': response_pairs})
            
            return f"{prompt}\n{response}"
            
        except Exception as e:
            print(f"Error formatting extraction sample: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback format
            if isinstance(sample, dict):
                sentence = sample.get('sentence', '')
                return f"Extract causal pairs from: {sentence}"
            else:
                return str(sample)
    
    def prepare_datasets(self, train_file: str, test_file: str):
        """Load and prepare datasets"""
        # Load data
        with open(train_file, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        print(f"Loaded {len(train_data)} training samples")
        print(f"Loaded {len(test_data)} test samples")
        
        # Format all samples first
        if self.task_type == 'detection':
            train_texts = [self.format_detection_sample(sample) for sample in train_data]
            test_texts = [self.format_detection_sample(sample) for sample in test_data]
        else:  # extraction
            train_texts = [self.format_extraction_sample(sample) for sample in train_data]
            test_texts = [self.format_extraction_sample(sample) for sample in test_data]
        
        # Tokenize all texts
        def tokenize_function(examples):
            """Tokenize batch of text strings"""
            # examples is a dict with key 'text' containing list of strings
            return self.tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=256  # Reduced from 512 to save memory
            )
        
        # Create datasets from formatted texts
        train_dataset = Dataset.from_dict({'text': train_texts})
        test_dataset = Dataset.from_dict({'text': test_texts})
        
        # Apply tokenization in batches
        train_tokenized = train_dataset.map(
            tokenize_function,
            batched=True,
            batch_size=100,
            remove_columns=['text']
        )
        
        test_tokenized = test_dataset.map(
            tokenize_function,
            batched=True,
            batch_size=100,
            remove_columns=['text']
        )
        
        return train_tokenized, test_tokenized


class LoRAFineTuner:
    """Main fine-tuning class using LoRA"""
    
    def __init__(self, config: FineTuneConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_model_and_tokenizer(self):
        """Load model and tokenizer with LoRA"""
        # Get model path (local or HuggingFace)
        model_path = get_model_path(self.config.model_name)
        
        print(f"Loading {self.config.model_name}...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=False
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # NEW QUANTIZATION API
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
            bnb_8bit_use_double_quant=True
        )
        
        # Load model with new quantization config
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,  # ? NEW API
            device_map="auto",
            trust_remote_code=True,
            local_files_only=False
        )
        
        # Prepare for k-bit training
        # Disable gradient checkpointing - it's causing CUDA errors
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        
        # LoRA configuration
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"]
        )
        
        # Add LoRA adapters
        model = get_peft_model(model, lora_config)
        
        # Print trainable parameters
        model.print_trainable_parameters()
        
        return model, tokenizer
    
    def train(self):
        """Main training loop"""
        # Load model and tokenizer
        model, tokenizer = self.load_model_and_tokenizer()
        
        # Prepare datasets
        preprocessor = DataPreprocessor(tokenizer, self.config.task_type)
        train_dataset, test_dataset = preprocessor.prepare_datasets(
            self.config.train_file,
            self.config.test_file
        )
        
        # Training arguments - Simplified for stability
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_steps=50,
            logging_steps=10,
            save_steps=200,
            eval_strategy="steps",
            eval_steps=200,
            save_total_limit=2,
            fp16=True,
            gradient_accumulation_steps=8,  # Increased to maintain effective batch size
            optim="adamw_torch",  # Use standard optimizer (more stable than paged_adamw_8bit)
            report_to=[],
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            # Gradient checkpointing DISABLED - was causing CUDA errors
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=data_collator
        )
        
        # Train
        print("\nStarting training...")
        trainer.train()
        
        # Save final model
        final_dir = os.path.join(self.config.output_dir, "final")
        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)
        
        print(f"\n? Training complete! Model saved to {final_dir}")


def main():
    parser = argparse.ArgumentParser(description='Fine-tune LLM with LoRA')
    
    parser.add_argument('--model_name', type=str, required=True,
                       choices=['mistral-7b', 'llama-3b', 'llama-8b', 'qwen-7b', 'deepseek-7b'],
                       help='Model to fine-tune')
    parser.add_argument('--task_type', type=str, required=True,
                       choices=['detection', 'extraction'],
                       help='Task type')
    parser.add_argument('--train_file', type=str, required=True,
                       help='Path to training data')
    parser.add_argument('--test_file', type=str, required=True,
                       help='Path to test data')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for checkpoints')
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--lora_r', type=int, default=8,
                       help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16,
                       help='LoRA alpha')
    
    args = parser.parse_args()
    
    # Create config
    config = FineTuneConfig(
        model_name=args.model_name,
        task_type=args.task_type,
        train_file=args.train_file,
        test_file=args.test_file,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha
    )
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Fine-tune
    tuner = LoRAFineTuner(config)
    tuner.train()


if __name__ == "__main__":
    main()
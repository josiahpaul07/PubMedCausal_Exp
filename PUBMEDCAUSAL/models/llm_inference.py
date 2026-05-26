"""
LLM Inference Engine with Batch Processing
Supports: HuggingFace models, OpenAI-compatible APIs
FIXED: Proper left-padding for decoder-only models and removed invalid flags
OPTIMIZED: Faster inference with cache clearing, reduced max_new_tokens, better batching
"""

import os
import json
import asyncio
import aiohttp
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class ModelConfig:
    """Configuration for LLM models"""
    name: str
    model_path: str  # HuggingFace model ID or API endpoint
    model_type: str  # 'huggingface' or 'api'
    api_key: Optional[str] = None
    max_tokens: int = 512
    temperature: float = 0.0
    batch_size: int = 8
    # OPTIMIZATION: Task-specific max_new_tokens
    # Detection only needs Yes/No (32 tokens), Extraction needs more (256)
    max_new_tokens_detection: int = 32
    max_new_tokens_extraction: int = 256
    load_in_4bit: bool = False  # Only enabled for large models (32B, 70B, Mixtral 8x7B)


class HuggingFaceInference:
    """Inference using HuggingFace transformers"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading {config.name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)

        # 4-bit quantization for large models only
        if config.load_in_4bit:
            print(f"  -> Using 4-bit quantization (NF4)")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_path,
                quantization_config=bnb_config,
                device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
        
        # FIX 1: Set padding side to left for decoder-only models
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'  # Critical for batch generation
        
        print(f"? Model loaded. Padding side: {self.tokenizer.padding_side}")
    
    def generate(self, prompts: List[str], max_new_tokens: int = None) -> List[str]:
        """Generate responses for a batch of prompts"""

        # OPTIMIZATION 1: Use task-specific max_new_tokens if not provided
        if max_new_tokens is None:
            max_new_tokens = self.config.max_tokens

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # FIX 2: Proper generation kwargs based on temperature
        generation_kwargs = {
            'max_new_tokens': max_new_tokens,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'use_cache': True,           # OPTIMIZATION 2: Explicit KV cache (speeds up generation)
            'repetition_penalty': 1.1,   # OPTIMIZATION 3: Prevents repetitive loops (saves time)
        }
        
        # Only add sampling parameters if temperature > 0
        if self.config.temperature > 0:
            generation_kwargs['temperature'] = self.config.temperature
            generation_kwargs['do_sample'] = True
            generation_kwargs['top_p'] = 0.9
        else:
            generation_kwargs['do_sample'] = False
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generation_kwargs
            )
        
        # Decode only the new tokens
        responses = []
        for i, output in enumerate(outputs):
            input_length = inputs['input_ids'][i].shape[0]
            new_tokens = output[input_length:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            responses.append(response)

        # OPTIMIZATION 4: Clear GPU cache after each batch
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        return responses


class APIInference:
    """Inference using OpenAI-compatible APIs"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.session = None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def _call_api(self, prompt: str, session: aiohttp.ClientSession) -> str:
        """Single API call with retry logic"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}"
        }
        
        payload = {
            "model": self.config.model_path,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }
        
        async with session.post(
            f"{self.config.api_endpoint}/chat/completions",
            headers=headers,
            json=payload
        ) as response:
            if response.status == 200:
                result = await response.json()
                return result['choices'][0]['message']['content']
            else:
                error = await response.text()
                raise Exception(f"API Error: {error}")
    
    async def generate_async(self, prompts: List[str]) -> List[str]:
        """Generate responses asynchronously"""
        async with aiohttp.ClientSession() as session:
            tasks = [self._call_api(prompt, session) for prompt in prompts]
            responses = await tqdm_asyncio.gather(*tasks, desc="API Calls")
        return responses
    
    def generate(self, prompts: List[str], max_new_tokens: int = None) -> List[str]:
        """Synchronous wrapper"""
        return asyncio.run(self.generate_async(prompts))


class LLMInference:
    """Main inference class with batch processing"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        
        if config.model_type == 'huggingface':
            self.engine = HuggingFaceInference(config)
        elif config.model_type == 'api':
            self.engine = APIInference(config)
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")
    
    def batch_generate(self, prompts: List[str], task: str = 'extraction') -> List[str]:
        """
        Generate with batching

        Args:
            prompts: List of prompts
            task: 'detection' or 'extraction' - controls max_new_tokens
                  detection = 32 tokens (Yes/No answer)
                  extraction = 256 tokens (cause/effect JSON)
        """
        all_responses = []
        batch_size = self.config.batch_size

        # OPTIMIZATION 5: Use task-specific max_new_tokens
        if task == 'detection':
            max_new_tokens = self.config.max_new_tokens_detection  # 32
        else:
            max_new_tokens = self.config.max_new_tokens_extraction  # 256

        print(f"Task: {task} | max_new_tokens: {max_new_tokens} | batch_size: {batch_size}")
        
        # Process in batches
        for i in tqdm(range(0, len(prompts), batch_size), desc=f"Batches ({self.config.name})"):
            batch = prompts[i:i + batch_size]
            responses = self.engine.generate(batch, max_new_tokens=max_new_tokens)
            all_responses.extend(responses)
        
        return all_responses
    
    def generate_single(self, prompt: str) -> str:
        """Generate for single prompt"""
        return self.engine.generate([prompt])[0]


def get_model_configs() -> Dict[str, ModelConfig]:
    """Pre-defined model configurations"""
    
    configs = {
        # -- Large models: 4-bit quantization enabled --------------------------
        'deepseek-r1-distill-qwen-32b': ModelConfig(
            name='DeepSeek-R1-Distill-Qwen-32B',
            model_path='deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
            model_type='huggingface',
            batch_size=4,
            max_new_tokens_detection=32,
            max_new_tokens_extraction=256,
            load_in_4bit=True
        ),
        'meta-llama-3.3-70b': ModelConfig(
            name='Meta-Llama-3.3-70B-Instruct',
            model_path='meta-llama/Llama-3.3-70B-Instruct',
            model_type='huggingface',
            batch_size=2,
            max_new_tokens_detection=32,
            max_new_tokens_extraction=256,
            load_in_4bit=True
        ),
        'deepseek-70b': ModelConfig(
            name='DeepSeek-70B',
            model_path='deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
            model_type='huggingface',
            batch_size=2,
            max_new_tokens_detection=32,
            max_new_tokens_extraction=256,
            load_in_4bit=True
        ),
        'mixtral-8x7b': ModelConfig(
            name='Mixtral-8x7B-Instruct-v0.1',
            model_path='mistralai/Mixtral-8x7B-Instruct-v0.1',
            model_type='huggingface',
            batch_size=4,
            max_new_tokens_detection=32,
            max_new_tokens_extraction=256,
            load_in_4bit=True
        ),
        
        # -- Small models: unchanged, no quantization ---------------------------
        'mistral-7b': ModelConfig(
            name='Mistral-7B',
            model_path='mistralai/Mistral-7B-Instruct-v0.2',
            model_type='huggingface',
            batch_size=8,
            max_new_tokens_detection=32,
            max_new_tokens_extraction=256
        ),
        'llama-3b': ModelConfig(
            name='Llama-3B',
            model_path='meta-llama/Llama-3.2-3B-Instruct',
            model_type='huggingface',
            batch_size=16,
            max_new_tokens_detection=32,
            max_new_tokens_extraction=256
        ),
        'llama-8b': ModelConfig(
            name='Llama-8B',
            model_path='meta-llama/Llama-3.1-8B-Instruct',
            model_type='huggingface',
            batch_size=8,
            max_new_tokens_detection=32,
            max_new_tokens_extraction=256
        ),
        'qwen-7b': ModelConfig(
            name='Qwen-7B',
            model_path='Qwen/Qwen2.5-7B-Instruct',
            model_type='huggingface',
            batch_size=8,
            max_new_tokens_detection=32,
            max_new_tokens_extraction=256
        ),
        'deepseek-7b': ModelConfig(
            name='DeepSeek-7B',
            model_path='deepseek-ai/deepseek-llm-7b-chat',
            model_type='huggingface',
            batch_size=8,
            max_new_tokens_detection=32,
            max_new_tokens_extraction=256
        ),
    }
    
    return configs


def test_inference():
    """Test inference with a sample prompt"""
    config = ModelConfig(
        name='Llama-3B-Test',
        model_path='meta-llama/Llama-3.2-3B-Instruct',
        model_type='huggingface',
        batch_size=2
    )
    
    inference = LLMInference(config)
    
    test_prompts = [
        "What is the capital of France?",
        "Explain photosynthesis in one sentence."
    ]
    
    print("\nTesting batch inference...")
    responses = inference.batch_generate(test_prompts, task='extraction')
    
    for prompt, response in zip(test_prompts, responses):
        print(f"\nPrompt: {prompt}")
        print(f"Response: {response}")


if __name__ == "__main__":
    test_inference()
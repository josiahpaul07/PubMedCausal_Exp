# ?? Fine-tune ALL 5 Models - All Splits - Background Execution
# Uses LoRA (Parameter-Efficient Fine-tuning)
# Models: llama-3b, llama-8b, mistral-7b, qwen-7b, deepseek-7b

# ============================================
# SETUP
# ============================================
mkdir -p ./logs
mkdir -p ./checkpoints
mkdir -p ./results/finetuned_models

# ============================================
# COMBINED SPLIT - ALL 5 MODELS
# ============================================

# Llama-3B - Combined
CUDA_VISIBLE_DEVICES=6 nohup python3 models/llm_finetune.py \
    --model_name llama-3b \
    --task_type extraction \
    --train_file ./data/prepared/extraction_combined/train.json \
    --test_file ./data/prepared/extraction_combined/test.json \
    --output_dir ./checkpoints/extraction_combined_llama3b \
    --num_epochs 3 \
    --batch_size 8 > ./logs/finetune_combined_llama3b.log 2>&1 && \
CUDA_VISIBLE_DEVICES=6 nohup python3 run_extraction.py \
    --test_file ./data/prepared/extraction_combined/test.json \
    --split_type combined \
    --models llama-3b-finetuned \
    --model_path ./checkpoints/extraction_combined_llama3b/final \
    --strategies zero-shot \
    --batch_size 16 \
    --results_dir ./results/finetuned_models > ./logs/test_finetuned_combined_llama3b.log 2>&1&

# Llama-8B - Combined
CUDA_VISIBLE_DEVICES=6 nohup python3 models/llm_finetune.py \
    --model_name llama-8b \
    --task_type extraction \
    --train_file ./data/prepared/extraction_combined/train.json \
    --test_file ./data/prepared/extraction_combined/test.json \
    --output_dir ./checkpoints/extraction_combined_llama8b \
    --num_epochs 3 \
    --batch_size 4 > ./logs/finetune_combined_llama8b.log 2>&1 && \
CUDA_VISIBLE_DEVICES=6 nohup python3 run_extraction.py \
    --test_file ./data/prepared/extraction_combined/test.json \
    --split_type combined \
    --models llama-8b-finetuned \
    --model_path ./checkpoints/extraction_combined_llama8b/final \
    --strategies zero-shot \
    --batch_size 8 \
    --results_dir ./results/finetuned_models > ./logs/test_finetuned_combined_llama8b.log 2>&1&

# Mistral-7B - Combined
CUDA_VISIBLE_DEVICES=6 nohup python3 models/llm_finetune.py \
    --model_name mistral-7b \
    --task_type extraction \
    --train_file ./data/prepared/extraction_combined/train.json \
    --test_file ./data/prepared/extraction_combined/test.json \
    --output_dir ./checkpoints/extraction_combined_mistral7b \
    --num_epochs 3 \
    --batch_size 4 > ./logs/finetune_combined_mistral7b.log 2>&1 && \
CUDA_VISIBLE_DEVICES=6 nohup python3 run_extraction.py \
    --test_file ./data/prepared/extraction_combined/test.json \
    --split_type combined \
    --models mistral-7b-finetuned \
    --model_path ./checkpoints/extraction_combined_mistral7b/final \
    --strategies zero-shot \
    --batch_size 8 \
    --results_dir ./results/finetuned_models > ./logs/test_finetuned_combined_mistral7b.log 2>&1&

# Qwen-7B - Combined
CUDA_VISIBLE_DEVICES=6 nohup python3 models/llm_finetune.py \
    --model_name qwen-7b \
    --task_type extraction \
    --train_file ./data/prepared/extraction_combined/train.json \
    --test_file ./data/prepared/extraction_combined/test.json \
    --output_dir ./checkpoints/extraction_combined_qwen7b \
    --num_epochs 3 \
    --batch_size 4 > ./logs/finetune_combined_qwen7b.log 2>&1 && \
CUDA_VISIBLE_DEVICES=6 nohup python3 run_extraction.py \
    --test_file ./data/prepared/extraction_combined/test.json \
    --split_type combined \
    --models qwen-7b-finetuned \
    --model_path ./checkpoints/extraction_combined_qwen7b/final \
    --strategies zero-shot \
    --batch_size 8 \
    --results_dir ./results/finetuned_models > ./logs/test_finetuned_combined_qwen7b.log 2>&1&

# DeepSeek-7B - Combined
CUDA_VISIBLE_DEVICES=6 nohup python3 models/llm_finetune.py \
    --model_name deepseek-7b \
    --task_type extraction \
    --train_file ./data/prepared/extraction_combined/train.json \
    --test_file ./data/prepared/extraction_combined/test.json \
    --output_dir ./checkpoints/extraction_combined_deepseek7b \
    --num_epochs 3 \
    --batch_size 4 > ./logs/finetune_combined_deepseek7b.log 2>&1 && \
CUDA_VISIBLE_DEVICES=6 nohup python3 run_extraction.py \
    --test_file ./data/prepared/extraction_combined/test.json \
    --split_type combined \
    --models deepseek-7b-finetuned \
    --model_path ./checkpoints/extraction_combined_deepseek7b/final \
    --strategies zero-shot \
    --batch_size 8 \
    --results_dir ./results/finetuned_models > ./logs/test_finetuned_combined_deepseek7b.log 2>&1&

# ============================================
# X_ONLY SPLIT - ALL 5 MODELS
# ============================================

# Llama-3B - X_only
CUDA_VISIBLE_DEVICES=6 nohup python3 models/llm_finetune.py \
    --model_name llama-3b \
    --task_type extraction \
    --train_file ./data/prepared/extraction_X_only/train.json \
    --test_file ./data/prepared/extraction_X_only/test.json \
    --output_dir ./checkpoints/extraction_X_only_llama3b \
    --num_epochs 3 \
    --batch_size 8 > ./logs/finetune_x_only_llama3b.log 2>&1 && \
CUDA_VISIBLE_DEVICES=6 nohup python3 run_extraction.py \
    --test_file ./data/prepared/extraction_X_only/test.json \
    --split_type X_only \
    --models llama-3b-finetuned \
    --model_path ./checkpoints/extraction_X_only_llama3b/final \
    --strategies zero-shot \
    --batch_size 16 \
    --results_dir ./results/finetuned_models > ./logs/test_finetuned_x_only_llama3b.log 2>&1&

# Llama-8B - X_only
CUDA_VISIBLE_DEVICES=6 nohup python3 models/llm_finetune.py \
    --model_name llama-8b \
    --task_type extraction \
    --train_file ./data/prepared/extraction_X_only/train.json \
    --test_file ./data/prepared/extraction_X_only/test.json \
    --output_dir ./checkpoints/extraction_X_only_llama8b \
    --num_epochs 3 \
    --batch_size 4 > ./logs/finetune_x_only_llama8b.log 2>&1 && \
CUDA_VISIBLE_DEVICES=6 nohup python3 run_extraction.py \
    --test_file ./data/prepared/extraction_X_only/test.json \
    --split_type X_only \
    --models llama-8b-finetuned \
    --model_path ./checkpoints/extraction_X_only_llama8b/final \
    --strategies zero-shot \
    --batch_size 8 \
    --results_dir ./results/finetuned_models > ./logs/test_finetuned_x_only_llama8b.log 2>&1&

# Mistral-7B - X_only
CUDA_VISIBLE_DEVICES=6 nohup python3 models/llm_finetune.py \
    --model_name mistral-7b \
    --task_type extraction \
    --train_file ./data/prepared/extraction_X_only/train.json \
    --test_file ./data/prepared/extraction_X_only/test.json \
    --output_dir ./checkpoints/extraction_X_only_mistral7b \
    --num_epochs 3 \
    --batch_size 4 > ./logs/finetune_x_only_mistral7b.log 2>&1 && \
CUDA_VISIBLE_DEVICES=6 nohup python3 run_extraction.py \
    --test_file ./data/prepared/extraction_X_only/test.json \
    --split_type X_only \
    --models mistral-7b-finetuned \
    --model_path ./checkpoints/extraction_X_only_mistral7b/final \
    --strategies zero-shot \
    --batch_size 8 \
    --results_dir ./results/finetuned_models > ./logs/test_finetuned_x_only_mistral7b.log 2>&1&

# Qwen-7B - X_only
CUDA_VISIBLE_DEVICES=6 nohup python3 models/llm_finetune.py \
    --model_name qwen-7b \
    --task_type extraction \
    --train_file ./data/prepared/extraction_X_only/train.json \
    --test_file ./data/prepared/extraction_X_only/test.json \
    --output_dir ./checkpoints/extraction_X_only_qwen7b \
    --num_epochs 3 \
    --batch_size 4 > ./logs/finetune_x_only_qwen7b.log 2>&1 && \
CUDA_VISIBLE_DEVICES=6 nohup python3 run_extraction.py \
    --test_file ./data/prepared/extraction_X_only/test.json \
    --split_type X_only \
    --models qwen-7b-finetuned \
    --model_path ./checkpoints/extraction_X_only_qwen7b/final \
    --strategies zero-shot \
    --batch_size 8 \
    --results_dir ./results/finetuned_models > ./logs/test_finetuned_x_only_qwen7b.log 2>&1&

# DeepSeek-7B - X_only
CUDA_VISIBLE_DEVICES=6 nohup python3 models/llm_finetune.py \
    --model_name deepseek-7b \
    --task_type extraction \
    --train_file ./data/prepared/extraction_X_only/train.json \
    --test_file ./data/prepared/extraction_X_only/test.json \
    --output_dir ./checkpoints/extraction_X_only_deepseek7b \
    --num_epochs 3 \
    --batch_size 4 > ./logs/finetune_x_only_deepseek7b.log 2>&1 && \
CUDA_VISIBLE_DEVICES=6 nohup python3 run_extraction.py \
    --test_file ./data/prepared/extraction_X_only/test.json \
    --split_type X_only \
    --models deepseek-7b-finetuned \
    --model_path ./checkpoints/extraction_X_only_deepseek7b/final \
    --strategies zero-shot \
    --batch_size 8 \
    --results_dir ./results/finetuned_models > ./logs/test_finetuned_x_only_deepseek7b.log 2>&1&

# ============================================
# Y_ONLY SPLIT - ALL 5 MODELS
# ============================================

# Llama-3B - Y_only
CUDA_VISIBLE_DEVICES=6 nohup python3 models/llm_finetune.py \
    --model_name llama-3b \
    --task_type extraction \
    --train_file ./data/prepared/extraction_Y_only/train.json \
    --test_file ./data/prepared/extraction_Y_only/test.json \
    --output_dir ./checkpoints/extraction_Y_only_llama3b \
    --num_epochs 3 \
    --batch_size 8 > ./logs/finetune_y_only_llama3b.log 2>&1 && \
CUDA_VISIBLE_DEVICES=6 nohup python3 run_extraction.py \
    --test_file ./data/prepared/extraction_Y_only/test.json \
    --split_type Y_only \
    --models llama-3b-finetuned \
    --model_path ./checkpoints/extraction_Y_only_llama3b/final \
    --strategies zero-shot \
    --batch_size 16 \
    --results_dir ./results/finetuned_models > ./logs/test_finetuned_y_only_llama3b.log 2>&1&

# Llama-8B - Y_only
CUDA_VISIBLE_DEVICES=6 nohup python3 models/llm_finetune.py \
    --model_name llama-8b \
    --task_type extraction \
    --train_file ./data/prepared/extraction_Y_only/train.json \
    --test_file ./data/prepared/extraction_Y_only/test.json \
    --output_dir ./checkpoints/extraction_Y_only_llama8b \
    --num_epochs 3 \
    --batch_size 4 > ./logs/finetune_y_only_llama8b.log 2>&1 && \
CUDA_VISIBLE_DEVICES=6 nohup python3 run_extraction.py \
    --test_file ./data/prepared/extraction_Y_only/test.json \
    --split_type Y_only \
    --models llama-8b-finetuned \
    --model_path ./checkpoints/extraction_Y_only_llama8b/final \
    --strategies zero-shot \
    --batch_size 8 \
    --results_dir ./results/finetuned_models > ./logs/test_finetuned_y_only_llama8b.log 2>&1&

# Mistral-7B - Y_only
CUDA_VISIBLE_DEVICES=6 nohup python3 models/llm_finetune.py \
    --model_name mistral-7b \
    --task_type extraction \
    --train_file ./data/prepared/extraction_Y_only/train.json \
    --test_file ./data/prepared/extraction_Y_only/test.json \
    --output_dir ./checkpoints/extraction_Y_only_mistral7b \
    --num_epochs 3 \
    --batch_size 4 > ./logs/finetune_y_only_mistral7b.log 2>&1 && \
CUDA_VISIBLE_DEVICES=6 nohup python3 run_extraction.py \
    --test_file ./data/prepared/extraction_Y_only/test.json \
    --split_type Y_only \
    --models mistral-7b-finetuned \
    --model_path ./checkpoints/extraction_Y_only_mistral7b/final \
    --strategies zero-shot \
    --batch_size 8 \
    --results_dir ./results/finetuned_models > ./logs/test_finetuned_y_only_mistral7b.log 2>&1&

# Qwen-7B - Y_only
CUDA_VISIBLE_DEVICES=6 nohup python3 models/llm_finetune.py \
    --model_name qwen-7b \
    --task_type extraction \
    --train_file ./data/prepared/extraction_Y_only/train.json \
    --test_file ./data/prepared/extraction_Y_only/test.json \
    --output_dir ./checkpoints/extraction_Y_only_qwen7b \
    --num_epochs 3 \
    --batch_size 4 > ./logs/finetune_y_only_qwen7b.log 2>&1 && \
CUDA_VISIBLE_DEVICES=6 nohup python3 run_extraction.py \
    --test_file ./data/prepared/extraction_Y_only/test.json \
    --split_type Y_only \
    --models qwen-7b-finetuned \
    --model_path ./checkpoints/extraction_Y_only_qwen7b/final \
    --strategies zero-shot \
    --batch_size 8 \
    --results_dir ./results/finetuned_models > ./logs/test_finetuned_y_only_qwen7b.log 2>&1&

# DeepSeek-7B - Y_only
CUDA_VISIBLE_DEVICES=6 nohup python3 models/llm_finetune.py \
    --model_name deepseek-7b \
    --task_type extraction \
    --train_file ./data/prepared/extraction_Y_only/train.json \
    --test_file ./data/prepared/extraction_Y_only/test.json \
    --output_dir ./checkpoints/extraction_Y_only_deepseek7b \
    --num_epochs 3 \
    --batch_size 4 > ./logs/finetune_y_only_deepseek7b.log 2>&1 && \
CUDA_VISIBLE_DEVICES=6 nohup python3 run_extraction.py \
    --test_file ./data/prepared/extraction_Y_only/test.json \
    --split_type Y_only \
    --models deepseek-7b-finetuned \
    --model_path ./checkpoints/extraction_Y_only_deepseek7b/final \
    --strategies zero-shot \
    --batch_size 8 \
    --results_dir ./results/finetuned_models > ./logs/test_finetuned_y_only_deepseek7b.log 2>&1&

# ============================================
# NOTE: Commands use && to chain fine-tuning and testing
# Each model fine-tunes THEN immediately tests
# Run ONE at a time, or change GPU numbers to parallelize
# ============================================
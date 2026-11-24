#!/bin/bash

# ==============================================================================
# NeuroLM Grid Search Script for SEED IV (15-Fold LOSO CV)
# ==============================================================================

# nohup ./run_grid_search_neurolm_seed4.sh < /dev/null > grid_search_seed4.log 2>&1 &

export WANDB_MODE=offline

# Grid Search Hyperparameters (Adjust as needed)
LRS=(1e-5 5e-5)
WDS=(0.01 0.1)
EPS=(50 80)
BSS=(16 32)

# LRS=(1e-5)
# WDS=(0.01)
# EPS=(5)
# BSS=(16)


# Paths Configuration
BASE_DIR="/homes/yy2541/henry/LLM_eva_fast/NeuroLM/NeuoLM_test"
PYTHON_SCRIPT="${BASE_DIR}/train_instruction_seediv.py" # Point to your SEED IV python script
BASE_LOG_DIR="${BASE_DIR}/grid_search_logs_neurolm_seed4"
DATASET_DIR="/homes/yy2541/henry/data/SEED_IV"
CKPT_PATH="${BASE_DIR}/checkpoints/NeuroLM-B.pt"

mkdir -p "$BASE_LOG_DIR"

total_jobs=$(( ${#LRS[@]} * ${#WDS[@]} * ${#EPS[@]} * ${#BSS[@]} ))
job_counter=1

echo "Starting NeuroLM Grid Search on SEED IV ($total_jobs configurations sequentially)"

for lr in "${LRS[@]}"; do
  for wd in "${WDS[@]}"; do
    for ep in "${EPS[@]}"; do
      for bs in "${BSS[@]}"; do
      
        out_dir="${BASE_LOG_DIR}/output_grid/lr_${lr}_wd_${wd}_ep_${ep}_bs_${bs}"
        log_file="${BASE_LOG_DIR}/run_lr_${lr}_wd_${wd}_ep_${ep}_bs_${bs}.log"
        
        CURRENT_PORT=$((29600 + job_counter))
        
        echo "------------------------------------------------------------------"
        echo "Launching job $job_counter/$total_jobs: LR=${lr}, WD=${wd}, Epochs=${ep}, Batch Size=${bs} on Port ${CURRENT_PORT}"
        echo "------------------------------------------------------------------"
        
        export CUDA_VISIBLE_DEVICES=0,1,2
        torchrun --nnodes=1 --nproc_per_node=3 --master_port="$CURRENT_PORT" \
          "$PYTHON_SCRIPT" \
          --dataset "SEEDIV" \
          --dataset_dir "$DATASET_DIR" \
          --out_dir "$out_dir" \
          --NeuroLM_path "$CKPT_PATH" \
          --seed 42 \
          --epochs "$ep" \
          --block_size 1024 \
          --eeg_batch_size "$bs" \
          --gradient_accumulation_steps 1 \
          --learning_rate "$lr" \
          --weight_decay "$wd" \
          > "$log_file" 2>&1
          
        job_counter=$((job_counter + 1))

      done
    done
  done
done

echo "Grid search completed! Analyzing 15-Fold LOSO results..."

# Inline Python script to parse logs and extract the best average metric across 15 folds
python3 -c "
import os
import re

base_dir = '${BASE_LOG_DIR}'
best_metric = -1.0
best_combo = ''

for log_file in os.listdir(base_dir):
    if log_file.startswith('run_') and log_file.endswith('.log'):
        combo = log_file.replace('run_', '').replace('.log', '')
        file_path = os.path.join(base_dir, log_file)
        
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Matches the format printed by the SEED IV Python evaluation loop
        match = re.search(r'Average Accuracy:\s*([0-9.]+)', content)
        if not match:
            match = re.search(r'Average Balanced Accuracy:\s*([0-9.]+)', content)
            
        if match:
            avg_acc = float(match.group(1))
            if avg_acc > best_metric:
                best_metric = avg_acc
                best_combo = combo

if best_combo:
    print(f'\n======================================================')
    print(f' BEST COMBINATION: {best_combo}')
    print(f' BEST AVERAGE ACCURACY (15 FOLDS): {best_metric:.4f}')
    print(f'======================================================\n')
else:
    print('\nCould not determine the best hyperparameters. Check stdout logs in ${BASE_LOG_DIR}.')
"
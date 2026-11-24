#!/bin/bash

# ==============================================================================
# NeuroLM Grid Search Script for TUAB (Sequential Execution)
# ==============================================================================

export WANDB_MODE=offline

# Grid Parameters
LRS=(1e-5 5e-5)
WDS=(0.01 0.1)
BSS=(32 64)
EPOCHS=15

# Setup logging directory for TUAB
BASE_LOG_DIR="/homes/yy2541/henry/LLM_eva_fast/NeuroLM/NeuoLM_test/grid_search_logs_neurolm_tuab"
mkdir -p "$BASE_LOG_DIR"

total_jobs=$(( ${#LRS[@]} * ${#WDS[@]} * ${#BSS[@]} ))
job_counter=1

echo "Starting NeuroLM Grid Search on TUAB ($total_jobs configurations sequentially)"

for lr in "${LRS[@]}"; do
  for wd in "${WDS[@]}"; do
    for bs in "${BSS[@]}"; do
      
      out_dir="${BASE_LOG_DIR}/output_grid/lr_${lr}_wd_${wd}_ep_${EPOCHS}_bs_${bs}"
      log_file="${BASE_LOG_DIR}/run_lr_${lr}_wd_${wd}_ep_${EPOCHS}_bs_${bs}.log"
      mkdir -p "$out_dir"
      
      CURRENT_PORT=$((29505 + job_counter))
      
      echo "------------------------------------------------------------------"
      echo "Launching job $job_counter/$total_jobs: LR=${lr}, WD=${wd}, Epochs=${EPOCHS}, Batch Size=${bs} on Port ${CURRENT_PORT}"
      echo "------------------------------------------------------------------"
      
      export CUDA_VISIBLE_DEVICES=3,4,5,6,7
      torchrun --nnodes=1 --nproc_per_node=5 --master_port="$CURRENT_PORT" \
        /homes/yy2541/henry/LLM_eva_fast/NeuroLM/NeuoLM_test/run_funetine_tuab.py \
        --dataset_dir "/homes/yy2541/henry" \
        --out_dir "$out_dir" \
        --NeuroLM_path "/homes/yy2541/henry/LLM_eva_fast/NeuroLM/NeuoLM_test/checkpoints/NeuroLM-B.pt" \
        --seeds "42,123,1,2,3" \
        --epochs "$EPOCHS" \
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

echo "Grid search completed! Analyzing results..."

# Inline Python script to parse the logs and find the highest average balanced accuracy
python3 -c "
import os
import re

base_dir = '${BASE_LOG_DIR}'
best_ba = -1.0
best_std = 0.0
best_combo = ''

for log_file in os.listdir(base_dir):
    if log_file.startswith('run_') and log_file.endswith('.log'):
        combo = log_file.replace('run_', '').replace('.log', '')
        file_path = os.path.join(base_dir, log_file)
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Extracts mean and std from the TUAB print format: 'balanced_accuracy    : 0.XXXX ± 0.XXXX'
            match = re.search(r'balanced_accuracy\s*:\s*([0-9.]+)\s*±\s*([0-9.]+)', content)
            if match:
                avg_ba = float(match.group(1))
                std_ba = float(match.group(2))
                if avg_ba > best_ba:
                    best_ba = avg_ba
                    best_std = std_ba
                    best_combo = combo
        except Exception:
            pass

if best_combo:
    print(f'\n======================================================')
    print(f' BEST COMBINATION: {best_combo}')
    print(f' AVERAGE BALANCED ACCURACY: {best_ba:.4f} ± {best_std:.4f}')
    print(f'======================================================\n')
else:
    print('\nCould not determine the best hyperparameters. Check stdout logs in ${BASE_LOG_DIR}.')
"
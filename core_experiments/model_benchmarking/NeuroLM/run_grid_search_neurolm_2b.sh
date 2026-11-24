#!/bin/bash

# ==============================================================================
# NeuroLM Grid Search Script for BCI IV 2b (Sequential Execution)
# ==============================================================================

# nohup ./run_grid_search_neurolm_2b.sh < /dev/null > grid_search_2b.log 2>&1 &

export WANDB_MODE=offline

# Grid Parameters
LRS=(1e-5 5e-5)
WDS=(0.01 0.1)
EPS=(50 80)
BSS=(16 32) # Added batch sizes

# LRS=(1e-5 5e-5)
# WDS=(0.01 0.1)
# EPS=(80)
# BSS=(8) # Added batch sizes

# LRS=(1e-5)
# WDS=(0.01)
# EPS=(80)
# BSS=(16) # Added batch sizes


# Setup logging directory
BASE_LOG_DIR="/homes/yy2541/henry/LLM_eva_fast/NeuroLM/NeuoLM_test/grid_search_logs_neurolm_2b"
mkdir -p "$BASE_LOG_DIR"

total_jobs=$(( ${#LRS[@]} * ${#WDS[@]} * ${#EPS[@]} * ${#BSS[@]} ))
job_counter=1

echo "Starting NeuroLM Grid Search ($total_jobs configurations sequentially)"

for lr in "${LRS[@]}"; do
  for wd in "${WDS[@]}"; do
    for ep in "${EPS[@]}"; do
      for bs in "${BSS[@]}"; do
      
        out_dir="${BASE_LOG_DIR}/output_grid/lr_${lr}_wd_${wd}_ep_${ep}_bs_${bs}"
        log_file="${BASE_LOG_DIR}/run_lr_${lr}_wd_${wd}_ep_${ep}_bs_${bs}.log"
        
        CURRENT_PORT=$((29505 + job_counter))
        
        echo "------------------------------------------------------------------"
        echo "Launching job $job_counter/$total_jobs: LR=${lr}, WD=${wd}, Epochs=${ep}, Batch Size=${bs} on Port ${CURRENT_PORT}"
        echo "------------------------------------------------------------------"
        
        # REMOVED the background '& disown' wrapper so jobs run one-by-one safely
        export CUDA_VISIBLE_DEVICES=0,1,2
        torchrun --nnodes=1 --nproc_per_node=3 --master_port="$CURRENT_PORT" \
          /homes/yy2541/henry/LLM_eva_fast/NeuroLM/NeuoLM_test/train_instruction_2b.py \
          --dataset "BCICIV2b" \
          --dataset_dir "/homes/yy2541/henry/data" \
          --out_dir "$out_dir" \
          --NeuroLM_path "/homes/yy2541/henry/LLM_eva_fast/NeuroLM/NeuoLM_test/checkpoints/NeuroLM-B.pt" \
          --seed 42 \
          --epochs "$ep" \
          --block_size 512 \
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

echo "Grid search completed! Analyzing results..."

# Inline Python script to parse the logs and find the highest average balanced accuracy
python3 -c "
import os
import re

base_dir = '${BASE_LOG_DIR}'
best_ba = -1.0
best_combo = ''

for log_file in os.listdir(base_dir):
    if log_file.startswith('run_') and log_file.endswith('.log'):
        combo = log_file.replace('run_', '').replace('.log', '')
        file_path = os.path.join(base_dir, log_file)
        
        with open(file_path, 'r') as f:
            content = f.read()
            
        match = re.search(r'Average Balanced Accuracy:\s*([0-9.]+)', content)
        if match:
            avg_ba = float(match.group(1))
            if avg_ba > best_ba:
                best_ba = avg_ba
                best_combo = combo

if best_combo:
    print(f'\n======================================================')
    print(f' BEST COMBINATION: {best_combo}')
    print(f' AVERAGE BALANCED ACCURACY (9 FOLDS): {best_ba:.4f}')
    print(f'======================================================\n')
else:
    print('\nCould not determine the best hyperparameters. Check stdout logs in ${BASE_LOG_DIR}.')
"
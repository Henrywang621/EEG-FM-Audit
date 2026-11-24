#!/bin/bash

# ==============================================================================
# NeuroLM Grid Search Script for TUEV (Sequential Execution)
# ==============================================================================

export WANDB_MODE=offline
export PYTHONPATH=/homes/yy2541/Henry_shine3/my_python_libs:$PYTHONPATH

# Grid Parameters
LRS=(1e-5 5e-5)
WDS=(0.01 0.1)
EPS=(10)
BSS=(32 64)


# Setup logging directory
BASE_LOG_DIR="/homes/yy2541/Henry_shine3/NeuroLM/NeuoLM_test/grid_search_logs_neurolm_tuev"
mkdir -p "$BASE_LOG_DIR"

total_jobs=$(( ${#LRS[@]} * ${#WDS[@]} * ${#EPS[@]} * ${#BSS[@]} ))
job_counter=1

echo "Starting NeuroLM TUEV Grid Search ($total_jobs configurations sequentially)"

for lr in "${LRS[@]}"; do
  for wd in "${WDS[@]}"; do
    for ep in "${EPS[@]}"; do
      for bs in "${BSS[@]}"; do
      
        out_dir="${BASE_LOG_DIR}/output_grid/lr_${lr}_wd_${wd}_ep_${ep}_bs_${bs}"
        log_file="${BASE_LOG_DIR}/run_lr_${lr}_wd_${wd}_ep_${ep}_bs_${bs}.log"
        mkdir -p "$out_dir"
        
        CURRENT_PORT=$((29600 + job_counter))
        
        echo "------------------------------------------------------------------"
        echo "Launching job $job_counter/$total_jobs: LR=${lr}, WD=${wd}, Epochs=${ep}, Batch Size=${bs} on Port ${CURRENT_PORT}"
        echo "Logs are being written to: $log_file"
        echo "------------------------------------------------------------------"
        
        export CUDA_VISIBLE_DEVICES=3,5
        
        # This command runs sequentially because there is no trailing '&'
        /mnt/faster3/xw2336/miniconda3_fast/envs/EEG26pytorch/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port="$CURRENT_PORT" \
          /homes/yy2541/Henry_shine3/NeuroLM/NeuoLM_test/train_instruction_tuev.py \
          --dataset_dir "/homes/yy2541/Henry_shine2" \
          --out_dir "$out_dir" \
          --NeuroLM_path "/homes/yy2541/Henry_shine3/NeuroLM/NeuoLM_test/checkpoints/NeuroLM-B.pt" \
          --seeds "42,123,1,2,3" \
          --epochs "$ep" \
          --block_size 1024 \
          --eeg_batch_size "$bs" \
          --gradient_accumulation_steps 1 \
          --learning_rate "$lr" \
          --weight_decay "$wd" \
          > "$log_file" 2>&1
          
        echo "Job $job_counter/$total_jobs has finished. Proceeding to the next configuration..."
        job_counter=$((job_counter + 1))

      done
    done
  done
done

echo "Grid search completed! Analyzing results..."

# Inline Python script to parse the logs and find the highest mean balanced accuracy
python3 -c "
import os
import re

base_dir = '${BASE_LOG_DIR}'
best_ba = -1.0
best_combo = ''
best_std = 0.0

for log_file in os.listdir(base_dir):
    if log_file.startswith('run_') and log_file.endswith('.log'):
        combo = log_file.replace('run_', '').replace('.log', '')
        file_path = os.path.join(base_dir, log_file)
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Extracts mean and std from: 'balanced_accuracy    : 0.XXXX ± 0.XXXX'
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
    print(f' MEAN BALANCED ACCURACY (SEEDS): {best_ba:.4f} ± {best_std:.4f}')
    print(f'======================================================\n')
else:
    print('\nCould not determine the best hyperparameters. Check stdout logs in ${BASE_LOG_DIR}.')
"
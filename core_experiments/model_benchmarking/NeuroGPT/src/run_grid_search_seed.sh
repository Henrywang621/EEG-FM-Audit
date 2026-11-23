#!/bin/bash

# ==============================================================================
# NeuroGPT Multi-GPU Grid Search Script for SEED-IV (Updated for BS & Epochs)
# ==============================================================================

# GPU allocation setup
CUDA_DEVICES=(4,5,6,7)
NUM_GPUS=${#CUDA_DEVICES[@]}

# Grid Parameters (Add/adjust your values here)
LRS=(1e-4 5e-4)
WDS=(0.01 0.1)
BSS=(16 32)       # Batch Sizes
EPS=(50 80)      # Epochs

# Setup logging directory
BASE_LOG_DIR="./grid_search_logs_neurogpt_seed_iv"
mkdir -p "$BASE_LOG_DIR"

gpu_counter=0
total_jobs=$(( ${#LRS[@]} * ${#WDS[@]} * ${#BSS[@]} * ${#EPS[@]} ))

echo "Starting Grid Search across ${NUM_GPUS} GPUs: (${CUDA_DEVICES[*]})"
echo "Note: Each job will execute a full 15-fold LOOCV internally."

for lr in "${LRS[@]}"; do
  for wd in "${WDS[@]}"; do
    for bs in "${BSS[@]}"; do
      for ep in "${EPS[@]}"; do
      
        # Select GPU ID
        gpu_id=${CUDA_DEVICES[$((gpu_counter % NUM_GPUS))]}
        out_dir="${BASE_LOG_DIR}/output_grid/lr_${lr}_wd_${wd}_bs_${bs}_ep_${ep}"
        log_file="${BASE_LOG_DIR}/run_lr_${lr}_wd_${wd}_bs_${bs}_ep_${ep}.log"
        
        echo "Launching job $((gpu_counter + 1))/$total_jobs on GPU ${gpu_id}: LR=${lr}, WD=${wd}, BS=${bs}, EP=${ep}"
        
        # Assign process exclusively to target GPU and pass arguments
        CUDA_VISIBLE_DEVICES=$gpu_id python train_seed.py \
          --pretrained-model /homes/yy2541/henry/LLM_eva_fast/NeuroGPT/src/results/models/seed_iv_pretrained_new/model_SeedTS.bin \
          --train-data-path /homes/yy2541/henry/data/SEED_IV/ \
          --training-style decoding \
          --use-encoder True \
          --fp16 False \
          --max-grad-norm 1.0 \
          --learning-rate "$lr" \
          --weight-decay "$wd" \
          --per-device-training-batch-size "$bs" \
          --per-device-validation-batch-size "$bs" \
          --num-epochs "$ep" \
          --log-dir "$out_dir" \
          > "$log_file" 2>&1 &
        
        gpu_counter=$((gpu_counter + 1))
        
        # Throttle: Wait whenever all GPUs have been assigned a task
        if [ $((gpu_counter % NUM_GPUS)) -eq 0 ]; then
          echo "All $NUM_GPUS GPUs assigned a job. Waiting for current batch to complete..."
          wait
        fi
        
      done
    done
  done
done

# Wait for remaining background jobs to finish
wait
echo "Grid search completed successfully!"

echo "Analyzing results to find the best hyperparameter combination..."

# Inline Python script to parse the fold CSVs / log files and find the highest average balanced accuracy
python3 -c "
import os
import re
import glob
import pandas as pd

base_dir = '${BASE_LOG_DIR}'
out_grid = os.path.join(base_dir, 'output_grid')
best_ba = -1.0
best_combo = ''

if os.path.exists(out_grid):
    for combo in os.listdir(out_grid):
        combo_dir = os.path.join(out_grid, combo)
        if not os.path.isdir(combo_dir):
            continue
        
        # Read test metrics across all 15 LOOCV fold folders
        csv_files = glob.glob(os.path.join(combo_dir, 'fold_*', 'test_metrics.csv'))
        bas = []
        for csv_f in csv_files:
            try:
                df = pd.read_csv(csv_f)
                if 'test_balanced_accuracy' in df.columns:
                    bas.append(df['test_balanced_accuracy'].iloc[0])
                elif 'balanced_accuracy' in df.columns:
                    bas.append(df['balanced_accuracy'].iloc[0])
            except Exception:
                pass
        
        if bas:
            avg_ba = sum(bas) / len(bas)
            if avg_ba > best_ba:
                best_ba = avg_ba
                best_combo = combo
        else:
            # Fallback: Parse stdout log file if CSVs aren't found
            log_f = os.path.join(base_dir, f'run_{combo}.log')
            if os.path.exists(log_f):
                with open(log_f, 'r') as f:
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
    print(f' AVERAGE BALANCED ACCURACY (15 FOLDS): {best_ba:.4f}')
    print(f'======================================================\n')
else:
    print('\nCould not determine the best hyperparameters. Check if logs were generated successfully.')
"
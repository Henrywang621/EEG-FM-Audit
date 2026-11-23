#!/bin/bash
# Usage:
# nohup ./run_grid_search_neurogpt_tuev.sh > grid_search_tuev.log 2>&1 &
# Kill command: pkill -f run_grid_search_neurogpt_tuev.sh; pkill -f train_tuev.py

# Configuration - Updated to include allocated GPUs
CUDA_DEVICES=(0 1 2 3)
NUM_GPUS=${#CUDA_DEVICES[@]}

LRS=(1e-4 5e-4)
WDS=(0.01 0.1)
BATCH_SIZES=(16 32)

PRETRAINED_CKPT="/homes/yy2541/henry/LLM_eva_fast/NeuroGPT/pretrained_model/pytorch_model.bin" # Adjust path as needed
BASE_LOG_DIR="./grid_search_logs_neurogpt_tuev"
mkdir -p "$BASE_LOG_DIR"

gpu_counter=0
total_jobs=$(( ${#LRS[@]} * ${#WDS[@]} * ${#BATCH_SIZES[@]} ))

for lr in "${LRS[@]}"; do
    for wd in "${WDS[@]}"; do
        for bs in "${BATCH_SIZES[@]}"; do
            # Select GPU ID
            gpu_id=${CUDA_DEVICES[$((gpu_counter % NUM_GPUS))]}
            out_dir="${BASE_LOG_DIR}/output_grid/lr_${lr}_wd_${wd}_bs_${bs}"
            log_file="${BASE_LOG_DIR}/run_lr_${lr}_wd_${wd}_bs_${bs}.log"
            
            echo "Launching job $((gpu_counter + 1))/$total_jobs on GPU ${gpu_id}: LR=${lr}, WD=${wd}, BS=${bs}"
            
            CUDA_VISIBLE_DEVICES=$gpu_id python train_tuev.py \
                --pretrained-model "$PRETRAINED_CKPT" \
                --learning-rate "$lr" \
                --weight-decay "$wd" \
                --num-decoding-classes 6 \
                --per-device-training-batch-size "$bs" \
                --per-device-validation-batch-size "$bs" \
                --num_train_epochs 15 \
                --log-dir "$out_dir" > "$log_file" 2>&1 &
                
            gpu_counter=$((gpu_counter + 1))

            # Throttle: Wait whenever all GPUs have been assigned a task
            if [ $((gpu_counter % NUM_GPUS)) -eq 0 ]; then
                echo "All $NUM_GPUS GPUs assigned a job. Waiting for current batch to complete..."
                wait
            fi
        done
    done
done

# Wait for remaining jobs to complete
wait
echo "NeuroGPT TUEV Grid Search completed successfully!"

echo "Analyzing results to find the best hyperparameter combination..."

# Inline Python script to parse the CSV logs and find the highest average balanced accuracy
python3 -c "
import os
import pandas as pd

base_dir = '${BASE_LOG_DIR}/output_grid'
best_ba = -1.0
best_combo = ''

if os.path.exists(base_dir):
    for combo in os.listdir(base_dir):
        csv_path = os.path.join(base_dir, combo, 'final_robustness_summary.csv')
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if 'balanced_accuracy' in df.columns:
                    avg_ba = df['balanced_accuracy'].mean()
                    if avg_ba > best_ba:
                        best_ba = avg_ba
                        best_combo = combo
            except Exception as e:
                pass

if best_combo:
    print(f'\n======================================================')
    print(f' BEST COMBINATION: {best_combo}')
    print(f' AVERAGE VAL BALANCED ACCURACY: {best_ba:.4f}')
    print(f'======================================================\n')
else:
    print('\nCould not determine the best hyperparameters. Check if CSV logs were generated successfully.')
"
#!/bin/bash
# nohup ./run_grid_search_seedIV.sh > grid_search_seedIV_main.log 2>&1 &
# pkill -f run_grid_search_seedIV.sh; pkill -f run_class_finetuning_seedIV.py

# Configuration
CUDA_DEVICES=(0 1 2 3)
NUM_GPUS=${#CUDA_DEVICES[@]}

# Grid Parameters
LRS=(1e-5 5e-6)
WDS=(0.01 0.05)
DROPOUTS=(0.2)
BATCH_SIZES=(16 32)
EPOCHS=(20 50)
DATASET="SEED_IV" 
FINETUNE_CKPT="/homes/yy2541/Henry1_parkin/LaBram/checkpoints/labram-base.pth"

# Isolated logging and output directories for SEED-IV
LOG_DIR="./grid_search_logs_seedIV"
mkdir -p "$LOG_DIR"

gpu_counter=0
total_jobs=$(( ${#LRS[@]} * ${#WDS[@]} * ${#DROPOUTS[@]} * ${#BATCH_SIZES[@]} * ${#EPOCHS[@]} ))

for lr in "${LRS[@]}"; do
    for wd in "${WDS[@]}"; do
        for dp in "${DROPOUTS[@]}"; do
            for bs in "${BATCH_SIZES[@]}"; do
                for ep in "${EPOCHS[@]}"; do
                    # Select the assigned physical GPU ID from array safely
                    gpu_id=${CUDA_DEVICES[$((gpu_counter % NUM_GPUS))]}
                    out_dir="./output_grid_seedIV/lr_${lr}_wd_${wd}_dp_${dp}_bs_${bs}_ep_${ep}"
                    log_file="${LOG_DIR}/run_lr_${lr}_wd_${wd}_dp_${dp}_bs_${bs}_ep_${ep}.log"
                    
                    echo "Launching job $((gpu_counter + 1))/$total_jobs on GPU ${gpu_id}: LR=${lr}, WD=${wd}, DP=${dp}, BS=${bs}, EP=${ep}"
                    
                    # NOTE: SEED-IV uses an input size of 800
                    CUDA_VISIBLE_DEVICES=$gpu_id python run_class_finetuning_seedIV.py \
                        --dataset "$DATASET" \
                        --model labram_base_patch200_200 \
                        --finetune "$FINETUNE_CKPT" \
                        --input_size 800 \
                        --lr "$lr" \
                        --weight_decay "$wd" \
                        --drop "$dp" \
                        --abs_pos_emb \
                        --batch_size "$bs" \
                        --epochs "$ep" \
                        --output_dir "$out_dir" > "$log_file" 2>&1 &
                        
                    gpu_counter=$((gpu_counter + 1))

                    # Throttle: Only wait AFTER all available GPUs have been assigned a job
                    if [ $((gpu_counter % NUM_GPUS)) -eq 0 ]; then
                        echo "All $NUM_GPUS GPUs assigned a job. Waiting for current batch to complete..."
                        wait
                    fi
                done
            done
        done
    done
done

# Wait for remaining jobs to finish
wait
echo "Grid search for SEED-IV completed successfully!"

echo "Analyzing log files to find the best hyperparameter combination..."

# Inline Python script to parse the output logs and find the highest average balanced accuracy across the 15 LOOCV folds
python3 -c "
import os
import re
import glob

log_dir = '${LOG_DIR}'
best_acc = -1.0
best_ba = -1.0
best_combo = ''

log_files = glob.glob(os.path.join(log_dir, 'run_*.log'))

for log_file in log_files:
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            
            # Search for the final LOOCV averages printed at the end of the script
            match_ba = re.search(r'Average Balanced Accuracy:\s*([0-9.]+)', content)
            match_acc = re.search(r'Average Accuracy:\s*([0-9.]+)', content)
            
            if match_ba and match_acc:
                avg_ba = float(match_ba.group(1))
                avg_acc = float(match_acc.group(1))
                
                # Using Balanced Accuracy to determine the best model (you can change this to avg_acc if preferred)
                if avg_ba > best_ba:
                    best_ba = avg_ba
                    best_acc = avg_acc
                    # Extract combination details from the filename
                    best_combo = os.path.basename(log_file).replace('.log', '').replace('run_', '')
    except Exception as e:
        pass

if best_combo:
    print(f'\n======================================================')
    print(f' BEST COMBINATION: {best_combo}')
    print(f' AVERAGE BALANCED ACCURACY (15 FOLDS): {best_ba:.4f}')
    print(f' AVERAGE ACCURACY (15 FOLDS):          {best_acc:.4f}')
    print(f'======================================================\n')
else:
    print('\nCould not determine the best hyperparameters. Check if the logs were generated successfully and LOOCV finished.')
"
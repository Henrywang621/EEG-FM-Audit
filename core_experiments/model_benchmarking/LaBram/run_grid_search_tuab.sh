#!/bin/bash
# ==============================================================================
# LaBraM Grid Search Script for TUAB/TUEV (Distributed Execution)
# ==============================================================================

# GPU and Python Environment Configuration
CUDA_DEVICES="3,4"
NUM_GPUS=2
PYTHON_BIN="/mnt/faster3/xw2336/miniconda3_fast/envs/EEG26pytorch/bin/python"

# Grid Parameters
LRS=(1e-4 5e-5)
WDS=(0.01 0.1)
BSS=(32 64)
EPOCHS=15

DATASET="TUAB"
FINETUNE_CKPT="/homes/yy2541/Henry_shine3/LaBraM/checkpoints/labram-base.pth"

BASE_LOG_DIR="./grid_search_logs_tuab"
mkdir -p "$BASE_LOG_DIR"

job_counter=1
total_jobs=$(( ${#LRS[@]} * ${#WDS[@]} * ${#BSS[@]} ))

echo "Starting LaBraM Grid Search on $DATASET ($total_jobs configurations sequentially on GPUs: $CUDA_DEVICES)"

for lr in "${LRS[@]}"; do
  for wd in "${WDS[@]}"; do
    for bs in "${BSS[@]}"; do

      out_dir="${BASE_LOG_DIR}/output_grid/lr_${lr}_wd_${wd}_ep_${EPOCHS}_bs_${bs}"
      log_file="${BASE_LOG_DIR}/run_lr_${lr}_wd_${wd}_ep_${EPOCHS}_bs_${bs}.log"
      
      CURRENT_PORT=$((29500 + job_counter))
      
      echo "------------------------------------------------------------------"
      echo "Launching job $job_counter/$total_jobs: LR=${lr}, WD=${wd}, EP=${EPOCHS}, BS=${bs} on Port ${CURRENT_PORT}"
      echo "------------------------------------------------------------------"
      
      export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
      $PYTHON_BIN -m torch.distributed.run \
        --nnodes=1 \
        --nproc_per_node=$NUM_GPUS \
        --master_port="$CURRENT_PORT" \
        run_class_finetuning_tuab_tuev.py \
        --dataset "$DATASET" \
        --model labram_base_patch200_200 \
        --finetune "$FINETUNE_CKPT" \
        --lr "$lr" \
        --weight_decay "$wd" \
        --batch_size "$bs" \
        --epochs "$EPOCHS" \
        --abs_pos_emb \
        --output_dir "$out_dir" \
        --log_dir "$out_dir" \
        > "$log_file" 2>&1

      job_counter=$((job_counter + 1))

    done
  done
done

echo "Grid search completed successfully! Total jobs executed: $total_jobs"
echo "Analyzing log files to find the best hyperparameter combination..."

# Inline Python script to parse stdout logs for the highest averaged Balanced Accuracy across 5 seeds
$PYTHON_BIN -c "
import os
import re
import glob

log_dir = '${BASE_LOG_DIR}'
best_ba = -1.0
best_combo = ''

log_files = glob.glob(os.path.join(log_dir, 'run_*.log'))

for log_file in log_files:
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            
            # Match the final aggregated Balanced Accuracy format across the seeds
            match = re.search(r'Balanced Accuracy:\s*([0-9.]+)', content)
            if match:
                avg_ba = float(match.group(1))
                if avg_ba > best_ba:
                    best_ba = avg_ba
                    best_combo = os.path.basename(log_file).replace('.log', '').replace('run_', '')
    except Exception:
        pass

if best_combo:
    print(f'\n======================================================')
    print(f' BEST COMBINATION: {best_combo}')
    print(f' AVERAGE BALANCED ACCURACY: {best_ba:.4f}')
    print(f'======================================================\n')
else:
    print('\nCould not determine the best hyperparameters. Check stdout logs in ${BASE_LOG_DIR}.')
"
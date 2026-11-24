#!/bin/bash
# ==============================================================================
# EEGPT Grid Search Script for SEED-IV (LOSO CV, Single Seed 42)
# ==============================================================================

# # GPU and Python Environment Configuration
# CUDA_DEVICES="2,3,4"
# NUM_GPUS=3
# PYTHON_BIN="/mnt/faster3/xw2336/miniconda3_fast/envs/EEG26pytorch/bin/python" # Replace with your specific python path if needed (e.g. /mnt/faster3/xw2336/miniconda3_fast/envs/EEG26pytorch/bin/python)

# # Grid Parameters
# LRS=(1e-4 1e-5)
# WDS=(0.01 0.1)
# EPS=(50 80)
# BSS=(32 64)

# GPU and Python Environment Configuration
CUDA_DEVICES="7"
NUM_GPUS=1
PYTHON_BIN="/mnt/faster3/xw2336/miniconda3_fast/envs/EEG26pytorch/bin/python" # Replace with your specific python path if needed (e.g. /mnt/faster3/xw2336/miniconda3_fast/envs/EEG26pytorch/bin/python)

# Grid Parameters
LRS=(1e4 5e-4)
WDS=(0.01 0.1)
EPS=(50 80)
BSS=(16 32)

# Data & Checkpoint Paths
DATA_DIR="/homes/yy2541/Henry_shine2/data/SEED_IV/"
PRETRAIN_CKPT="/homes/yy2541/Henry_shine3/EEGPT/checkpoints/eegpt_mcae_58chs_4s_large4E.ckpt"

BASE_LOG_DIR="./grid_search_logs_eegpt_seed_iv_D"
mkdir -p "$BASE_LOG_DIR"

job_counter=1
total_jobs=$(( ${#LRS[@]} * ${#WDS[@]} * ${#EPS[@]} * ${#BSS[@]} ))

echo "Starting EEGPT Grid Search on SEED-IV ($total_jobs configurations sequentially on GPU: $CUDA_DEVICES)"

for lr in "${LRS[@]}"; do
  for wd in "${WDS[@]}"; do
    for ep in "${EPS[@]}"; do
      for bs in "${BSS[@]}"; do

        out_dir="${BASE_LOG_DIR}/output_grid/lr_${lr}_wd_${wd}_ep_${ep}_bs_${bs}"
        log_file="${BASE_LOG_DIR}/run_lr_${lr}_wd_${wd}_ep_${ep}_bs_${bs}.log"
        
        echo "------------------------------------------------------------------"
        echo "Launching job $job_counter/$total_jobs: LR=${lr}, WD=${wd}, EP=${ep}, BS=${bs}"
        echo "------------------------------------------------------------------"
        
        export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
        $PYTHON_BIN EEGPT_SeedIV.py \
          --ckpt_path "$PRETRAIN_CKPT" \
          --data_dir "$DATA_DIR" \
          --lr "$lr" \
          --weight_decay "$wd" \
          --batch_size "$bs" \
          --epochs "$ep" \
          --devices "$NUM_GPUS" \
          --out_dir "$out_dir" \
          > "$log_file" 2>&1

        job_counter=$((job_counter + 1))

      done
    done
  done
done

echo "Grid search completed successfully! Total jobs executed: $total_jobs"
echo "Analyzing log files to find the best hyperparameter combination..."

# Parse stdout logs for the highest averaged Balanced Accuracy across the 15 subjects
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
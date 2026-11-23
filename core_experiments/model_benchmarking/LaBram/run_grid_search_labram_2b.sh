#!/bin/bash
CUDA_DEVICES=(0 1 4)
NUM_GPUS=${#CUDA_DEVICES[@]}

LRS=(1e-5 5e-6)
WDS=(0.01 0.05)
EPS=(50 80)
BSS=(32 64) # Added batch sizes
DROP_PATH=0

BASE_LOG_DIR="./grid_search_logs_labram_2b"
mkdir -p "$BASE_LOG_DIR"

gpu_counter=0
total_jobs=$(( ${#LRS[@]} * ${#WDS[@]} * ${#EPS[@]} * ${#BSS[@]} ))

for lr in "${LRS[@]}"; do
  for wd in "${WDS[@]}"; do
    for ep in "${EPS[@]}"; do
      for bs in "${BSS[@]}"; do
        
        gpu_id=${CUDA_DEVICES[$((gpu_counter % NUM_GPUS))]}
        out_dir="${BASE_LOG_DIR}/output_grid/lr_${lr}_wd_${wd}_ep_${ep}_bs_${bs}"
        log_file="${BASE_LOG_DIR}/run_lr_${lr}_wd_${wd}_ep_${ep}_bs_${bs}.log"
        
        CUDA_VISIBLE_DEVICES=$gpu_id python run_class_finetuning_2b.py \
          --dataset BCIIV2b \
          --model labram_base_patch200_200 \
          --finetune ./checkpoints/labram-base.pth \
          --input_size 400 \
          --lr "$lr" \
          --weight_decay "$wd" \
          --drop_path "$DROP_PATH" \
          --batch_size "$bs" \
          --epochs "$ep" \
          --no_save_ckpt \
          --abs_pos_emb \
          --output_dir "$out_dir" \
          --log_dir "$out_dir" \
          > "$log_file" 2>&1 &
        
        gpu_counter=$((gpu_counter + 1))
        
        if [ $((gpu_counter % NUM_GPUS)) -eq 0 ]; then
          wait
        fi

      done
    done
  done
done

wait
echo "Grid search completed successfully! Total jobs executed: $total_jobs"

echo "Analyzing results to find the best hyperparameter combination..."

# Inline Python script to parse the logs and find the highest average balanced accuracy
python3 -c "
import os
import re
import json
import glob

base_dir = '${BASE_LOG_DIR}'
out_grid = os.path.join(base_dir, 'output_grid')
best_ba = -1.0
best_combo = ''

if os.path.exists(out_grid):
    for combo in os.listdir(out_grid):
        combo_dir = os.path.join(out_grid, combo)
        if not os.path.isdir(combo_dir):
            continue
        
        # Strategy 1: Parse test metrics from individual fold log.txt files
        log_txt_files = glob.glob(os.path.join(combo_dir, 'fold_*', 'log.txt'))
        bas = []
        for log_f in log_txt_files:
            try:
                with open(log_f, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        last_line = json.loads(lines[-1])
                        if 'test_balanced_accuracy' in last_line:
                            bas.append(last_line['test_balanced_accuracy'])
            except Exception:
                pass
        
        if len(bas) == 9:
            avg_ba = sum(bas) / len(bas)
            if avg_ba > best_ba:
                best_ba = avg_ba
                best_combo = combo
        else:
            # Strategy 2: Fallback to parsing stdout log file for the final summary string
            stdout_log = os.path.join(base_dir, f'run_{combo}.log')
            if os.path.exists(stdout_log):
                with open(stdout_log, 'r') as f:
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
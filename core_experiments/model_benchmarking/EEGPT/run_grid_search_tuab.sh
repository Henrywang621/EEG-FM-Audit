#!/bin/bash
# ==============================================================================
# EEGPT Grid Search Script for TUAB
# ==============================================================================

CUDA_DEVICES="2,3,4"
NUM_GPUS=3
PYTHON_BIN="/mnt/faster3/xw2336/miniconda3_fast/envs/EEG26pytorch/bin/python"

LRS=(1e-4 1e-5)
WDS=(0.01 0.1)
BSS=(32 64)
EPS=(10)

BASE_LOG_DIR="./grid_search_logs_eegpt_tuab"
mkdir -p "$BASE_LOG_DIR"

job_counter=1
total_jobs=$(( ${#LRS[@]} * ${#WDS[@]} * ${#EPS[@]} * ${#BSS[@]} ))

echo "Starting EEGPT Grid Search on TUAB ($total_jobs configurations on GPUs: $CUDA_DEVICES)"

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
        
        # We pass --devices $NUM_GPUS so PyTorch Lightning automatically uses DDP
        $PYTHON_BIN EEGPT_TUAB.py \
          --lr "$lr" \
          --weight_decay "$wd" \
          --batch_size "$bs" \
          --epochs "$ep" \
          --out_dir "$out_dir" \
          --devices "$NUM_GPUS" \
          > "$log_file" 2>&1

        job_counter=$((job_counter + 1))

      done
    done
  done
done

echo "Grid search completed successfully! Total jobs executed: $total_jobs"
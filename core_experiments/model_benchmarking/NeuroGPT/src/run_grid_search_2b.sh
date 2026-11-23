#!/bin/bash

# ==============================================================================
# NeuroGPT Multi-GPU Grid Search Script for BCI IV 2b
# ==============================================================================

# GPU allocation setup
CUDA_DEVICES=(0 1 2 4 5 6 7)
NUM_GPUS=${#CUDA_DEVICES[@]}

# Define hyperparameter search space
LEARNING_RATES=(5e-4 1e-4)
BATCH_SIZES=(16 32)
WEIGHT_DECAYS=(0.01 0.1)
EPOCHS=(30 50)

# Setup logging directory
LOG_DIR="logs/grid_search_2b"
mkdir -p "$LOG_DIR"

gpu_idx=0

echo "Starting Grid Search across ${NUM_GPUS} GPUs: (${CUDA_DEVICES[*]})"
echo "Note: Each job will execute a full 9-fold LOSO CV internally."

for lr in "${LEARNING_RATES[@]}"; do
  for bs in "${BATCH_SIZES[@]}"; do
    for wd in "${WEIGHT_DECAYS[@]}"; do
      for ep in "${EPOCHS[@]}"; do
        
        # Pick current GPU from device pool
        TARGET_GPU=${CUDA_DEVICES[$gpu_idx]}
        
        RUN_NAME="lr${lr}_bs${bs}_wd${wd}_ep${ep}"
        LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
        
        echo "Launching -> LR: ${lr} | BS: ${bs} | WD: ${wd} | EP: ${ep} on GPU ${TARGET_GPU}"
        
        # Assign process exclusively to TARGET_GPU and pass the exact matching arguments
        CUDA_VISIBLE_DEVICES=$TARGET_GPU python train_2b.py \
          --training-style decoding \
          --learning-rate "$lr" \
          --per-device-training-batch-size "$bs" \
          --weight-decay "$wd" \
          --num-train-epochs "$ep" \
          --training-steps -1 \
          --run-name "$RUN_NAME" \
          > "$LOG_FILE" 2>&1 &
        
        # Advance GPU pointer
        gpu_idx=$(( (gpu_idx + 1) % NUM_GPUS ))
        
        # Wait when all available GPUs are saturated before launching the next batch
        if [ $gpu_idx -eq 0 ]; then
          wait
        fi

      done
    done
  done
done

# Wait for any remaining background jobs to finish
wait
echo "Grid search completed successfully. Check $LOG_DIR/ for results."
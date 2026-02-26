#!/bin/bash

# --- Experiment Settings ---
# GPU Settings
export CUDA_VISIBLE_DEVICES=4,5,6,7  # Set available GPUs
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c)
NUM_GPUS=$((NUM_GPUS + 1))

# Project Settings
PROJECT_NAME="ppg-simclr"
EXPERIMENT_NAME="simclr"
WANDB_ENTITY=""  # Replace with your WandB entity

# Data Settings
# Dataset paths are hardcoded in main.py based on base.yaml mapping
DATASETS="pulsedb mcmed hsp cfs mesa"  # Space-separated list of datasets (pulsedb, mesa, hsp, mcmed, cfs)

# Training Hyperparameters
EPOCHS=5
# BATCH_SIZE=5
BATCH_SIZE=4 # Increased batch size for SimCLR
LEARNING_RATE=4e-4 # SimCLR usually benefits from larger LR (LARS/AdamW)
# LEARNING_RATE=5e-4
WEIGHT_DECAY=1e-2
MIXED_PRECISION="bf16" # no, fp16, bf16

# Model Hyperparameters
EMBED_DIM=128
TEMPERATURE=0.07

# Checkpoint & Logging Settings
SAVE_LOG_DIR="./logs"
SAVE_CKPT_DIR="./checkpoints"
SAVE_RESULT_DIR="./results"
LOG_INTERVAL=1
SAVE_INTERVAL=500

# --- Execution ---

# Using torchrun for distributed training (DDP)
# If using a single GPU, you can also run `python main.py ...` directly
MASTER_PORT=29502 # Use a different port to avoid conflicts
cmd="torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT main.py \
    --memo $EXPERIMENT_NAME \
    --datasets $DATASETS \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --mixed_precision $MIXED_PRECISION \
    --embed_dim $EMBED_DIM \
    --temperature $TEMPERATURE \
    --save_log_dir $SAVE_LOG_DIR \
    --save_ckpt_dir $SAVE_CKPT_DIR \
    --save_result_dir $SAVE_RESULT_DIR \
    --log_interval $LOG_INTERVAL \
    --save_interval $SAVE_INTERVAL \
    --wandb_project $PROJECT_NAME \
    --train"

if [ -n "$WANDB_ENTITY" ]; then
    cmd="$cmd --wandb_entity \"$WANDB_ENTITY\""
fi

echo "Running command:"
echo $cmd

# Execute
eval $cmd

#!/bin/bash

CONFIG_FILE="$1"
CHECKPOINT_DIR="$2"
CHECKPOINT_NUM="$3"

if [ -z "$CONFIG_FILE" ] || [ -z "$CHECKPOINT_DIR" ] || [ -z "$CHECKPOINT_NUM" ]; then
    echo "Usage: $0 <config_file.yaml> <checkpoint_folder> <checkpoint_number>"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file does not exist: $CONFIG_FILE"
    exit 1
fi

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Checkpoint directory does not exist: $CHECKPOINT_DIR"
    exit 1
fi

# Pad checkpoint number to 6 digits (e.g., 10000 -> 010000)
PADDED_CKPT_NUM=$(printf "%06d" "$CHECKPOINT_NUM")
CHECKPOINT_FILE="${CHECKPOINT_DIR}/${PADDED_CKPT_NUM}.pth"

if [ ! -f "$CHECKPOINT_FILE" ]; then
    echo "Checkpoint ${CHECKPOINT_NUM}.pth not found in $CHECKPOINT_DIR"
    exit 1
fi

# Determine experiment name
EXP_NAME=$(basename "$CHECKPOINT_DIR")_${CHECKPOINT_NUM}

# Detect model type
MODEL_FLAG="--model=original"
if [[ "$CHECKPOINT_FILE" == *DEVO.pth ]]; then
    MODEL_FLAG="--model=DEVO"
fi

# Check pyramid flag
USE_PYRAMID="--use_pyramid=True"
if [[ "$CHECKPOINT_DIR" == *nopyr* ]]; then
    USE_PYRAMID="--use_pyramid=False"
fi

# Output and run setup
OUTPUT_DIR="/usr/scratch/badile43/amarchei/checkpoints/${EXP_NAME}"
mkdir -p "$OUTPUT_DIR"
CSV_NAME="${EXP_NAME}.csv"
CSV_PATH="${OUTPUT_DIR}/${CSV_NAME}"

echo "Using checkpoint: $CHECKPOINT_FILE"
echo "Experiment name: $EXP_NAME"

echo "Output directory: $OUTPUT_DIR"
echo "CSV path: $CSV_PATH"
echo "Model flag: $MODEL_FLAG"
echo "Pyramid flag: $USE_PYRAMID"


python evals/eval_evs/eval_mvsec_evs.py \
    --config="$CONFIG_FILE" \
    --weights="$CHECKPOINT_FILE" \
    $MODEL_FLAG \
    --outdir="$OUTPUT_DIR" \
    --expname="$EXP_NAME" \
    --trials=1 \
    --plot \
    $USE_PYRAMID \
    --csv_name="$CSV_PATH" \
    --save_csv

echo "✅ Done evaluation for PATCHES=$PATCHES, REMOVAL_WINDOW=$REMOVAL_WINDOW"

echo "🎉 All evaluations complete. Final CSV path: $CSV_PATH"

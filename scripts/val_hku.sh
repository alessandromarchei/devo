#!/bin/bash

CONFIG_FILE="$1"
CHECKPOINT_INPUT="$2"

if [ -z "$CONFIG_FILE" ] || [ -z "$CHECKPOINT_INPUT" ]; then
    echo "Usage: $0 <config_file.yaml> <checkpoint_folder_or_file.pth>"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file does not exist: $CONFIG_FILE"
    exit 1
fi

USE_PYRAMID="--use_pyramid=True"
MODEL_FLAG="--model=original"

if [[ "$CHECKPOINT_INPUT" == *.pth ]]; then
    # Input is a .pth file directly
    LATEST_CKPT_FILE="$CHECKPOINT_INPUT"
    EXP_NAME=$(basename "$CHECKPOINT_INPUT" .pth)

    # If file name is DEVO.pth => use --model=DEVO
    if [[ "$LATEST_CKPT_FILE" == *DEVO.pth ]]; then
        MODEL_FLAG="--model=DEVO"
    fi
else
    # Input is a folder; find latest checkpoint
    if [ ! -d "$CHECKPOINT_INPUT" ]; then
        echo "Checkpoint folder does not exist: $CHECKPOINT_INPUT"
        exit 1
    fi

    if [[ "$CHECKPOINT_INPUT" == *nopyr* ]]; then
        USE_PYRAMID="--use_pyramid=False"
    fi

    LATEST_STEP=$(ls "$CHECKPOINT_INPUT"/*.pth | grep -oE '[0-9]+' | sort -n | tail -1)
    LATEST_CKPT_FILE=$(ls "$CHECKPOINT_INPUT"/*"$LATEST_STEP"*.pth)

    if [ -z "$LATEST_CKPT_FILE" ]; then
        echo "No valid checkpoint found in $CHECKPOINT_INPUT"
        exit 1
    fi

    EXP_NAME=$(basename "$CHECKPOINT_INPUT")
fi

CSV_NAME="${EXP_NAME}_hku.csv"
CSV_PATH="/usr/scratch/badile43/amarchei/checkpoints/${CSV_NAME}"

echo "Using checkpoint: $LATEST_CKPT_FILE"
echo "Experiment name: $EXP_NAME"
echo "CSV path: $CSV_PATH"
echo "Model flag: $MODEL_FLAG"
echo "Pyramid flag: $USE_PYRAMID"

for PATCHES in $(seq 96 -8 8); do
    echo "Evaluating with PATCHES_PER_FRAME = $PATCHES"

    TMP_CONFIG="tmp_eval_config_hku_${PATCHES}.yaml"    
    cp "$CONFIG_FILE" "$TMP_CONFIG"

    # Replace the PATCHES_PER_FRAME value
    sed -i "s/^PATCHES_PER_FRAME: .*/PATCHES_PER_FRAME: $PATCHES/" "$TMP_CONFIG"

    python evals/eval_evs/eval_hku_evs.py \
        --config="$TMP_CONFIG" \
        --weights="$LATEST_CKPT_FILE" \
        $MODEL_FLAG \
        --expname="$EXP_NAME" \
        --trials=1 \
        $USE_PYRAMID \
        --csv_name="$CSV_PATH" \
        --save_csv \

    echo "✅ Done evaluation for PATCHES_PER_FRAME = $PATCHES"
done

echo "🎉 All evaluations complete. Results written to: $CSV_PATH"

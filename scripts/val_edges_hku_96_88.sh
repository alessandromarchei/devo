#!/bin/bash

CONFIG_FILE="$1"
CHECKPOINT_INPUT="$2"
MACHINE_NAME="$3"

if [ -z "$CONFIG_FILE" ] || [ -z "$CHECKPOINT_INPUT" ] || [ -z "$MACHINE_NAME" ]; then
    echo "Usage: $0 <config_file.yaml> <checkpoint_folder_or_file.pth> <machine_name>"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file does not exist: $CONFIG_FILE"
    exit 1
fi

RUN_ID=$(date +"%Y%m%d_%H%M%S")_$RANDOM

USE_PYRAMID="--use_pyramid=True"
MODEL_FLAG="--model=original"

if [[ "$CHECKPOINT_INPUT" == *.pth ]]; then
    LATEST_CKPT_FILE="$CHECKPOINT_INPUT"
    EXP_NAME=$(basename "$CHECKPOINT_INPUT" .pth)

    if [[ "$LATEST_CKPT_FILE" == *DEVO.pth ]]; then
        MODEL_FLAG="--model=DEVO"
    fi
else
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

OUTPUT_DIR="/usr/scratch/badile43/amarchei/checkpoints/DEVO_nosoft_nedges_hku"
mkdir -p "$OUTPUT_DIR"

CSV_NAME="${EXP_NAME}_${MACHINE_NAME}.csv"
CSV_PATH="${OUTPUT_DIR}/${CSV_NAME}"

echo "Using checkpoint: $LATEST_CKPT_FILE"
echo "Experiment name: $EXP_NAME"
echo "Machine name: $MACHINE_NAME"
echo "Run ID: $RUN_ID"
echo "Output directory: $OUTPUT_DIR"
echo "CSV path: $CSV_PATH"
echo "Model flag: $MODEL_FLAG"
echo "Pyramid flag: $USE_PYRAMID"

# Custom PATCHES list
# Define PATCH_RANGE and REMOVAL_WINDOWS manually per machine or via env
PATCH_RANGE=$(seq 96 -8 88)
REMOVAL_WINDOWS=(22 16 12 8)

for PATCHES in $PATCH_RANGE; do
    for REMOVAL_WINDOW in "${REMOVAL_WINDOWS[@]}"; do
        echo "Evaluating with PATCHES_PER_FRAME = $PATCHES, REMOVAL_WINDOW = $REMOVAL_WINDOW"

        TMP_CONFIG="tmp_eval_config_${PATCHES}_r${REMOVAL_WINDOW}_${RUN_ID}.yaml"
        cp "$CONFIG_FILE" "$TMP_CONFIG"

        sed -i "s/^PATCHES_PER_FRAME: .*/PATCHES_PER_FRAME: $PATCHES/" "$TMP_CONFIG"
        sed -i "s/^REMOVAL_WINDOW: .*/REMOVAL_WINDOW: $REMOVAL_WINDOW/" "$TMP_CONFIG"

        python evals/eval_evs/eval_hku_evs.py \
            --config="$TMP_CONFIG" \
            --weights="$LATEST_CKPT_FILE" \
            $MODEL_FLAG \
            --outdir="$OUTPUT_DIR" \
            --expname="$EXP_NAME" \
            --trials=3 \
            --plot \
            $USE_PYRAMID \
            --csv_name="$CSV_PATH" \
            --save_csv \
            --use_softagg=False

        echo "✅ Done evaluation for PATCHES=$PATCHES, REMOVAL_WINDOW=$REMOVAL_WINDOW"
    done
done

echo "🎉 All evaluations complete. Results written to: $CSV_PATH"

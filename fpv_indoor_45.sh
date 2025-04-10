#!/bin/bash

# Force matplotlib to use non-interactive backend
export MPLBACKEND=Agg

YAML_FILE="config/configs_fpv_indoor_45.yaml"
SPLIT_FILE="splits/fpv/fpv_val_indoor_45.txt"
CSV_NAME="eval_fpv_indoor_45.csv"
TRIALS=2
DATASET_PATH="/usr/scratch/badile13/amarchei/fpv/"

# PATCHES_PER_FRAME must be divisible by 4
patches_per_frame_values=(96 44 24 12)
removal_window_values=(22 10 5)
optimization_window_values=(10 5)


SCENES=(
    "Indoor_45_2_Davis_With_Gt"
    "Indoor_45_4_Davis_With_Gt"
    "Indoor_45_9_Davis_With_Gt"
    "Indoor_45_12_Davis_With_Gt"
    "Indoor_45_13_Davis_With_Gt"
    "Indoor_45_14_Davis_With_Gt"
    "Indoor_45_1_Davis"
    "Indoor_45_3_Davis"
    "Indoor_45_11_Davis"
    "Indoor_45_16_Davis"
)


for scene in "${SCENES[@]}"; do
  for removal in "${removal_window_values[@]}"; do
    for patches in "${patches_per_frame_values[@]}"; do
      for opt in "${optimization_window_values[@]}"; do

        echo "Checking SCENE=$scene, PATCHES=$patches, REMOVAL=$removal, OPT=$opt"

        # Check if the config already exists in the CSV
        if grep -q "$scene,$patches,$opt,$removal" "$CSV_NAME"; then
          echo " → Skipping: already evaluated"
          continue
        fi

        echo " → Running evaluation..."

        # Modify YAML config
        sed -i "s/^PATCHES_PER_FRAME: .*/PATCHES_PER_FRAME: $patches/" "$YAML_FILE"
        sed -i "s/^REMOVAL_WINDOW: .*/REMOVAL_WINDOW: $removal/" "$YAML_FILE"
        sed -i "s/^OPTIMIZATION_WINDOW: .*/OPTIMIZATION_WINDOW: $opt/" "$YAML_FILE"

        # Run the model
        python evals/eval_evs/eval_fpv_evs.py \
          --datapath="$DATASET_PATH" \
          --weights="DEVO.pth" \
          --trials=$TRIALS \
          --save_csv \
          --config="$YAML_FILE" \
          --val_split="$SPLIT_FILE" \
          --csv_name="$CSV_NAME" \
          --expname="fpv_45_${patches}_${removal}_${opt}_${scene}"
        sleep 1

      done
    done
  done
done

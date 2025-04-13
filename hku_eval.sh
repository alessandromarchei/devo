#!/bin/bash

# Force matplotlib to use non-interactive backend
export MPLBACKEND=Agg

YAML_FILE="config/configs_hku.yaml"
SPLIT_FILE="splits/hku/hku_val.txt"
CSV_NAME="eval_hku.csv"
TRIALS=1
DATASET_PATH="/usr/scratch/badile13/amarchei/HKU/"

# PATCHES_PER_FRAME must be divisible by 4
patches_per_frame_values=(12)
removal_window_values=(22 10 5)
optimization_window_values=(10 5)

SCENES=(
  
  "Hku_Aggressive_Translation"
  "Hku_Aggressive_Rotation"
  "Hku_Aggressive_Small_Flip"
  "Hku_Aggressive_Walk"
  "Hku_Hdr_Circle"
  "Hku_Hdr_Slow"
  "Hku_Hdr_Tran_Rota"
  "Hku_Hdr_Agg"
  "Hku_Dark_Normal"
)

for scene in "${SCENES[@]}"; do
  for removal in "${removal_window_values[@]}"; do
    for patches in "${patches_per_frame_values[@]}"; do
      for opt in "${optimization_window_values[@]}"; do

        echo "Checking SCENE=$scene, PATCHES=$patches, REMOVAL=$removal, OPT=$opt"

        match=$(grep "$scene,$patches,$opt,$removal" "$CSV_NAME")

        echo " → Match: $match"
        if [[ -n "$match" ]]; then
          echo " → Found match in CSV:"
          echo "$match"
          echo " → Skipping: already evaluated"
          continue
        fi

        echo " → Running evaluation..."

        # Modify YAML config
        sed -i "s/^PATCHES_PER_FRAME: .*/PATCHES_PER_FRAME: $patches/" "$YAML_FILE"
        sed -i "s/^REMOVAL_WINDOW: .*/REMOVAL_WINDOW: $removal/" "$YAML_FILE"
        sed -i "s/^OPTIMIZATION_WINDOW: .*/OPTIMIZATION_WINDOW: $opt/" "$YAML_FILE"

        # Run the model
        python evals/eval_evs/eval_hku_evs.py \
          --datapath="$DATASET_PATH" \
          --weights="DEVO.pth" \
          --trials=$TRIALS \
          --save_csv \
          --config="$YAML_FILE" \
          --val_split="$SPLIT_FILE" \
          --csv_name="$CSV_NAME" \
          --expname="hku_${patches}_${removal}_${opt}_${scene}" \
          --model="DEVO"
        sleep 1

      done
    done
  done
done

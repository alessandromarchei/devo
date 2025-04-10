#!/bin/bash

# Force matplotlib to use non-interactive backend
export MPLBACKEND=Agg

YAML_FILE="config/configs_mvsec_outnight.yaml"
SPLIT_FILE="splits/mvsec/mvsec_val_outnight.txt"
CSV_NAME="eval_outnight.csv"
TRIALS=2

# Parameter values to test
patches_per_frame_values=(96 44 24 12)
removal_window_values=(22 10 5)
optimization_window_values=(10 5)

SCENES=("Outdoor_Night1_Data" "Outdoor_Night2_Data" "Outdoor_Night3_Data")

for scene in "${SCENES[@]}"; do
  for removal in "${removal_window_values[@]}"; do
    for patches in "${patches_per_frame_values[@]}"; do
      for opt in "${optimization_window_values[@]}"; do

        echo "Checking SCENE=$scene, PATCHES=$patches, REMOVAL=$removal, OPT=$opt"

        # Check if result is already in CSV
        if grep -q "$scene,$patches,$opt,$removal" "$CSV_NAME"; then
          echo " → Skipping: already evaluated."
          continue
        fi

        echo " → Running evaluation..."

        # Modify YAML file
        sed -i "s/^PATCHES_PER_FRAME: .*/PATCHES_PER_FRAME: $patches/" "$YAML_FILE"
        sed -i "s/^REMOVAL_WINDOW: .*/REMOVAL_WINDOW: $removal/" "$YAML_FILE"
        sed -i "s/^OPTIMIZATION_WINDOW: .*/OPTIMIZATION_WINDOW: $opt/" "$YAML_FILE"

        # Run the evaluation
        python evals/eval_evs/eval_mvsec_evs.py \
          --datapath=/usr/scratch/badile13/amarchei/mvsec/outdoor_night/ \
          --weights="DEVO.pth" \
          --trials=$TRIALS \
          --save_csv \
          --config="$YAML_FILE" \
          --val_split="$SPLIT_FILE" \
          --csv_name="$CSV_NAME"

        sleep 1

      done
    done
  done
done

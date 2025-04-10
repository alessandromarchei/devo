#!/bin/bash
# Force matplotlib to use non-interactive backend
export MPLBACKEND=Agg

YAML_FILE="config/configs_mvsec_indoor.yaml"
CSV_NAME="eval_indoor_flying.csv"

patches_per_frame_values=(96 44 24 12)
removal_window_values=(22 10 5)
optimization_window_values=(10 5)
TRIALS=2

SCENES=("Indoor_Flying1_Data" "Indoor_Flying2_Data" "Indoor_Flying3_Data" "Indoor_Flying4_Data")

for scene in "${SCENES[@]}"; do
  for removal in "${removal_window_values[@]}"; do
    for patches in "${patches_per_frame_values[@]}"; do
      for opt in "${optimization_window_values[@]}"; do

        echo "Checking PATCHES=$patches, REMOVAL=$removal, OPT=$opt, SCENE=$scene"

        # Check if the config was already evaluated
        if grep -q "$scene,$patches,$opt,$removal" "$CSV_NAME"; then
          echo " → Skipping: already in CSV"
          continue
        fi

        echo " → Running: PATCHES=$patches, OPT=$opt, REMOVAL=$removal"

        # Modify YAML file
        sed -i "s/^PATCHES_PER_FRAME: .*/PATCHES_PER_FRAME: $patches/" "$YAML_FILE"
        sed -i "s/^REMOVAL_WINDOW: .*/REMOVAL_WINDOW: $removal/" "$YAML_FILE"
        sed -i "s/^OPTIMIZATION_WINDOW: .*/OPTIMIZATION_WINDOW: $opt/" "$YAML_FILE"

        # Run evaluation
        python evals/eval_evs/eval_mvsec_evs.py --datapath=/usr/scratch/badile13/amarchei/mvsec/indoor_flying/ --weights="DEVO.pth" --trials=$TRIALS --save_csv --config="$YAML_FILE" --csv_name="$CSV_NAME"

        sleep 1
      done
    done
  done
done

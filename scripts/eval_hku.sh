#!/bin/bash

# Fixed paths
DATAPATH="/usr/scratch/badile13/amarchei/HKU/"
CONFIG="config/eval_hku.yaml"
EVAL_SCRIPT="evals/eval_evs/eval_hku_evs.py"
LOG_DIR="logs/eval"

# Make sure the log dir exists
mkdir -p "$LOG_DIR"

# List of model folders
models=(
  #"baseline_320x240"
  # "baseline_nopyr"
  # "patchifier_smallv2"
  "patchifier_small_v3"
  "patchifier_small_v4"
  "v4_nopyr"
)

for model_folder in "${models[@]}"; do
  # Determine model type and feature dimensions
  if [[ "$model_folder" == baseline* ]]; then
    model_type="original"
    dim_inet=384
    dim_fnet=128
  else
    model_type="mksmall"
    case "$model_folder" in
      patchifier_small_v3)
        dim_inet=196
        dim_fnet=64
        ;;
      patchifier_small_v4 | v4_nopyr)
        dim_inet=96
        dim_fnet=32
        ;;
      *)
        dim_inet=384
        dim_fnet=128
        ;;
    esac
  fi

  # Set pyramid flag based on model name
  if [[ "$model_folder" == *nopyr* ]]; then
    use_pyramid="False"
  else
    use_pyramid="True"
  fi

  # Define paths
  weights_path="checkpoints/${model_folder}/100000.pth"
  csv_name="eval_hku_${model_folder}.csv"
  log_file="${LOG_DIR}/${model_folder}_hku.txt"

  # Timestamp
  echo "[$(date)] Starting evaluation for $model_folder" | tee -a "$log_file"

  # Run evaluation
  python "$EVAL_SCRIPT" \
    --weights "$weights_path" \
    --config "$CONFIG" \
    --model "$model_type" \
    --expname "$csv_name" \
    --csv_name "$csv_name" \
    --save_csv \
    --trials 2 \
    --datapath "$DATAPATH" \
    --dim_inet "$dim_inet" \
    --dim_fnet "$dim_fnet" \
    --use_pyramid "$use_pyramid" \
    >> "$log_file" 2>&1

  echo "[$(date)] Finished evaluation for $model_folder" | tee -a "$log_file"
  echo "-----------------------------------------------------" | tee -a "$log_file"
done

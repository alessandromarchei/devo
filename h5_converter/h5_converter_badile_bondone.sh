#!/bin/bash

# ========== CONFIG ==========
H5_LIST_FILE=$1           # Text file with paths to events.h5
CONFIG_PATH=$2            # Path to YAML config
NUM_WORKERS=${3:-4}       # Optional: number of parallel jobs (default = 4)
DEBUG_FLAG=$4             # Optional: "--debug"
# ============================

if [ -z "$H5_LIST_FILE" ] || [ -z "$CONFIG_PATH" ]; then
    echo "Usage: $0 <h5_list.txt> <config.yml> [num_workers] [--debug]"
    exit 1
fi

if [ ! -f "$H5_LIST_FILE" ]; then
    echo "[ERROR] File not found: $H5_LIST_FILE"
    exit 1
fi

echo "[INFO] Reading H5 paths from: $H5_LIST_FILE"
echo "[INFO] Using config: $CONFIG_PATH"
echo "[INFO] Running with $NUM_WORKERS parallel workers..."

cat "$H5_LIST_FILE" | xargs -I {} -P "$NUM_WORKERS" \
    bash -c "echo '[RUNNING] Processing {}'; python h5_converter/h5_badile_bondone.py --h5 '{}' --config '$CONFIG_PATH' $DEBUG_FLAG"

echo "[DONE] All files processed."

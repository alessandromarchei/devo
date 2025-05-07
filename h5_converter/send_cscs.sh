#!/bin/bash

# Usage: ./parallel_rsync_single_log.sh folders.txt remote:/dest/path [num_jobs]
# Example: ./parallel_rsync_single_log.sh folders.txt daint:/scratch/amarchei/TartanEvent 8

LIST_FILE="$1"
DEST_BASE="$2"
NUM_JOBS="${3:-4}"
LOG_FILE="send_cscs.log"

if [ -z "$LIST_FILE" ] || [ -z "$DEST_BASE" ]; then
    echo "Usage: $0 <folders_to_move.txt> <remote_base_dest> [num_parallel_jobs]"
    exit 1
fi

if [ ! -f "$LIST_FILE" ]; then
    echo "[ERROR] List file not found: $LIST_FILE"
    exit 1
fi

LOCAL_BASE="/usr/scratch/badile44/amarchei/TartanEvent"
echo "[INFO] Starting parallel rsync with $NUM_JOBS jobs..." | tee "$LOG_FILE"

# Function to run one rsync task and append to shared log
rsync_one() {
    SRC_FOLDER="$1"
    REL_PATH="${SRC_FOLDER#$LOCAL_BASE/}"
    DEST_DIR="$DEST_BASE/$REL_PATH"

    {
        echo ""
        echo "[START] Syncing $SRC_FOLDER → $DEST_DIR"

        if [ ! -d "$SRC_FOLDER" ]; then
            echo "[SKIP] Source does not exist: $SRC_FOLDER"
        else
            rsync -a --info=progress2 "$SRC_FOLDER/" "$DEST_DIR/"
            echo "[DONE] $SRC_FOLDER"
        fi
    } >> "$LOG_FILE" 2>&1
}

export -f rsync_one
export LOCAL_BASE DEST_BASE LOG_FILE

# Run in parallel
cat "$LIST_FILE" | xargs -P "$NUM_JOBS" -I {} bash -c 'rsync_one "$@"' _ "{}"

echo ""
echo "[DONE] All folders synced. Log: $LOG_FILE"

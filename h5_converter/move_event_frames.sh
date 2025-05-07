#!/bin/bash

# Usage: ./parallel_move_event_frames.sh paths.txt [num_parallel_jobs]
# Example: ./parallel_move_event_frames.sh paths.txt 8

input_file="$1"
num_jobs="${2:-8}"  # Default to 4 parallel jobs

if [ -z "$input_file" ] || [ ! -f "$input_file" ]; then
    echo "Usage: $0 paths.txt [num_parallel_jobs]"
    exit 1
fi

echo "[INFO] Starting parallel move with $num_jobs jobs"
echo "[INFO] Reading from: $input_file"

move_one() {
    src="$1"
    dst="${src/badile43/badile44}"

    if [ ! -d "$src" ]; then
        echo "[SKIP] Source missing: $src"
        return
    fi

    if [ -z "$(ls -A "$src")" ]; then
        echo "[SKIP] Source empty: $src"
        return
    fi

    if [ ! -d "$dst" ]; then
        echo "[SKIP] Destination folder does not exist: $dst"
        return
    fi

    echo "[CHECK] Moving from $src to $dst"
    moved=0
    for file in "$src"/*; do
        base=$(basename "$file")
        if [ ! -e "$dst/$base" ]; then
            mv "$file" "$dst/"
            echo "[MOVE] $file → $dst/"
            moved=1
        fi
    done

    if [ -z "$(ls -A "$src")" ]; then
        rmdir "$src"
        echo "[CLEAN] Removed empty source: $src"
    elif [ "$moved" -eq 0 ]; then
        echo "[SKIP] Already complete: $src"
    else
        echo "[INFO] Partial move completed: $src"
    fi
}

export -f move_one

# Run moves in parallel
cat "$input_file" | xargs -P "$num_jobs" -I {} bash -c 'move_one "$@"' _ {}

echo ""
echo "[DONE] Parallel move completed."

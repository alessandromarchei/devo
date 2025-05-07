#!/bin/bash

# Usage: ./fix_nested_event_frames_from_list.sh paths.txt

input_file="$1"

if [ -z "$input_file" ] || [ ! -f "$input_file" ]; then
    echo "[ERROR] Provide a valid text file with paths to event_frames folders."
    exit 1
fi

echo "[INFO] Processing folders from list: $input_file"
echo

while IFS= read -r parent_path; do
    nested_path="$parent_path/event_frames"

    # Check if the nested folder exists
    if [ -d "$nested_path" ]; then
        echo "[FIXING] Found nested: $nested_path → flattening into $parent_path"

        shopt -s dotglob  # Include hidden files (e.g., .npy, .meta)
        for file in "$nested_path"/*; do
            mv "$file" "$parent_path/"
        done
        shopt -u dotglob

        rmdir "$nested_path" && echo "[CLEAN] Removed nested folder: $nested_path"
        echo
    else
        echo "[OK] No nested event_frames in: $parent_path"
    fi
done < "$input_file"

echo "[DONE] All listed folders checked and fixed where necessary."

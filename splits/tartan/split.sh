#!/bin/bash

# Number of smallest entries to keep
N=100

# Resolve the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Input and output files (relative to the script's directory)
INPUT_FILE="${SCRIPT_DIR}/events_size.txt"
OUTPUT_FILE="${SCRIPT_DIR}/train_split.txt"

# Make sure input exists
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: File '$INPUT_FILE' not found in script directory!"
    exit 1
fi

# Clear output file
> "$OUTPUT_FILE"

# Process the last N lines (smallest files)
tail -n "$N" "$INPUT_FILE" | awk '{print $NF}' | while read -r path; do
    # Remove leading slash and extension
    stripped_path="${path#/}"                    # remove first /
    no_ext="${stripped_path%/events.h5}"         # remove ending /events.h5

    # Get first part of path (e.g., soulcity)
    first_part="${no_ext%%/*}"

    # Build final path
    final_path="${first_part}/${no_ext}"

    echo "$final_path" >> "$OUTPUT_FILE"
done

echo "Written to $OUTPUT_FILE"

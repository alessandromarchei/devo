#!/bin/bash

# Base path to search in
BASE_PATH="/usr/scratch/badile43/amarchei/TartanEvent/"

# Step interval
STEP=33333333.333333332

echo "Filling empty timestamps.txt files in: $BASE_PATH"
echo

# Find and process each empty timestamps.txt file
while IFS= read -r -d '' file; do
  if [ ! -s "$file" ]; then
    # Parent dir and image_left folder
    parent_dir=$(dirname "$file")
    image_left_dir="$parent_dir/image_left"

    if [ -d "$image_left_dir" ]; then
      num_images=$(find "$image_left_dir" -type f | wc -l)
      num_lines=$((num_images - 1))

      if [ "$num_lines" -le 0 ]; then
        echo "Skipping $file — not enough images in image_left."
        continue
      fi

      echo "Writing $num_lines lines to: $file"

      # Generate and write the lines
      > "$file"  # Clear the file (it's empty already, but to be sure)
      for ((i = 0; i < num_lines; i++)); do
        value=$(echo "$STEP * $i" | bc -l)
        printf "%.15g\n" "$value" >> "$file"
      done
    else
      echo "WARNING: No image_left folder found for: $file"
    fi
  fi
done < <(find "$BASE_PATH" -type f -name "timestamps.txt" -print0)

echo
echo "Done. All empty timestamps.txt files have been filled."

total=0
while IFS= read -r path; do
    size_bytes=$(du -sb "$path" | cut -f1)
    size_mb=$(echo "$size_bytes / 1024 / 1024" | bc)
    echo "$path -> ${size_mb} MB"
    total=$((total + size_bytes))
done < h5_converter/event_frames_folders_badile44.txt

total_mb=$(echo "$total / 1024 / 1024" | bc)
echo "Total size: ${total_mb} MB"

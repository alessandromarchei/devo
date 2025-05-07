#!/bin/bash

# Usage: ./compare_lines.sh file1.txt file2.txt

# === Safety checks ===
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 file1.txt file2.txt"
    exit 1
fi

file1="$1"
file2="$2"

if [ ! -f "$file1" ]; then
    echo "File not found: $file1"
    exit 2
fi

if [ ! -f "$file2" ]; then
    echo "File not found: $file2"
    exit 3
fi

# === Count lines ===
lines1=$(wc -l < "$file1")
lines2=$(wc -l < "$file2")
max_lines=$(( lines1 > lines2 ? lines1 : lines2 ))

echo "=== File comparison report ==="
echo "File 1: $file1 ($lines1 lines)"
echo "File 2: $file2 ($lines2 lines)"
echo "Comparing up to $max_lines lines..."
echo ""

# === Initialize counters ===
same_lines=0
diff_lines=0
extra_lines=0

# === Compare lines ===
for i in $(seq 1 "$max_lines"); do
    line1=$(sed -n "${i}p" "$file1")
    line2=$(sed -n "${i}p" "$file2")

    if [ -z "$line1" ] && [ -n "$line2" ]; then
        echo "[$i] File1 missing | File2: $line2"
        ((extra_lines++))
    elif [ -n "$line1" ] && [ -z "$line2" ]; then
        echo "[$i] File2 missing | File1: $line1"
        ((extra_lines++))
    elif [ "$line1" == "$line2" ]; then
        echo "[$i] SAME     | $line1"
        ((same_lines++))
    else
        echo "[$i] DIFFERS  | File1: $line1"
        echo "              | File2: $line2"
        ((diff_lines++))
    fi
done

# === Summary ===
echo ""
echo "=== Summary ==="
echo "Lines in $file1: $lines1"
echo "Lines in $file2: $lines2"
echo "Same lines     : $same_lines"
echo "Differing lines: $diff_lines"
echo "Extra lines    : $extra_lines"

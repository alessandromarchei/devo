#!/bin/bash

TARGET_DIR="/scratch/amarchei/fpv"
mkdir -p "$TARGET_DIR"

download_file() {
    local url="$1"
    local filename=$(basename "$url")
    local filepath="$TARGET_DIR/$filename"
    local unzip_dir="$TARGET_DIR/${filename%.zip}"

    if [ -f "$filepath" ]; then
        echo "⚠️  Skipping download (already exists): $filename"
    else
        echo "🔽 Downloading: $filename"
        wget -q -P "$TARGET_DIR" "$url"
        if [ $? -ne 0 ]; then
            echo "❌ Not found: $url"
            rm -f "$filepath"  # Remove any partial file
            return
        else
            echo "✅ Downloaded: $filepath"
        fi
    fi

    # Unzip if not already extracted
    if [ -d "$unzip_dir" ]; then
        echo "📦 Skipping unzip (already exists): $unzip_dir"
    else
        echo "📂 Unzipping: $filename → $unzip_dir"
        mkdir -p "$unzip_dir"
        unzip -q "$filepath" -d "$unzip_dir"
    fi
}

# ----------- 1. indoor_forward_1..12 ------------
echo ""
echo "--- Downloading indoor_forward_X_davis_with_gt.zip ---"
for i in $(seq 1 15); do
    url="http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/indoor_forward_${i}_davis_with_gt.zip"
    download_file "$url"
done

# ----------- 2. indoor_45_1..16 ------------
echo ""
echo "--- Downloading indoor_45_X_davis_with_gt.zip ---"
for i in $(seq 1 16); do
    url="http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/indoor_45_${i}_davis_with_gt.zip"
    download_file "$url"
done

# ----------- 3. outdoor_forward_1..10 ------------
echo ""
echo "--- Downloading outdoor_forward_X_davis_with_gt.zip ---"
for i in $(seq 1 15); do
    url="http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/outdoor_forward_${i}_davis_with_gt.zip"
    download_file "$url"
done

# ----------- 4. NO GT ------------
echo ""
echo "--- Downloading indoor_forward_X_davis.zip ---"
for i in $(seq 1 16); do
    url="http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/indoor_forward_${i}_davis.zip"
    download_file "$url"
done

echo ""
echo "--- Downloading indoor_45_X_davis.zip ---"
for i in $(seq 1 16); do
    url="http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/indoor_45_${i}_davis.zip"
    download_file "$url"
done

echo ""
echo "--- Downloading outdoor_forward_X_davis.zip ---"
for i in $(seq 1 16); do
    url="http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/outdoor_forward_${i}_davis.zip"
    download_file "$url"
done

echo ""
echo "✅ All downloads and extractions completed successfully!"

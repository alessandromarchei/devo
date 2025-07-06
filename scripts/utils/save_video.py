import os
import cv2
import argparse
from pathlib import Path
import numpy as np
import re

def extract_frame_number(filename):
    match = re.search(r"(\d+)", Path(filename).stem)
    return int(match.group(1)) if match else None

def map_images_by_frame(folder, label):
    image_map = {}
    print(f"\n[INFO] Scanning folder '{label}': {folder}")
    for p in Path(folder).rglob("*.png"):
        frame_num = extract_frame_number(p.name)
        if frame_num is not None:
            print(f"  - Found {p.name} with frame {frame_num}")
            image_map[frame_num] = str(p)
        else:
            print(f"  [WARNING] Skipped file (no number found): {p.name}")
    print(f"[INFO] Total images in '{label}': {len(image_map)}\n")
    return image_map

def debug_folder_contents(folder_path, label):
    print(f"\n[DEBUG] Listing files in '{label}' folder: {folder_path}")
    if not os.path.exists(folder_path):
        print(f"[ERROR] Folder '{folder_path}' does not exist.")
        return
    files = list(Path(folder_path).iterdir())
    if not files:
        print(f"[WARNING] Folder '{folder_path}' is empty.")
    else:
        for f in files:
            print(f"  - {f.name}")
    print()

def main(folder_gray, folder_voxel, fps):
    debug_folder_contents(folder_gray, "Grayscale")
    debug_folder_contents(folder_voxel, "Voxel")

    gray_map = map_images_by_frame(folder_gray, "Grayscale")
    voxel_map = map_images_by_frame(folder_voxel, "Voxel")

    common_frames = sorted(set(gray_map.keys()) & set(voxel_map.keys()))
    print(f"[INFO] Found {len(common_frames)} common frames: {common_frames}\n")

    if not common_frames:
        print("[ERROR] No matching frame numbers found between folders.")
        return

    sample_gray = cv2.imread(gray_map[common_frames[0]], cv2.IMREAD_GRAYSCALE)
    sample_voxel = cv2.imread(voxel_map[common_frames[0]], cv2.IMREAD_COLOR)
    height, width = sample_gray.shape
    combined_size = (width * 2, height)

    # Output video is now COLOR
    out = cv2.VideoWriter('combined_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, combined_size)

    for idx, frame_num in enumerate(common_frames):
        gray_path = gray_map[frame_num]
        voxel_path = voxel_map[frame_num]
        print(f"[FRAME {idx}] Matching frame number {frame_num}:\n"
              f"    Gray  -> {Path(gray_path).name}\n"
              f"    Voxel -> {Path(voxel_path).name}")

        gray_img = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
        voxel_img = cv2.imread(voxel_path, cv2.IMREAD_COLOR)

        if gray_img is None or voxel_img is None:
            print(f"[WARNING] Failed to load one of the images for frame {frame_num}. Skipping.")
            continue

        gray_bgr = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

        if gray_bgr.shape != voxel_img.shape:
            print(f"[INFO] Resizing voxel image to match grayscale for frame {frame_num}")
            voxel_img = cv2.resize(voxel_img, (gray_bgr.shape[1], gray_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

        combined = np.hstack((gray_bgr, voxel_img))

        time_sec = idx / fps
        text = f"Frame: {frame_num} | Time: {time_sec:.2f}s"
        cv2.putText(combined, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)

        out.write(combined)

    out.release()
    print(f"\n[INFO] Video saved as 'combined_video.mp4'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("gray_folder", type=str, help="Path to folder with grayscale PNGs")
    parser.add_argument("voxel_folder", type=str, help="Path to folder with voxel grid PNGs (RGB)")
    parser.add_argument("--fps", type=float, required=True, help="Frames per second for the output video")
    args = parser.parse_args()

    main(args.gray_folder, args.voxel_folder, args.fps)
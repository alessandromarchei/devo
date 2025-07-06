import os
import cv2
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

import re

def extract_frame_number(filename):
    match = re.search(r"(\d+)", Path(filename).stem)
    return int(match.group(1)) if match else None

def map_images_by_frame(folder, label):
    image_map = {}
    for p in Path(folder).rglob("*.png"):
        frame_num = extract_frame_number(p.name)
        if frame_num is not None:
            image_map[frame_num] = str(p)
    return image_map

def debug_folder_contents(folder_path, label):
    if not os.path.exists(folder_path):
        print(f"[ERROR] Folder '{folder_path}' does not exist.")
        return
    files = list(Path(folder_path).iterdir())
    if not files:
        print(f"[WARNING] Folder '{folder_path}' is empty.")
    else:
        for f in files:
            print(f"  - {f.name}")

def load_trajectory(file_path):
    timestamps, positions = [], []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            tokens = line.strip().split()
            if len(tokens) == 8:
                timestamps.append(float(tokens[0]))
                pos = list(map(float, tokens[1:4]))  # tx, ty, tz
                positions.append(pos)
    return np.array(timestamps), np.array(positions)
def create_trajectory_image(est_xyz, gt_xyz, width=300, height=300):
    fig = plt.figure(figsize=(3, 3), dpi=100)
    ax = fig.add_subplot(111)
    if len(gt_xyz) > 0:
        #plotting zy trajectory
        ax.plot(gt_xyz[:, 2], gt_xyz[:, 1], label="GT", color="green")
    if len(est_xyz) > 0:
        ax.plot(est_xyz[:, 2], est_xyz[:, 1], label="Pred", color="blue")
    ax.set_title("Trajectory XY", fontsize=8)
    ax.set_xlabel("X", fontsize=6)
    ax.set_ylabel("Y", fontsize=6)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=6, loc='upper right')
    ax.grid(True)
    fig.tight_layout(pad=0.5)
    fig.canvas.draw()

    # Convert canvas to RGB image
    img = np.frombuffer(fig.canvas.renderer.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img = img[..., :3]  # Remove alpha

    plt.close(fig)
    return cv2.resize(img, (width, height))


def main(folder_gray, folder_voxel, pred_file, gt_file, fps):
    debug_folder_contents(folder_gray, "Grayscale")
    debug_folder_contents(folder_voxel, "Voxel")

    gray_map = map_images_by_frame(folder_gray, "Grayscale")
    voxel_map = map_images_by_frame(folder_voxel, "Voxel")

    common_frames = sorted(set(gray_map.keys()) & set(voxel_map.keys()))
    if not common_frames:
        print("[ERROR] No matching frame numbers found between folders.")
        return

    sample_gray = cv2.imread(gray_map[common_frames[0]], cv2.IMREAD_GRAYSCALE)
    sample_voxel = cv2.imread(voxel_map[common_frames[0]], cv2.IMREAD_COLOR)
    height, width = sample_gray.shape

    _, est_xyz = load_trajectory(pred_file)
    _, gt_xyz = load_trajectory(gt_file)

    traj_width = width // 2
    combined_size = (width * 2 + traj_width, height)
    out = cv2.VideoWriter('combined_video_with_traj.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, combined_size)

    for idx, frame_num in enumerate(common_frames):
        print(f"[INFO] Processing frame {idx+1}/{len(common_frames)}: {frame_num}")
        
        gray_img = cv2.imread(gray_map[frame_num], cv2.IMREAD_GRAYSCALE)
        voxel_img = cv2.imread(voxel_map[frame_num], cv2.IMREAD_COLOR)
        if gray_img is None or voxel_img is None:
            continue

        gray_bgr = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        if voxel_img.shape != gray_bgr.shape:
            voxel_img = cv2.resize(voxel_img, (gray_bgr.shape[1], gray_bgr.shape[0]))

        # Update trajectory panel
        est_part = est_xyz[:idx+1] if idx < len(est_xyz) else est_xyz
        gt_part = gt_xyz[:idx+1] if idx < len(gt_xyz) else gt_xyz
        traj_img = create_trajectory_image(est_part, gt_part, width=traj_width, height=gray_bgr.shape[0])

        combined = np.hstack((gray_bgr, voxel_img, traj_img))

        time_sec = idx / fps
        text = f"Frame: {frame_num} | Time: {time_sec:.2f}s"
        cv2.putText(combined, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)

        out.write(combined)

    out.release()
    print("[INFO] Video saved as 'combined_video_with_traj.mp4'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("gray_folder", type=str, help="Path to folder with grayscale PNGs")
    parser.add_argument("voxel_folder", type=str, help="Path to folder with voxel grid PNGs (RGB)")
    parser.add_argument("--pred", type=str, required=True, help="Path to predicted trajectory file (.txt)")
    parser.add_argument("--gt", type=str, required=True, help="Path to ground truth trajectory file (.txt)")
    parser.add_argument("--fps", type=float, required=True, help="Frames per second for the output video")
    args = parser.parse_args()

    main(args.gray_folder, args.voxel_folder, args.pred, args.gt, args.fps)

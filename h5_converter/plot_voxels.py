import os
import argparse
import numpy as np
import torch
import time
import cv2

def visualize_voxel_cv(*voxel_in, EPS=1e-3, window_name="Voxel View", max_size=900, wait=100):
    colors = {
        -1: [0, 0, 255],     # red
         0: [255, 255, 255], # white
         1: [255, 0, 0]      # blue
    }

    all_images = []

    for i, vox in enumerate(voxel_in):
        if isinstance(vox, np.ndarray):
            voxel = torch.from_numpy(vox)
        else:
            voxel = vox
        if voxel.device != 'cpu':
            voxel = voxel.detach().cpu()

        voxel = voxel.clone().numpy()
        bins, H, W = voxel.shape
        out_images = []

        for b in range(bins):
            v = voxel[b]
            v[np.bitwise_and(v < EPS, v > 0)] = 0
            v[np.bitwise_and(v > -EPS, v < 0)] = 0
            v[v < 0] = -1
            v[v > 0] = 1

            img = np.zeros((H, W, 3), dtype=np.uint8)
            for val, color in colors.items():
                img[v == val] = color

            out_images.append(img)

        combined = np.hstack(out_images)
        all_images.append(combined)

    final_image = np.vstack(all_images)
    h, w = final_image.shape[:2]
    scale = min(max_size / h, max_size / w, 1.0)
    if scale < 1.0:
        final_image = cv2.resize(final_image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)

    cv2.imshow(window_name, final_image)
    key = cv2.waitKey(wait)
    return key

def main(folder, delay_ms):
    files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".npy")])
    if not files:
        print(f"[ERROR] No .npy files found in {folder}")
        return

    print(f"[INFO] Playing {len(files)} event frame files from {folder}")
    for i, file in enumerate(files):
        data = np.load(file)
        key = visualize_voxel_cv(data, wait=delay_ms)
        if key == 27:  # ESC
            break

    print("[DONE]")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="Path to folder containing .npy event frame files")
    parser.add_argument("--delay", type=int, default=100, help="Delay between frames in milliseconds")
    args = parser.parse_args()

    main(args.folder, args.delay)

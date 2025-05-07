import os
import numpy as np
import argparse
from tqdm import tqdm

def inspect_event_folder(folder):
    npy_files = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".npy")
    ])

    if not npy_files:
        print(f"[ERROR] No .npy files found in: {folder}")
        return

    global_max = 0
    per_file_max = []
    per_file_min = []
    print(f"[INFO] Inspecting {len(npy_files)} .npy files in: {folder}\n")

    for fpath in tqdm(npy_files, desc="Analyzing frames"):
        data = np.load(fpath)
        local_max = np.max(np.abs(data))
        local_min = np.min(data)
        per_file_max.append(local_max)
        per_file_min.append(local_min)
        global_max = max(global_max, local_max)

    print("\n[SUMMARY]")
    print(f"Global max absolute value across all files: {global_max}")
    print(f"Max per file (sample): {per_file_max[:10]} ...")
    print(f"Min per file (sample): {per_file_min[:10]} ...")

    return per_file_max, global_max

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True, help="Path to folder containing .npy event stacks")
    args = parser.parse_args()

    inspect_event_folder(args.folder)

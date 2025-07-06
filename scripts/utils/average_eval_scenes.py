import re
from collections import defaultdict
import sys

def parse_file(file_path):
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("Scene")]

    scene_data = defaultdict(list)

    for line in lines:
        parts = re.split(r'\s+', line)
        if len(parts) >= 4:
            scene = parts[0]
            try:
                ate = float(parts[1])
                r_rmse = float(parts[2])
                mpe = float(parts[3])
                scene_data[scene].append((ate, r_rmse, mpe))
            except ValueError:
                print(f"Warning: Skipped malformed line: {line}")
                continue

    return scene_data

def compute_and_print_averages(scene_data):
    print(f"{'Scene':<35} {'Avg_ATE[cm]':>12} {'Avg_R_rmse[deg]':>16} {'Avg_MPE[%/m]':>14}")
    all_avg_ate, all_avg_r, all_avg_mpe = [], [], []

    for scene, values in scene_data.items():
        ate_avg = sum(v[0] for v in values) / len(values)
        r_avg = sum(v[1] for v in values) / len(values)
        mpe_avg = sum(v[2] for v in values) / len(values)
        print(f"{scene:<35} {ate_avg:12.3f} {r_avg:16.3f} {mpe_avg:14.3f}")
        all_avg_ate.append(ate_avg)
        all_avg_r.append(r_avg)
        all_avg_mpe.append(mpe_avg)

    # Global average across all scenes
    print("\n" + "-" * 80)
    print(f"{'Overall Mean of Scene Averages':<35} "
          f"{sum(all_avg_ate) / len(all_avg_ate):12.3f} "
          f"{sum(all_avg_r) / len(all_avg_r):16.3f} "
          f"{sum(all_avg_mpe) / len(all_avg_mpe):14.3f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python compute_averages.py <input_file.txt>")
        sys.exit(1)

    file_path = sys.argv[1]
    scene_data = parse_file(file_path)
    compute_and_print_averages(scene_data)

import sys
import numpy as np
from collections import defaultdict

def parse_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    print("==== File Content ====")
    print("".join(lines))
    print("======================\n")

    data = defaultdict(lambda: {'ATE': [], 'R_rmse': [], 'MPE': []})

    for line in lines:
        if line.strip().startswith("Scene") or not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) >= 4:
            scene = parts[0]
            try:
                ATE = float(parts[1])
                R_rmse = float(parts[2])
                MPE = float(parts[3])
                data[scene]['ATE'].append(ATE)
                data[scene]['R_rmse'].append(R_rmse)
                data[scene]['MPE'].append(MPE)
            except ValueError:
                continue

    return data

def compute_medians(data):
    medians = {}
    for scene, metrics in data.items():
        medians[scene] = {
            'ATE': np.median(metrics['ATE']),
            'R_rmse': np.median(metrics['R_rmse']),
            'MPE': np.median(metrics['MPE']),
        }
    return medians

def compute_average_across_scenes(medians):
    avg = {}
    for key in ['ATE', 'R_rmse', 'MPE']:
        values = [scene_data[key] for scene_data in medians.values()]
        avg[key] = np.mean(values)
    return avg

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_metrics.py <input_file.txt>")
        sys.exit(1)

    file_path = sys.argv[1]
    data = parse_file(file_path)
    medians = compute_medians(data)

    print("== Median per Scene ==")
    for scene, values in medians.items():
        print(f"{scene}: ATE = {values['ATE']:.3f} cm, R_rmse = {values['R_rmse']:.3f} deg, MPE = {values['MPE']:.3f} %/m")

    avg_medians = compute_average_across_scenes(medians)
    print("\n== Average of Medians Across All Scenes ==")
    print(f"ATE = {avg_medians['ATE']:.3f} cm, R_rmse = {avg_medians['R_rmse']:.3f} deg, MPE = {avg_medians['MPE']:.3f} %/m")

if __name__ == "__main__":
    main()

import sys
import statistics
from collections import defaultdict

def is_float(s):
    try:
        float(s)
        return True
    except:
        return False

def parse_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    scene_ates = defaultdict(list)

    for line in lines:
        line = line.strip()
        if not line or line.startswith("Scene") or line.endswith("\\\\") is False:
            continue

        parts = line.replace("\\\\", "").split()
        if len(parts) < 4:
            continue

        scene, ate = parts[0], parts[1]
        if is_float(ate):
            scene_ates[scene].append(float(ate))

    return scene_ates

def main(file_path):
    print(f"📂 Reading: {file_path}\n")
    scene_ates = parse_file(file_path)

    if not scene_ates:
        print("⚠️ No valid ATE data found.")
        return

    medians = {}

    print("🔍 Median ATE per scene:")
    for scene, ates in scene_ates.items():
        median_val = statistics.median(ates)
        medians[scene] = median_val
        print(f" - {scene}: median ATE = {median_val:.3f} cm from {len(ates)} entries")

    global_avg = statistics.mean(medians.values())
    print(f"\n📊 Final ATE (mean of medians): {global_avg:.3f} cm")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python compute_scene_ate.py results.txt")
        sys.exit(1)

    main(sys.argv[1])

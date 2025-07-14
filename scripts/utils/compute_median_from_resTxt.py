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
        if not line or line.startswith("Scene") or not line.endswith("\\\\"):
            continue

        parts = line.replace("\\\\", "").split()
        if len(parts) < 4:
            continue

        scene, ate = parts[0], parts[1]
        if is_float(ate):
            scene_ates[scene].append(float(ate))

    return scene_ates

def convert_for_locale(val):
    # Change dot to comma for European locale
    return f"{val:.3f}".replace(".", ",")

def main(file_path):
    print(f"📂 Reading: {file_path}\n")
    scene_ates = parse_file(file_path)

    if not scene_ates:
        print("⚠️ No valid ATE data found.")
        return

    medians = {}
    bests = {}

    print("🔍 ATE per scene:")
    for scene, ates in scene_ates.items():
        if len(ates) < 5:
            print(f"⚠️ Scene '{scene}' has less than 5 entries ({len(ates)} runs), using all available for best.")
        median_val = statistics.median(ates)
        medians[scene] = median_val

        best_val = min(ates[:5]) if len(ates) >= 5 else min(ates)
        bests[scene] = best_val

        print(f" - {scene}: median ATE = {median_val:.3f} cm, best ATE (out of 5) = {best_val:.3f} cm from {len(ates)} entries")

    global_mean_median = statistics.mean(medians.values())
    global_mean_best = statistics.mean(bests.values())

    print(f"\n📊 Final ATE summary:")
    print(f" • Mean of medians: {global_mean_median:.3f} cm")
    print(f" • Mean of best out of 5: {global_mean_best:.3f} cm")

    # ✅ Print median column with comma
    print("\n📄 Copy and paste the following *median ATEs* column into Calc:\n")
    for scene in sorted(medians.keys()):
        print(convert_for_locale(medians[scene]))

    # ✅ Print best column with comma
    print("\n📄 Copy and paste the following *best ATEs* column into Calc:\n")
    for scene in sorted(bests.keys()):
        print(convert_for_locale(bests[scene]))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python compute_scene_ate.py results.txt")
        sys.exit(1)

    main(sys.argv[1])

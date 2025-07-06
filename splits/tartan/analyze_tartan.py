import os
import sys
import h5py

def get_folder_size(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            try:
                fp = os.path.join(dirpath, f)
                total += os.path.getsize(fp)
            except FileNotFoundError:
                continue
    return total

def analyze_scene(scene_path):
    print(f"\nAnalyzing scene: {scene_path}")

    # Get total size of the folder
    folder_size_bytes = get_folder_size(scene_path)
    folder_size_gb = folder_size_bytes / (1024**3)
    print(f"  - Folder size: {folder_size_gb:.2f} GB")

    # Locate events.h5
    events_file = os.path.join(scene_path, "events.h5")
    if not os.path.exists(events_file):
        print("  ❌ events.h5 not found.")
        return

    try:
        with h5py.File(events_file, 'r') as f:
            num_events = len(f["events"]["x"])
            print(f"  - Number of events: {num_events}")
    except Exception as e:
        print(f"  ❌ Failed to open events.h5: {e}")
        return

    # Locate timestamps.txt
    timestamps_file = os.path.join(scene_path, "timestamps.txt")
    if not os.path.exists(timestamps_file):
        print("  ❌ timestamps.txt not found.")
        return

    try:
        with open(timestamps_file, 'r') as f:
            lines = f.readlines()
            num_lines = len(lines)
            if num_lines < 2:
                print("  ⚠️ Not enough timestamps to compute duration.")
                return
            duration_seconds = (num_lines - 1) * 0.03
            num_frames = num_lines - 1
            print(f"  - Number of timestamps: {num_lines}")
            print(f"  - Scene duration: {duration_seconds:.2f} seconds")
            print(f"  - Number of frames: {num_frames}")
    except Exception as e:
        print(f"  ❌ Failed to read timestamps.txt: {e}")
        return

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_tartanevent.py paths.txt")
        sys.exit(1)

    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        sys.exit(1)

    with open(input_file, 'r') as f:
        scene_paths = [line.strip() for line in f if line.strip()]

    print(f"Found {len(scene_paths)} scene paths.")

    for path in scene_paths:
        analyze_scene(path)

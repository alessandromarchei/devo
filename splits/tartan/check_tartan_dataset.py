#!/usr/bin/env python3
import argparse
import os
from multiprocessing import Pool, cpu_count

def extract_index(fname):
    try:
        parts = fname.replace("ev", "").replace(".npy", "").split("_")
        return int(parts[0]), int(parts[1])
    except Exception:
        return None

def find_missing_indices(files):
    """Find all missing indices and return count + example filenames."""
    indices = sorted(filter(None, (extract_index(f) for f in files)))
    missing_pairs = []
    for (a1, a2), (b1, b2) in zip(indices, indices[1:]):
        if b1 != a1 + 1 or b2 != a2 + 1:
            # There is a gap — fill it
            expected_a1 = a1 + 1
            expected_a2 = a2 + 1
            while expected_a1 < b1 or expected_a2 < b2:
                missing_pairs.append((expected_a1, expected_a2))
                expected_a1 += 1
                expected_a2 += 1
    # Convert to filenames
    missing_files = [f"ev{a1:04d}_{a2:04d}.npy" for (a1, a2) in missing_pairs]
    return len(missing_pairs), missing_files

def process_line(event_voxel_path):
    event_voxel_path = event_voxel_path.strip()
    if not event_voxel_path:
        return None

    # Event voxel folder (badile44)
    if not os.path.isdir(event_voxel_path):
        return (event_voxel_path, None, None, None, 0, [], "[ERROR] Event voxel path not found")

    event_files = [f for f in os.listdir(event_voxel_path) if f.endswith(".npy")]
    num_event = len(event_files)
    missing_count, missing_examples = find_missing_indices(event_files)

    # Depth left folder (badile43)
    depth_left_path = event_voxel_path.replace("badile44", "badile43").replace("event_voxel", "depth_left")
    if not os.path.isdir(depth_left_path):
        return (event_voxel_path, num_event, None, None, missing_count, missing_examples, "[ERROR] Depth left path not found")
    num_depth = len([f for f in os.listdir(depth_left_path) if f.endswith(".npy")])

    # Timestamps.txt (badile43)
    timestamps_path = os.path.join(os.path.dirname(depth_left_path), "timestamps.txt")
    if not os.path.isfile(timestamps_path):
        return (event_voxel_path, num_event, num_depth, None, missing_count, missing_examples, "[ERROR] timestamps.txt not found")
    with open(timestamps_path, "r") as f:
        num_timestamps = sum(1 for _ in f if _.strip())

    # Check mismatch condition
    if num_event != num_depth - 1  or missing_count > 0:
        return (event_voxel_path, num_event, num_depth, num_timestamps, missing_count, missing_examples, "MISMATCH")

    return None  # No mismatch

def main():
    parser = argparse.ArgumentParser(description="Check npy/timestamps consistency.")
    parser.add_argument("txt_file", help="Path to txt file containing event_voxel paths")
    parser.add_argument("--log", default="mismatch_log.txt", help="Output log file")
    parser.add_argument("--max-missing-examples", type=int, default=10, help="Number of missing filenames to show per folder")
    args = parser.parse_args()

    if not os.path.isfile(args.txt_file):
        print(f"[ERROR] File not found: {args.txt_file}")
        return

    with open(args.txt_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_line, lines)

    mismatches = [r for r in results if r is not None]

    # Write log
    with open(args.log, "w") as log_file:
        for path, num_event, num_depth, num_timestamps, missing_count, missing_examples, status in mismatches:
            log_file.write(
                f"{status} | {path}\n"
                f"  Event voxel: {num_event}\n"
                f"  Depth left : {num_depth}\n"
                f"  Timestamps : {num_timestamps}\n"
                f"  Missing index gaps: {missing_count}\n"
            )
            if missing_count > 0:
                shown = missing_examples[:args.max_missing_examples]
                log_file.write(f"  Missing files (first {len(shown)}): {shown}\n")
            log_file.write("\n")

    print(f"[INFO] Found {len(mismatches)} mismatches. Log saved to {args.log}")

if __name__ == "__main__":
    main()

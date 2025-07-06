import h5py
import argparse
import numpy as np
import hdf5plugin
import warnings

def print_structure(name, obj):
    """Prints the structure of the HDF5 file (datasets and groups)."""
    obj_type = "Dataset" if isinstance(obj, h5py.Dataset) else "Group"
    print(f"{obj_type}: {name}")
    if isinstance(obj, h5py.Dataset):
        print(f"  Shape: {obj.shape}, Type: {obj.dtype}")

def inspect_hdf5(file_path):
    """Opens and inspects an HDF5 file and computes Δt statistics."""
    try:
        with h5py.File(file_path, "r") as hdf5_file:
            print(f"\nInspecting HDF5 file: {file_path}")
            print("=" * 50)
            hdf5_file.visititems(print_structure)

            topic = "davis/left/image_raw_ts"
            if topic in hdf5_file:
                ts_dataset = hdf5_file[topic]
                total = ts_dataset.shape[0]

                if total < 2:
                    print("[ERROR] Not enough timestamps to compute deltas.")
                    return

                last_n = min(100, total)
                timestamps = ts_dataset[-last_n:]

                deltas = np.diff(timestamps)
                avg_dt = np.mean(deltas)

                print(f"\n[INFO] Computed Δt for the last {last_n} timestamps in '{topic}':")
                print(f"  Average Δt = {avg_dt:.12f} seconds")
                print(f"  Min Δt     = {np.min(deltas):.12f} seconds")
                print(f"  Max Δt     = {np.max(deltas):.12f} seconds")
                print(f"  Std Δt     = {np.std(deltas):.12f} seconds")
            else:
                print(f"\nDataset '{topic}' not found.")

    except Exception as e:
        print(f"Error opening file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect an HDF5 file and compute Δt of timestamps.")
    parser.add_argument("file", type=str, help="Path to the HDF5 file")
    args = parser.parse_args()
    
    inspect_hdf5(args.file)

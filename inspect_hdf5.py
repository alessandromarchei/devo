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
    """Opens and inspects an HDF5 file."""
    try:
        with h5py.File(file_path, "r") as hdf5_file:
            print(f"\nInspecting HDF5 file: {file_path}")
            print("=" * 50)
            hdf5_file.visititems(print_structure)

            # Attempt to read the last 2 timestamps from 'events/t'
            if "events/t" in hdf5_file:
                print("\nPreview of the last 2 timestamps in 'events/t':")
                timestamps = hdf5_file["events/t"]
                if timestamps.shape[0] >= 2:
                    print(f"  {timestamps[0:2]}")
                else:
                    print("  Dataset contains fewer than 2 timestamps.")
            else:
                print("\nDataset 'events/t' not found.")

    except Exception as e:
        print(f"Error opening file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect an HDF5 file.")
    parser.add_argument("file", type=str, help="Path to the HDF5 file")
    args = parser.parse_args()
    
    inspect_hdf5(args.file)

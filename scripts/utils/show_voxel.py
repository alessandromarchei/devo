from utils.viz_utils import visualize_voxel
import numpy as np
from pathlib import Path
import torch
import argparse


# Parse command-line argument
parser = argparse.ArgumentParser(description="Investigate a voxel File")
parser.add_argument("voxel", type=str, help="Path to the voxel file")
args = parser.parse_args()


# Load the voxel file
voxel_path = Path(args.voxel)
if not voxel_path.exists():
    raise FileNotFoundError(f"Voxel file {voxel_path} does not exist.")

# Load the voxel data
voxel_data = np.load(voxel_path)
if not isinstance(voxel_data, np.ndarray):
    raise TypeError(f"Voxel data is not a numpy array. Found: {type(voxel_data)}")

#now print the voxel on screen
print(f"Voxel data shape: {voxel_data.shape}")
print(f"Voxel data type: {voxel_data.dtype}")


voxel_data = torch.from_numpy(voxel_data)


visualize_voxel(voxel_data)
import os
import argparse
from pathlib import Path

def replicate_structure(src_root, dst_root):
    src_root = Path(src_root).resolve()
    dst_root = Path(dst_root).resolve()

    for dirpath, dirnames, filenames in os.walk(src_root):
        # Compute relative path from source root
        rel_path = Path(dirpath).relative_to(src_root)
        # Create corresponding directory at destination
        target_dir = dst_root / rel_path
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created: {target_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replicate directory structure without copying files.")
    parser.add_argument("src", type=str, help="Source root directory")
    parser.add_argument("dst", type=str, help="Destination base path")

    args = parser.parse_args()

    # Convert to Path before using /
    src = Path(args.src)
    dst = Path(args.dst) / src.name  # dst will contain the root folder name from src

    replicate_structure(src, dst)

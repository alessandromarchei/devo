import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from scipy.ndimage import zoom
import matplotlib.colors as mcolors
import numpy.ma as ma


def merge_and_plot(json_file, output_dir=None):
    with open(json_file, 'r') as f:
        voxel_data = json.load(f)

    scene_coverage = {}
    scene_heatmaps = {}
    dataset_counts = {}

    max_value = 0
    max_info = ("", -1, -1)  # (scene, frame_index, value)

    for full_path, counts in voxel_data.items():
        parts = full_path.strip("/").split("/")
        dataset_root = parts[-3]
        scene_id = parts[-1]
        dataset_name = "/".join(parts[-4:])

        if dataset_root not in dataset_counts:
            dataset_counts[dataset_root] = 0
        dataset_counts[dataset_root] += 1

        counts_array = np.array(counts)
        nonzero_percentage = (counts_array > 0).sum() / len(counts_array) * 100

        local_max = counts_array.max()
        if local_max > max_value:
            max_value = local_max
            max_info = (dataset_name, int(np.argmax(counts_array)), int(local_max))

        scene_coverage[dataset_name] = nonzero_percentage
        scene_heatmaps[dataset_name] = counts_array

        # === Compute total dataset coverage ===
    total_voxels = sum(len(arr) for arr in scene_heatmaps.values())
    total_covered_voxels = sum((arr > 0).sum() for arr in scene_heatmaps.values())
    total_coverage_pct = (total_covered_voxels / total_voxels) * 100

    print(f"\nTotal dataset voxel grid coverage: {total_covered_voxels} / {total_voxels} "
        f"({total_coverage_pct:.2f}%)")


    print(f"\nLoaded {len(scene_heatmaps)} scenes in total.")
    #print("Scene counts by dataset:")
    #for k, v in dataset_counts.items():
    #    print(f"  {k}: {v} scenes")

    avg_coverage = np.mean(list(scene_coverage.values()))
    print(f"\nAverage scene coverage: {avg_coverage:.2f}%")
    print(f"\nMax frame occurrence: {max_info[2]} in scene: {max_info[0]}, frame index: {max_info[1]}")

    # === Short labels for axis ===
    short_labels = {}
    for scene_path in scene_heatmaps:
        parts = scene_path.strip("/").split("/")
        dataset = parts[1]
        difficulty = parts[2]
        scene = parts[3]
        label = f"{dataset}_{difficulty}_{scene}"
        short_labels[scene_path] = label

    sorted_scenes = sorted(scene_heatmaps.keys())
    sorted_labels = [short_labels[k] for k in sorted_scenes]

    # === Bar Chart: Coverage Percentage ===
    plt.figure(figsize=(max(20, len(sorted_scenes) * 0.4), 6))
    coverage_vals = [scene_coverage[k] for k in sorted_scenes]
    plt.bar(range(len(sorted_scenes)), coverage_vals, tick_label=sorted_labels)
    plt.xticks(rotation=90)
    plt.ylabel("Coverage (%)")
    plt.title("Voxel Grid Coverage per Scene")
    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        json_filename = os.path.splitext(os.path.basename(json_file))[0]
        plt.savefig(os.path.join(output_dir, f"{json_filename}_coverage_bar.png"))
    else:
        plt.show()

    # === Resampled + Normalized Heatmap ===
    desired_height = 100
    resampled_array = np.zeros((desired_height, len(sorted_scenes)))

    for i, scene_key in enumerate(sorted_scenes):
        original = np.array(scene_heatmaps[scene_key])
        zoom_factor = desired_height / len(original)
        interpolated = zoom(original, zoom_factor, order=1)

        interpolated = interpolated.astype(np.float32)
        if interpolated.max() > 0:
            clip_val = np.percentile(interpolated, 80)
            interpolated = np.clip(interpolated, 0, clip_val)
            
            if clip_val == 0:
                clip_val = interpolated.max()/2
            interpolated /= clip_val


        resampled_array[:, i] = interpolated

    # === Heatmap with Zero-Masking (visually distinct) ===
    import matplotlib.colors as mcolors

    # Mask the zero values visually
    masked_array = np.where(resampled_array == 0, np.nan, resampled_array)

    # Create a custom colormap where NaNs (masked values) appear gray
    cmap = plt.get_cmap("Greens").copy()
    cmap.set_bad(color="#ff0000")  # Light gray for zero entries

    plt.figure(figsize=(max(20, len(sorted_scenes) * 0.3), 8))
    sns.heatmap(
        masked_array,
        cmap=cmap,
        cbar=True,
        xticklabels=sorted_labels,
        yticklabels=False,
        linewidths=0.0,
        square=False,
        vmin=0,
        vmax=1
    )

    plt.xticks(rotation=90)
    plt.xlabel("Scene")
    plt.ylabel("Frame Index (resampled)")
    plt.title("Frame Occurrence Heatmap (Each Scene Normalized, Zero=Gray)")
    plt.tight_layout()

    if output_dir:
        json_filename = os.path.splitext(os.path.basename(json_file))[0]
        plt.savefig(os.path.join(output_dir, f"{json_filename}_heatmap.png"))
    else:
        plt.show()


        # === Binary Occurrence Heatmap (Green = exists, Red = missing) ===
    binary_array = (resampled_array > 0).astype(int)

    from matplotlib.colors import ListedColormap
    binary_cmap = ListedColormap(["#ff0000", "#4CAF50"])  # red, green

    plt.figure(figsize=(max(20, len(sorted_scenes) * 0.3), 8))
    ax = sns.heatmap(
        binary_array,
        cmap=binary_cmap,
        cbar=False,
        xticklabels=sorted_labels,
        yticklabels=False,
        linewidths=0.0,       # no gridlines
        linecolor=None,
        square=False
    )

    # Draw vertical lines between scenes
    for x in range(1, len(sorted_scenes)):
        ax.axvline(x=x, color='black', linewidth=0.5)

    plt.xticks(rotation=90)
    plt.xlabel("Scene")
    plt.ylabel("Frame Index (resampled)")
    plt.title("Binary Occurrence Heatmap (Red = Zero, Green = Present)")
    plt.tight_layout()

    if output_dir:
        plt.savefig(os.path.join(output_dir, f"{json_filename}_binary_heatmap.png"))
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze voxel usage JSON and plot per-scene normalized heatmap.")
    parser.add_argument("json_file", help="Path to voxel_usage_debug.json")
    parser.add_argument("--output", default=None, help="Optional directory to save figures")
    args = parser.parse_args()

    merge_and_plot(args.json_file, args.output)

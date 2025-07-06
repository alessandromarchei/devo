import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

def main(csv_path):
    # Read CSV
    df = pd.read_csv(csv_path, header=None)
    df.columns = ['dataset', 'scene', 'patches_per_frame', 'optimization_window',
                  'removal_window', 'patch_lifetime', 'ate', 'r_rmse', 'mpe']

    # Compute number of edges
    df['num_edges'] = df['patches_per_frame'] * df['removal_window'] ** 2

    # For ordered output
    df = df.sort_values(by=['num_edges', 'scene'])

    print("Detailed Log (Group by num_edges, then scene):")
    print("=" * 60)

    # Group by scene and num_edges to get medians and sample counts
    scene_medians = []
    for (num_edges, scene), group in df.groupby(['num_edges', 'scene']):
        median_ate = group['ate'].median()
        count = len(group)
        scene_medians.append((num_edges, scene, median_ate, count))

    # Convert to DataFrame for further aggregation
    scene_medians_df = pd.DataFrame(scene_medians, columns=['num_edges', 'scene', 'median_ate', 'count'])

    # Print details per condition
    for num_edges in sorted(scene_medians_df['num_edges'].unique()):
        subset = scene_medians_df[scene_medians_df['num_edges'] == num_edges]
        print(f"\nNumber of edges: {num_edges}")
        for _, row in subset.iterrows():
            print(f"  Scene: {row['scene']:25} | Median ATE: {row['median_ate']:.3f} cm | Samples: {int(row['count'])}")
        avg_ate = subset['median_ate'].mean()
        print(f"  --> Average of medians across scenes: {avg_ate:.3f} cm")

    # Plot
    mean_medians = scene_medians_df.groupby('num_edges')['median_ate'].mean().reset_index()
    mean_medians = mean_medians.sort_values(by='num_edges')

    plt.figure(figsize=(10, 6))
    plt.plot(mean_medians['num_edges'], mean_medians['median_ate'], marker='o')
    plt.xlabel('Number of Edges (patches_per_frame × removal_window²)')
    plt.ylabel('Average of Scene Median ATE [cm]')
    plt.title('ATE vs Number of Edges')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <csv_path>")
    else:
        main(sys.argv[1])

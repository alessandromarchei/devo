#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def analyze_activation(npz_path, low_thr=0.2, high_thr=0.6, min_lifetime=3):
    data = np.load(npz_path)
    df = pd.DataFrame({
        "ii": data["ii"].astype(np.int64),
        "jj": data["jj"].astype(np.int64),
        "kk": data["kk"].astype(np.int64),
        "iter": data["iter"].astype(np.int64),
        "wmag": data["wmag"].astype(np.float32),
    })
    df["edge"] = list(zip(df["ii"], df["jj"], df["kk"]))

    activations = []
    all_iters = sorted(df["iter"].unique())

    for edge, grp in df.groupby("edge"):
        grp = grp.sort_values("iter")
        vals = grp["wmag"].values
        iters = grp["iter"].values

        if len(vals) < min_lifetime:
            continue

        # Find first time weight is below low_thr
        low_indices = np.where(vals < low_thr)[0]
        high_indices = np.where(vals > high_thr)[0]

        if len(low_indices) == 0 or len(high_indices) == 0:
            continue

        # Look for first low → high transition
        activation_iter = None
        activation_speed = None
        for li in low_indices:
            hi_candidates = high_indices[high_indices > li]
            if len(hi_candidates) > 0:
                activation_iter = int(iters[hi_candidates[0]])
                activation_speed = int(iters[hi_candidates[0]] - iters[li])
                break

        if activation_iter is not None:
            activations.append({
                "edge": edge,
                "lifetime": len(vals),
                "activation_iter": activation_iter,
                "activation_speed": activation_speed
            })

    act_df = pd.DataFrame(activations)
    return act_df, len(df["edge"].unique()), all_iters


def plot_activation_stats(act_df, total_edges, all_iters):
    print(f"Total edges in run: {total_edges}")
    print(f"Edges with activation: {len(act_df)} ({100*len(act_df)/total_edges:.1f}%)")

    # Histogram: lifetimes
    plt.figure(figsize=(10, 4))
    plt.hist(act_df["lifetime"], bins=20, color="skyblue", edgecolor="black")
    plt.xlabel("Lifetime (iterations)")
    plt.ylabel("Number of edges")
    plt.title("Lifetime distribution of activator edges")
    plt.grid(True)

    # Histogram: activation speeds
    plt.figure(figsize=(10, 4))
    plt.hist(act_df["activation_speed"], bins=20, color="orange", edgecolor="black")
    plt.xlabel("Activation speed (iterations to go low→high)")
    plt.ylabel("Number of edges")
    plt.title("Activation speed distribution")
    plt.grid(True)

    # Time series: number of edges activating per iteration
    act_counts = act_df.groupby("activation_iter").size()
    counts_over_time = [act_counts.get(it, 0) for it in all_iters]

    plt.figure(figsize=(10, 4))
    plt.plot(all_iters, counts_over_time, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Number of activations")
    plt.title("Activations per iteration")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze edges that activate from low to high weight.")
    parser.add_argument("npz_path", help="Path to .npz file with logged weights.")
    parser.add_argument("--low-thr", type=float, default=0.2, help="Low threshold (start below).")
    parser.add_argument("--high-thr", type=float, default=0.6, help="High threshold (rise above).")
    parser.add_argument("--min-lifetime", type=int, default=3, help="Minimum lifetime to consider.")
    args = parser.parse_args()

    act_df, total_edges, all_iters = analyze_activation(args.npz_path,
                                                        low_thr=args.low_thr,
                                                        high_thr=args.high_thr,
                                                        min_lifetime=args.min_lifetime)
    plot_activation_stats(act_df, total_edges, all_iters)

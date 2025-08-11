#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def plot_edge_weights_heatmap(npz_path, start=0, count=50, which="wmag",
                              vmin=0.0, vmax=1.0, title=None, red_for_missing=True):
    data = np.load(npz_path)
    ii, jj, kk = data["ii"], data["jj"], data["kk"]
    it = data["iter"]
    vals = data[which]

    df = pd.DataFrame({
        "ii": ii.astype(np.int64),
        "jj": jj.astype(np.int64),
        "kk": kk.astype(np.int64),
        "iter": it.astype(np.int64),
        "val": vals.astype(np.float32),
    })

    # Unique edge tuple
    df["edge"] = list(zip(df["ii"], df["jj"], df["kk"]))

    # Stable order: by first appearance, then tuple
    edge_first_it = df.groupby("edge")["iter"].min()
    edges_sorted = sorted(edge_first_it.index, key=lambda e: (edge_first_it[e], e))

    edges_sel = edges_sorted[start:start+count]
    print(f"Selected edges: {len(edges_sel)} from {len(edges_sorted)} total.")
    #print(f"Edges: {edges_sel}")
    if not edges_sel:
        raise ValueError("No edges selected — check start/count values.")

    sub = df[df["edge"].isin(edges_sel)]

    # Make a contiguous iteration axis over the visible window
    it_min, it_max = int(sub["iter"].min()), int(sub["iter"].max())
    all_iters = np.arange(it_min, it_max + 1, dtype=np.int64)

    # Build matrix: rows=edges, cols=iterations, values=weight
    piv = sub.pivot_table(index="edge", columns="iter", values="val", aggfunc="mean")
    piv = piv.reindex(index=edges_sel, columns=all_iters)

    # Keep NaN where the edge is absent so we can color it red
    A = piv.values.astype(np.float32)

    # Colormap: blue scale for 0..1; NaN shown as red
    cmap = plt.cm.Blues
    if red_for_missing:
        cmap = cmap.copy()
        cmap.set_bad("red")

    # Plot
    plt.figure(figsize=(min(18, 0.02*len(all_iters)+4), min(12, 0.25*len(edges_sel)+2)))
    im = plt.imshow(A, aspect="auto", interpolation="nearest",
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    extent=[all_iters[0]-0.5, all_iters[-1]+0.5, len(edges_sel)-0.5, -0.5])

    cbar = plt.colorbar(im, label=f"{which} (0→1, white→blue)")
    yticks = np.arange(len(edges_sel))
    plt.yticks(yticks, [str(e) for e in edges_sel], fontsize=7)
    plt.xlabel("iteration")
    plt.ylabel("edge (ii, jj, kk)")
    if title is None:
        title = f"Edge {which} over time — edges[{start}:{start+count}]"
    plt.title(title)

    # Legend patch for missing intervals
    if red_for_missing:
        missing_patch = Patch(facecolor="red", edgecolor="red", label="absent (dropped/not alive)")
        plt.legend(handles=[missing_patch], loc="upper right", fontsize=8, frameon=True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot edge weights over time as heatmap (blue=weight, red=absent).")
    parser.add_argument("npz_path", help="Path to the .npz file (ii, jj, kk, iter, wx, wy, wmag).")
    parser.add_argument("count", type=int, help="Number of edges to plot.")
    parser.add_argument("start", type=int, help="Starting edge index.")
    parser.add_argument("--which", default="wmag", choices=["wmag", "wx", "wy"],
                        help="Which value to plot (default: wmag).")
    parser.add_argument("--vmin", type=float, default=0.0, help="Color scale minimum (default: 0.0).")
    parser.add_argument("--vmax", type=float, default=1.0, help="Color scale maximum (default: 1.0).")
    parser.add_argument("--no-red-missing", action="store_true",
                        help="Do not color missing intervals red.")
    args = parser.parse_args()

    plot_edge_weights_heatmap(
        args.npz_path,
        start=args.start,
        count=args.count,
        which=args.which,
        vmin=args.vmin,
        vmax=args.vmax,
        red_for_missing=not args.no_red_missing
    )

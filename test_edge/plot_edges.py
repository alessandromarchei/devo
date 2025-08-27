#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# --------------------------- Utils ---------------------------

def _ensure_outdir(outdir: str):
    os.makedirs(outdir, exist_ok=True)

def _gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    if np.all(x == 0):
        return 0.0
    x_sorted = np.sort(x)
    n = x_sorted.size
    cumx = np.cumsum(x_sorted)
    return (n + 1 - 2 * (cumx / cumx[-1]).sum()) / n

def _entropy_normalized(x: np.ndarray, bins: int = 50) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    counts, _ = np.histogram(x, bins=bins, range=(0.0, 1.0))
    total = counts.sum()
    if total == 0:
        return np.nan
    p = counts / total
    p = p[p > 0]
    H = -(p * np.log(p)).sum()
    Hmax = np.log(bins)
    return float(H / Hmax) if Hmax > 0 else np.nan

def get_global_iters(df: pd.DataFrame) -> np.ndarray:
    it_min, it_max = int(df["iter"].min()), int(df["iter"].max())
    return np.arange(it_min, it_max + 1, dtype=np.int64)

def _figure_size_wide(n_iters, min_w=8, extra_per_100=2.0):
    return (min_w + extra_per_100 * (n_iters / 100.0), 4.5)

def _figure_size_tall(n_rows, min_h=4, extra_per_50=1.5):
    return (10, min_h + extra_per_50 * (n_rows / 50.0))

# --------------------------- Data load ---------------------------

def load_wmag_df(npz_path: str, which="wmag") -> pd.DataFrame:
    data = np.load(npz_path)
    ii, jj, kk = data["ii"], data["jj"], data["kk"]
    it = data["iter"]
    vals = data[which]

    df = pd.DataFrame({
        "ii": ii.astype(np.int64),
        "jj": jj.astype(np.int64),
        "kk": kk.astype(np.int64),
        "iter": it.astype(np.int64),
        "val": np.clip(vals.astype(np.float32), 0.0, 1.0),
    })
    df["edge"] = list(zip(df["ii"], df["jj"], df["kk"]))
    return df

# --------------------------- Plots ---------------------------

def plot_edge_weights_heatmap(df: pd.DataFrame, start=0, count=50, which="wmag",
                              vmin=0.0, vmax=1.0, title=None, red_for_missing=True,
                              outdir="test_edge"):
    _ensure_outdir(outdir)

    edge_first_it = df.groupby("edge")["iter"].min()
    edges_sorted = sorted(edge_first_it.index, key=lambda e: (edge_first_it[e], e))
    edges_sel = edges_sorted[start:start+count]
    print(f"Selected edges: {len(edges_sel)} from {len(edges_sorted)} total.")
    if not edges_sel:
        raise ValueError("No edges selected — check start/count values.")

    sub = df[df["edge"].isin(edges_sel)].copy()
    all_iters_window = np.arange(int(sub["iter"].min()), int(sub["iter"].max())+1, dtype=np.int64)

    piv = sub.pivot_table(index="edge", columns="iter", values="val", aggfunc="mean")
    piv = piv.reindex(index=edges_sel, columns=all_iters_window)

    A = piv.values.astype(np.float32)

    cmap = plt.cm.Blues
    if red_for_missing:
        cmap = cmap.copy()
        cmap.set_bad("red")

    plt.figure(figsize=(min(18, 0.02*len(all_iters_window)+4),
                        min(12, 0.25*len(edges_sel)+2)))
    im = plt.imshow(
        A, aspect="auto", interpolation="nearest",
        cmap=cmap, vmin=vmin, vmax=vmax,
        extent=[all_iters_window[0]-0.5, all_iters_window[-1]+0.5, len(edges_sel)-0.5, -0.5]
    )
    cbar = plt.colorbar(im, label=f"{which} (0→1)")
    yticks = np.arange(len(edges_sel))
    plt.yticks(yticks, [str(e) for e in edges_sel], fontsize=7)
    plt.xlabel("iteration")
    plt.ylabel("edge (ii, jj, kk)")
    if title is None:
        title = f"Edge {which} over time — edges[{start}:{start+count}]"
    plt.title(title)

    if red_for_missing:
        missing_patch = Patch(facecolor="red", edgecolor="red", label="absent (dropped/not alive)")
        plt.legend(handles=[missing_patch], loc="upper right", fontsize=8, frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"edge_heatmap_{which}_edges_{start}_{start+count}.pdf"), bbox_inches="tight")
    plt.close()

def plot_weight_density_map(df: pd.DataFrame, bins=50, which="wmag", outdir="test_edge"):
    """
    2D density across ALL iterations: y = wmag bins in [0,1], x = iteration (full range),
    color = fraction of alive edges in that bin.
    """
    _ensure_outdir(outdir)
    all_iters = get_global_iters(df)
    bin_edges = np.linspace(0.0, 1.0, bins + 1, dtype=np.float64)

    density = np.full((bins, len(all_iters)), np.nan, dtype=np.float32)
    grp = df.groupby("iter", sort=True)

    for col_idx, itv in enumerate(all_iters):
        if itv not in grp.groups:
            continue  # stays NaN (no data this iter)
        vals = grp.get_group(itv)["val"].to_numpy(dtype=np.float32)
        counts, _ = np.histogram(vals, bins=bin_edges, range=(0.0, 1.0))
        total = counts.sum()
        frac = counts.astype(np.float32) / total if total > 0 else np.nan
        density[:, col_idx] = frac

    plt.figure(figsize=_figure_size_wide(len(all_iters)))
    im = plt.imshow(
        density, origin="lower", aspect="auto", interpolation="nearest",
        extent=[all_iters[0]-0.5, all_iters[-1]+0.5, 0.0, 1.0]
    )
    cbar = plt.colorbar(im, label="fraction of alive edges")
    plt.xlabel("iteration")
    plt.ylabel(f"{which}")
    plt.title(f"Density of {which} over time (ALL edges, full iteration range)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"density_map_{which}_bins{bins}.pdf"), bbox_inches="tight")
    plt.close()

def plot_quantiles_over_time(df: pd.DataFrame, quantiles=(0.1,0.25,0.5,0.75,0.9,0.95,0.99),
                             which="wmag", outdir="test_edge"):
    _ensure_outdir(outdir)
    all_iters = get_global_iters(df)
    grp = df.groupby("iter", sort=True)["val"]
    qs = {f"q{int(q*100)}": grp.quantile(q) for q in quantiles}
    qdf = pd.DataFrame(qs).reindex(all_iters)  # include every iteration in the range
    plt.figure(figsize=_figure_size_wide(len(qdf)))
    for col in qdf.columns:
        plt.plot(qdf.index.values, qdf[col].values, label=col)
    plt.ylim(0.0, 1.0)
    plt.xlabel("iteration")
    plt.ylabel(which)
    plt.title(f"{which} quantiles over time (ALL edges)")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"quantiles_{which}.pdf"), bbox_inches="tight")
    plt.close()
    return qdf

def plot_low_weight_fractions(df: pd.DataFrame, thresholds=(0.05, 0.1, 0.2, 0.3, 0.5),
                              which="wmag", outdir="test_edge"):
    _ensure_outdir(outdir)
    all_iters = get_global_iters(df)
    grp = df.groupby("iter", sort=True)["val"]
    fdict = {}
    for th in thresholds:
        fdict[f"p_below_{th:g}"] = grp.apply(lambda v: float((v < th).mean() if len(v) else np.nan))
    fdf = pd.DataFrame(fdict).reindex(all_iters)
    plt.figure(figsize=_figure_size_wide(len(fdf)))
    for col in fdf.columns:
        plt.plot(fdf.index.values, fdf[col].values, label=col)
    plt.ylim(0.0, 1.0)
    plt.xlabel("iteration")
    plt.ylabel("fraction of alive edges")
    plt.title(f"Fraction of edges below thresholds ({which}) — ALL iterations")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"low_weight_fractions_{which}.pdf"), bbox_inches="tight")
    plt.close()
    return fdf

def plot_alive_and_churn(df: pd.DataFrame, outdir="test_edge"):
    _ensure_outdir(outdir)
    all_iters = get_global_iters(df)
    grp = df.groupby("iter", sort=True)

    # Alive per iter on present iters only, then align to full axis
    alive_present = grp["edge"].nunique().astype(float)
    alive = alive_present.reindex(all_iters)  # NaN where no data

    # Churn: added/dropped between consecutive *present* iterations, then reindex
    sets_present = grp["edge"].agg(set)
    iters_present = sets_present.index.to_list()
    added = []
    dropped = []
    for idx, itv in enumerate(iters_present):
        if idx == 0:
            added.append(len(sets_present[itv]))
            dropped.append(0)
        else:
            prev = sets_present[iters_present[idx-1]]
            curr = sets_present[itv]
            added.append(len(curr - prev))
            dropped.append(len(prev - curr))
    churn_present = pd.DataFrame({"added": added, "dropped": dropped}, index=iters_present).astype(float)
    churn = churn_present.reindex(all_iters)

    # Plot alive
    plt.figure(figsize=_figure_size_wide(len(all_iters)))
    plt.plot(all_iters, alive.values)
    plt.xlabel("iteration")
    plt.ylabel("# alive edges")
    plt.title("Alive edges over time (ALL iterations)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "alive_edges.pdf"), bbox_inches="tight")
    plt.close()

    # Plot churn
    plt.figure(figsize=_figure_size_wide(len(all_iters)))
    plt.plot(all_iters, churn["added"].values, label="added")
    plt.plot(all_iters, churn["dropped"].values, label="dropped")
    plt.xlabel("iteration")
    plt.ylabel("# edges")
    plt.title("Edge churn per iteration (present iters; gaps shown as NaN)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "edge_churn.pdf"), bbox_inches="tight")
    plt.close()

    churn_out = pd.DataFrame({"alive": alive, "added": churn["added"], "dropped": churn["dropped"]}, index=all_iters)
    return churn_out

def plot_dispersion_metrics(df: pd.DataFrame, bins=50, which="wmag", outdir="test_edge"):
    """
    Mean, std, min, max, median, quantiles, entropy (normalized), gini over the FULL iteration axis.
    """
    all_iters = get_global_iters(df)
    grp = df.groupby("iter", sort=True)["val"]
    agg = pd.DataFrame({
        "mean": grp.mean(),
        "std": grp.std(ddof=0),
        "min": grp.min(),
        "max": grp.max(),
        "median": grp.median(),
        "q10": grp.quantile(0.10),
        "q25": grp.quantile(0.25),
        "q75": grp.quantile(0.75),
        "q90": grp.quantile(0.90),
        "count": grp.count().astype(float),
    }).reindex(all_iters)

    # Entropy & Gini per present iter; then align to full axis
    ents = {}
    ginis = {}
    for itv, g in df.groupby("iter", sort=True):
        vals = g["val"].to_numpy()
        ents[itv] = _entropy_normalized(vals, bins=bins)
        ginis[itv] = _gini(vals)
    agg["entropy_norm"] = pd.Series(ents).reindex(all_iters)
    agg["gini"] = pd.Series(ginis).reindex(all_iters)

    # Plots
    for col, ylabel, fname, ylim in [
        ("mean", which, f"mean_{which}.pdf", (0.0, 1.0)),
        ("std", "std", f"std_{which}.pdf", (0.0, 0.5)),
        ("entropy_norm", "normalized entropy", f"entropy_{which}.pdf", (0.0, 1.0)),
        ("gini", "gini", f"gini_{which}.pdf", (0.0, 1.0)),
    ]:
        plt.figure(figsize=_figure_size_wide(len(all_iters)))
        plt.plot(all_iters, agg[col].values)
        plt.xlabel("iteration")
        plt.ylabel(ylabel)
        plt.title(f"{col} of {which} over time (ALL iterations)")
        if ylim is not None:
            plt.ylim(*ylim)
        plt.tight_layout()
        _ensure_outdir(outdir)
        plt.savefig(os.path.join(outdir, fname), bbox_inches="tight")
        plt.close()
    return agg

def export_metrics_csv(metrics: pd.DataFrame, low_fracs: pd.DataFrame,
                       churn: pd.DataFrame, outpath: str):
    df = metrics.join(low_fracs, how="outer").join(churn, how="outer")
    df.index.name = "iter"
    df.to_csv(outpath)
    print(f"[saved] {outpath}")

# --------------------------- CLI ---------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze edge weights over time (heatmap on selection; ALL-iteration stats for density, quantiles, thresholds, churn, dispersion)."
    )
    parser.add_argument("npz_path", help="Path to the .npz file (ii, jj, kk, iter, wmag, optionally wx, wy).")
    parser.add_argument("count", type=int, help="Number of edges to plot in the heatmap.")
    parser.add_argument("start", type=int, help="Starting edge index for the heatmap.")
    parser.add_argument("--which", default="wmag", choices=["wmag", "wx", "wy"], help="Which value to analyze.")
    parser.add_argument("--vmin", type=float, default=0.0, help="Color scale minimum.")
    parser.add_argument("--vmax", type=float, default=1.0, help="Color scale maximum.")
    parser.add_argument("--no-red-missing", action="store_true", help="Do not color missing intervals red in the heatmap.")
    parser.add_argument("--bins", type=int, default=50, help="Bins for density/entropy in [0,1].")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3, 0.5],
                        help="Thresholds for low-weight fraction plots.")
    parser.add_argument("--outdir", default="test_edge", help="Output directory for figures and CSVs.")
    args = parser.parse_args()

    df = load_wmag_df(args.npz_path, which=args.which)

    # 1) Heatmap on your selected window only
    plot_edge_weights_heatmap(
        df, start=args.start, count=args.count, which=args.which,
        vmin=args.vmin, vmax=args.vmax, red_for_missing=not args.no_red_missing,
        outdir=args.outdir
    )

    # 2) Global stats over the FULL iteration axis
    plot_weight_density_map(df, bins=args.bins, which=args.which, outdir=args.outdir)
    qdf = plot_quantiles_over_time(df, which=args.which, outdir=args.outdir)
    fdf = plot_low_weight_fractions(df, thresholds=args.thresholds, which=args.which, outdir=args.outdir)
    churn_df = plot_alive_and_churn(df, outdir=args.outdir)
    mdf = plot_dispersion_metrics(df, bins=args.bins, which=args.which, outdir=args.outdir)

    # 3) Single CSV with everything, aligned to the full iteration range
    export_metrics_csv(
        metrics=mdf,
        low_fracs=fdf,
        churn=churn_df,
        outpath=os.path.join(args.outdir, f"edge_weight_metrics_{args.which}.csv"),
    )

if __name__ == "__main__":
    main()

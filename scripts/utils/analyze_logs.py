import sys
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import matplotlib.ticker as mticker

def parse_log(filename):
    values = defaultdict(list)
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith('=') or not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                continue
            var, mean_str, _ = parts
            try:
                mean = float(mean_str)
                values[var].append(mean)
            except:
                continue
    return values

def plot_variable(name, vals1, vals2, name1, name2, out_dir, xtick_nbins):
    os.makedirs(out_dir, exist_ok=True)
    x1 = np.arange(len(vals1))
    x2 = np.arange(len(vals2))
    diff_len = min(len(vals1), len(vals2))
    diff = np.abs(np.array(vals1[:diff_len]) - np.array(vals2[:diff_len]))
    rel_diff = diff / (np.abs(np.array(vals1[:diff_len])) + np.abs(np.array(vals2[:diff_len]) + 1e-9)) * 100.0
    xdiff = np.arange(diff_len)

    fig, axs = plt.subplots(2, 1, figsize=(6, 4), dpi=200)

    axs[0].plot(x1, vals1, label=name1, linewidth=1.5)
    axs[0].plot(x2, vals2, label=name2, linewidth=1.5)
    axs[0].set_title(f"{name} - Mean")
    axs[0].legend()
    axs[0].grid(True)
    axs[0].xaxis.set_major_locator(mticker.MaxNLocator(nbins=xtick_nbins))
    axs[0].tick_params(labelsize=5)
    axs[0].set_ylabel("Mean")

    axs[1].plot(xdiff, rel_diff, color='orange', linewidth=1.5)
    axs[1].set_title(f"{name} - Relative Difference")
    axs[1].grid(True)
    axs[1].xaxis.set_major_locator(mticker.MaxNLocator(nbins=xtick_nbins))
    axs[1].tick_params(labelsize=5)
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel("Relative Difference [%]")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{name}.png"))
    plt.close(fig)

def save_variable_plots(log1, log2, name1, name2, out_dir, xtick_nbins):
    all_keys = sorted(set(log1.keys()) | set(log2.keys()))
    for key in all_keys:
        vals1 = log1.get(key, [])
        vals2 = log2.get(key, [])
        plot_variable(key, vals1, vals2, name1, name2, out_dir, xtick_nbins)

def save_all_in_one_grid(log1, log2, name1, name2, out_dir, xtick_nbins):
    os.makedirs(out_dir, exist_ok=True)
    all_keys = sorted(set(log1.keys()) | set(log2.keys()))
    ncols = 4
    nrows = (len(all_keys) + ncols - 1) // ncols

    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), dpi=200)
    axs = axs.flatten()

    for i, key in enumerate(all_keys):
        ax = axs[i]
        vals1 = log1.get(key, [])
        vals2 = log2.get(key, [])
        x1 = np.arange(len(vals1))
        x2 = np.arange(len(vals2))

        if vals1:
            ax.plot(x1, vals1, label=name1, linewidth=1)
        if vals2:
            ax.plot(x2, vals2, label=name2, linewidth=1)

        ax.set_title(key, fontsize=9)
        ax.grid(True)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=xtick_nbins))
        ax.tick_params(labelsize=5)

    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=10)
    fig.suptitle("All Variables - Mean Comparison", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(out_dir, "all_vars.png"))
    plt.close(fig)

def print_first_differences(log1, log2, name1, name2, atol=1e-15):
    all_vars = set(log1.keys()) | set(log2.keys())
    max_len = max(max(len(log1.get(k, [])), len(log2.get(k, []))) for k in all_vars)

    for i in range(max_len):
        for var in all_vars:
            l1 = log1.get(var, [])
            l2 = log2.get(var, [])
            if i < len(l1) and i < len(l2):
                v1 = l1[i]
                v2 = l2[i]
                if abs(v1 - v2) > atol:
                    print(f"Iter {i}: variable={var} diff={abs(v1 - v2):.9f} ({name1}={v1:.6f}, {name2}={v2:.6f})")
                    break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log1", type=str)
    parser.add_argument("log2", type=str)
    parser.add_argument("--mode", type=str, default="both", choices=["plot", "diff", "both"], help="What to run")
    parser.add_argument("--out_dir", type=str, default="test_randomness/full_runs", help="Output directory for plots")
    parser.add_argument("--xtick_nbins", type=int, default=10, help="Max number of x-axis ticks")

    args = parser.parse_args()

    file1 = args.log1
    file2 = args.log2
    mode = args.mode
    out_dir = args.out_dir
    xtick_nbins = args.xtick_nbins

    name1 = os.path.splitext(os.path.basename(file1))[0]
    name2 = os.path.splitext(os.path.basename(file2))[0]

    log1 = parse_log(file1)
    log2 = parse_log(file2)

    if mode in ["diff", "both"]:
        print("======== DIFFERENCE REPORT ========")
        print_first_differences(log1, log2, name1, name2)

    if mode in ["plot", "both"]:
        save_variable_plots(log1, log2, name1, name2, out_dir=out_dir, xtick_nbins=xtick_nbins)
        save_all_in_one_grid(log1, log2, name1, name2, out_dir=out_dir, xtick_nbins=xtick_nbins)

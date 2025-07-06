import re
import argparse
import os
import matplotlib.pyplot as plt
from collections import defaultdict

def extract_all_blocks(filepath):
    with open(filepath) as f:
        lines = f.readlines()

    blocks = []
    current_block = {}
    for line in lines:
        if "====== Forward Pass" in line:
            if current_block:
                blocks.append(current_block)
                current_block = {}
            continue
        m = re.match(r"(.+?): mean=([-0-9.e]+), std=([-0-9.e]+)", line)
        if m:
            name = m.group(1).strip()
            mean = float(m.group(2))
            std = float(m.group(3))
            current_block[name] = (mean, std)
    if current_block:
        blocks.append(current_block)
    return blocks

def plot_and_save_drift(tensor_name, means1, stds1, means2, stds2, label1, label2, out_dir):
    d_mean = [b - a for a, b in zip(means1, means2)]
    d_std = [b - a for a, b in zip(stds1, stds2)]
    x = list(range(len(d_mean)))

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(x, d_mean, label=f"Δ mean ({label2} - {label1})")
    plt.title(f"{tensor_name} - Mean drift")
    plt.xlabel("Block index")
    plt.ylabel("Mean difference")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(x, d_std, label=f"Δ std ({label2} - {label1})", color='orange')
    plt.title(f"{tensor_name} - Std drift")
    plt.xlabel("Block index")
    plt.ylabel("Std difference")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    fname = f"{out_dir}/{tensor_name.replace(' ', '_')}_drift.png"
    plt.savefig(fname)
    plt.close()

def main(file1, file2, label1, label2, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    blocks1 = extract_all_blocks(file1)
    blocks2 = extract_all_blocks(file2)

    if len(blocks1) != len(blocks2):
        print(f"❌ Block count mismatch: {len(blocks1)} vs {len(blocks2)}")
        return

    all_keys = set()
    for b in blocks1 + blocks2:
        all_keys.update(b.keys())

    for key in sorted(all_keys):
        means1, stds1, means2, stds2 = [], [], [], []

        for i, (b1, b2) in enumerate(zip(blocks1, blocks2)):
            if key in b1 and key in b2:
                m1, s1 = b1[key]
                m2, s2 = b2[key]
                means1.append(m1)
                stds1.append(s1)
                means2.append(m2)
                stds2.append(s2)
            else:
                print(f"[Block {i}] ⚠️ Missing '{key}' in one of the files. Skipping.")
                break
        else:
            # All values were present — we can plot
            plot_and_save_drift(key, means1, stds1, means2, stds2, label1, label2, out_dir)

    print(f"✅ Done. Plots saved to '{out_dir}/'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file1", help="First log file (e.g., debug/exp1.txt)")
    parser.add_argument("file2", help="Second log file (e.g., debug/exp2.txt)")
    parser.add_argument("--label1", default="exp1", help="Label for first file")
    parser.add_argument("--label2", default="exp2", help="Label for second file")
    parser.add_argument("--out", default="debug", help="Output folder for plots")
    args = parser.parse_args()

    main(args.file1, args.file2, args.label1, args.label2, args.out)

import torch
import argparse
import os
from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

console = Console()

TOPK_UPDATE = 14
TOPK_PATCHIFIER = 8

def load_checkpoint(path):
    if not os.path.exists(path):
        console.print(f"[bold red]File not found:[/bold red] {path}")
        return None
    try:
        checkpoint = torch.load(path, map_location='cpu')
        return checkpoint
    except Exception as e:
        console.print(f"[bold red]Error loading file:[/bold red] {e}")
        return None

def describe_channels(shape):
    dims = list(shape)
    if len(dims) == 4:
        return f"{dims[0]} ← {dims[1]} ← [{dims[2]}, {dims[3]}]"
    elif len(dims) == 2:
        return f"{dims[0]} ← {dims[1]}"
    elif len(dims) == 1:
        return "bias / 1D param"
    else:
        return "N/A"

def memory_size(params_dict):
    return sum(v.element_size() * v.numel() for v in params_dict.values())

def memory_size_optimizer(opt_dict):
    total = 0
    for group in opt_dict.get("state", {}).values():
        for tensor in group.values():
            if isinstance(tensor, torch.Tensor):
                total += tensor.element_size() * tensor.numel()
    return total

def inspect_state_dict(state_dict, plot_pie=False):
    table = Table(title="Model State Dict", show_lines=True)
    table.add_column("Layer Name", style="cyan", no_wrap=True)
    table.add_column("Shape", style="green")
    table.add_column("Num Params", justify="right", style="magenta")
    table.add_column("Channels Info", style="blue")

    param_summary = {}
    total_params = 0

    for key, value in state_dict.items():
        shape = list(value.shape)
        num_params = value.numel()
        total_params += num_params
        channel_info = describe_channels(shape)
        table.add_row(key, str(shape), f"{num_params:,}", channel_info)

        if len(shape) > 1:  # skip biases
            param_summary[key] = num_params

    console.print(table)
    console.print(f"[bold yellow]Total parameters:[/bold yellow] [green]{total_params:,}[/green]")

    if plot_pie:
        plot_param_pie(param_summary)
def plot_param_pie(param_summary):
    sorted_params = sorted(param_summary.items(), key=lambda x: x[1], reverse=True)

    # Split by component
    patch_layers = [(k, v) for k, v in sorted_params if 'patch' in k.lower()]
    update_layers = [(k, v) for k, v in sorted_params if 'update' in k.lower()]

    # Take top 4 from each
    top_patch = patch_layers[:TOPK_PATCHIFIER]
    top_update = update_layers[:TOPK_UPDATE]

    top_keys = set(k for k, _ in (top_patch + top_update))
    other_params = [(k, v) for k, v in param_summary.items() if k not in top_keys]

    others_sum = sum(v for _, v in other_params)
    total_sum = sum(param_summary.values())

    top_params = top_patch + top_update
    if others_sum > 0:
        top_params.append(("Others", others_sum))

    labels = [k for k, _ in top_params]
    values = [v for _, v in top_params]

    # Percentages based on total param count
    percent_labels = [f"{k}\n{v / total_sum * 100:.1f}%\n{v:,}" for k, v in top_params]

    # Color coding
    hot_colors = cm.get_cmap('autumn')(np.linspace(0.2, 0.9, len(top_patch)))
    cold_colors = cm.get_cmap('cool')(np.linspace(0.1, 0.8, len(top_update)))
    color_list = list(hot_colors) + list(cold_colors)

    if others_sum > 0:
        color_list.append("lightgray")

    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, _ = ax.pie(values, colors=color_list, startangle=90, wedgeprops=dict(width=0.4, edgecolor='white'))

    for i, wedge in enumerate(wedges):
        angle = (wedge.theta2 + wedge.theta1) / 2
        x = np.cos(np.deg2rad(angle)) * 1.3
        y = np.sin(np.deg2rad(angle)) * 1.3
        ax.text(x, y, percent_labels[i], ha='center', va='center', fontsize=9)

    plt.title("Top Patchifier & Update Layers (by Param Count)", fontsize=14)
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Inspect PyTorch model .pth/.pt files (state_dict).")
    parser.add_argument("file", type=str, help="Path to the .pth or .pt file")
    parser.add_argument("--pie", action="store_true", help="Plot pie chart of top layers by parameter count")
    args = parser.parse_args()

    console.print(f"\n[bold blue]Inspecting file:[/bold blue] {args.file}\n")
    checkpoint = load_checkpoint(args.file)
    if checkpoint is None:
        return

    file_size_bytes = os.path.getsize(args.file)
    console.print(f"[bold green]Total .pth file size:[/bold green] {file_size_bytes / 1024 / 1024:.2f} MB")

    model_state = checkpoint.get("model_state_dict", checkpoint)
    optimizer_state = checkpoint.get("optimizer_state_dict", {})
    scheduler_state = checkpoint.get("scheduler_state_dict", {})

    model_bytes = memory_size(model_state)
    optimizer_bytes = memory_size_optimizer(optimizer_state)
    scheduler_bytes = len(str(scheduler_state).encode("utf-8"))

    console.print(f"[bold cyan]Model weights size:[/bold cyan] {model_bytes / 1024 / 1024:.2f} MB")
    console.print(f"[bold cyan]Optimizer state size:[/bold cyan] {optimizer_bytes / 1024 / 1024:.2f} MB")
    console.print(f"[bold cyan]Scheduler state (estimated):[/bold cyan] {scheduler_bytes / 1024 / 1024:.2f} MB")

    if isinstance(model_state, dict):
        inspect_state_dict(model_state, plot_pie=args.pie)

if __name__ == "__main__":
    main()

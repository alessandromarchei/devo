import torch
import argparse
import os
from rich.console import Console
from rich.table import Table

console = Console()

def load_state_dict(path):
    if not os.path.exists(path):
        console.print(f"[bold red]File not found:[/bold red] {path}")
        return None

    try:
        state_dict = torch.load(path, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        return state_dict
    except Exception as e:
        console.print(f"[bold red]Error loading file:[/bold red] {e}")
        return None

def describe_channels(shape):
    dims = list(shape)
    if len(dims) == 4:
        # Convolutional: [out_channels, in_channels, kH, kW]
        return f"{dims[0]} ← {dims[1]} ← [{dims[2]}, {dims[3]}]"
    elif len(dims) == 2:
        # Linear: [out_features, in_features]
        return f"{dims[0]} ← {dims[1]}"
    elif len(dims) == 1:
        return "bias / 1D param"
    else:
        return "N/A"

def inspect_state_dict(state_dict):
    table = Table(title="Model State Dict", show_lines=True)
    table.add_column("Layer Name", style="cyan", no_wrap=True)
    table.add_column("Shape", style="green")
    table.add_column("Num Params", justify="right", style="magenta")
    table.add_column("Channels Info", style="blue")

    total_params = 0
    for key, value in state_dict.items():
        shape = list(value.shape)
        num_params = value.numel()
        total_params += num_params
        channel_info = describe_channels(shape)
        table.add_row(key, str(shape), f"{num_params:,}", channel_info)

    console.print(table)
    console.print(f"[bold yellow]Total parameters:[/bold yellow] [green]{total_params:,}[/green]")

def main():
    parser = argparse.ArgumentParser(description="Inspect PyTorch model .pth/.pt files (state_dict).")
    parser.add_argument("file", type=str, help="Path to the .pth or .pt file")
    args = parser.parse_args()

    console.print(f"\n[bold blue]Inspecting file:[/bold blue] {args.file}\n")
    state_dict = load_state_dict(args.file)

    if state_dict is not None:
        inspect_state_dict(state_dict)

if __name__ == "__main__":
    main()

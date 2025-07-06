import torch

def filter_update_only(input_path, output_path):
    # Load the original checkpoint
    checkpoint = torch.load(input_path, map_location='cpu')

    # Filter only update.* parameters from model_state_dict
    update_only = {
        k: v for k, v in checkpoint.get('model_state_dict', {}).items()
        if k.startswith('patchify.')
    }

    # Save ONLY the filtered model_state_dict (and optionally steps)
    new_checkpoint = {
        'model_state_dict': update_only,
        'steps': checkpoint.get('steps', 0)
    }

    # Write to new file (ensure you're not overwriting the original unless intended)
    torch.save(new_checkpoint, output_path)
    print(f"Filtered checkpoint saved to: {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Filter only `update.*` weights from a .pth file.")
    parser.add_argument("--input", default="checkpoints/patchifier_smallv2/100000.pth", help="Path to input .pth file")
    parser.add_argument("--output", default="checkpoints/patchify.pth", help="Path to save the filtered .pth file")
    args = parser.parse_args()

    filter_update_only(args.input, args.output)

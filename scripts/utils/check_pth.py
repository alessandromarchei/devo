import torch
import sys

def try_load_model(path):
    try:
        state_dict = torch.load(path, map_location='cpu')
        print("✅ Successfully loaded state_dict.")
        return state_dict
    except Exception as e:
        print("❌ Failed to load state_dict:")
        print(e)
        return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_pth.py <path_to_model>")
        sys.exit(1)

    model_path = sys.argv[1]
    _ = try_load_model(model_path)

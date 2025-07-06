import torch
import sys

checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else '/home/amarchei/Desktop/repos/DEVO/checkpoints/update_nosoft/150000.pth'


# Load the checkpoint
ckpt = torch.load(checkpoint_path, map_location='cpu')

# Print all top-level keys
print("Top-level keys in checkpoint:", ckpt.keys())

# Print training step
print(f"\nSteps: {ckpt['steps']}")

# --- MODEL STATE DICT ---
print("\nModel state dict (parameter names and shapes):")
for name, param in ckpt['model_state_dict'].items():
    print(f"  {name}: {tuple(param.shape)}")

# --- OPTIMIZER STATE DICT ---
print("\nOptimizer state dict summary:")
opt_state = ckpt['optimizer_state_dict']

# Print optimizer param groups
for i, group in enumerate(opt_state['param_groups']):
    print(f"  Param group {i}:")
    for k, v in group.items():
        if k != 'params':
            print(f"    {k}: {v}")

# --- SCHEDULER STATE DICT ---
print("\nScheduler state dict:")
sched_state = ckpt['scheduler_state_dict']
for k, v in sched_state.items():
    print(f"  {k}: {v}")

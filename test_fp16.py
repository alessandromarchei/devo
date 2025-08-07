import numpy as np

# Original float32 values
values = np.array([824.152515, 0.000000, 8.921875, -3.371094], dtype=np.float32)

print("Checking fp16 representability:\n")
for val in values:
    val_fp16 = np.float16(val)
    val_back = np.float32(val_fp16)
    error = abs(val - val_back)
    print(f"Value: {val:10.6f} -> fp16: {val_fp16:10.6f} -> back to float32: {val_back:10.6f} | Error: {error:.6e}")

import torch

# Set seed for reproducibility
torch.manual_seed(0)

# Generate a random 8x8 matrix
A = torch.randn(8, 8)

# Create a symmetric positive-definite matrix: S = A @ A.T + εI
# (Adding εI ensures positive-definiteness)
epsilon = 1e-3
S = A @ A.T + epsilon * torch.eye(8)

# Perform Cholesky decomposition
U = torch.linalg.cholesky(S)

# Reconstruct the matrix
S_reconstructed = U @ U.T

# Print the results
torch.set_printoptions(precision=4, linewidth=200)

print("Input matrix S:")
print(S)

print("\nCholesky factor U (lower-triangular):")
print(U)

print("\nReconstructed S from U:")
print(S_reconstructed)

print("\nMaximum absolute difference:")
print((S - S_reconstructed).abs().max())

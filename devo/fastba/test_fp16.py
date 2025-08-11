import sys
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns

'''
Usage Examples:

# Only check if PD
python test_pd_matrix.py /path/to/S_matrix_fp16.txt

# Compare FP16 matrix to FP32 reference
python test_pd_matrix.py /path/to/S_matrix_fp16.txt /path/to/S_matrix_fp32.txt
'''


# Add this inside analyze_pd() at the end if both decompositions work
def test_solver_equivalence(matrix: torch.Tensor):
    n = matrix.shape[0]
    y = torch.randn(n, dtype=matrix.dtype)

    try:
        L = torch.linalg.cholesky(matrix)
        x_chol = torch.cholesky_solve(y.unsqueeze(1), L).squeeze(1)
        chol_ok = True
        print("✅ torch.linalg.cholesky succeeded.")
    except Exception as e:
        print("❌ torch.linalg.cholesky failed:", e)
        chol_ok = False
        x_chol = None

    try:
        LU, piv = torch.linalg.lu_factor(matrix)
        x_lu = torch.linalg.lu_solve(LU, piv, y)
        lu_ok = True
        print("✅ LU decomposition succeeded.")
    except Exception as e:
        print("❌ LU decomposition failed:", e)
        lu_ok = False
        x_lu = None

    if chol_ok and lu_ok:
        rel_diff = torch.abs(x_chol - x_lu) / (torch.abs(x_lu) + 1e-6)
        print("\n🧪 Comparing solutions from Cholesky and LU:")
        print(f"  Max  relative difference: {rel_diff.max().item():.3e}")
        print(f"  Mean relative difference: {rel_diff.mean().item():.3e}")
        print(f"  Std  relative difference: {rel_diff.std().item():.3e}")
    else:
        print("ℹ️ Skipping solver comparison: one or both failed.")


# Add this at the end of analyze_pd(), replacing return statements if successful
    try:
        L = torch.linalg.cholesky(matrix)
        print("\n✅ torch.linalg.cholesky succeeded.")
        test_solver_equivalence(matrix)
        return
    except RuntimeError as e:
        print("\n❌ torch.linalg.cholesky failed:")
        print(f"  {e}")


#make matrices symmetric if they sare not
def make_symmetric(matrix):
    return (matrix + matrix.T) / 2


def load_matrix(filename):
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        values = []
        for line in lines:
            if line.strip() == "":
                continue
            if "iteration" in line or "Tensor" in line:
                continue
            row = [float(x) for x in line.strip().split()]
            values.append(row)
        matrix = np.array(values, dtype=np.float32)
        return torch.from_numpy(matrix)
    except Exception as e:
        print(f"❌ Error loading matrix from {filename}: {e}")
        sys.exit(1)

def cholesky_decompose_cstyle(A: np.ndarray):
    """C-style Cholesky decomposition. Returns (L, success_flag)"""
    n = A.shape[0]
    L = np.zeros((n, n), dtype=np.float32)
    
    for i in range(n):
        for j in range(i + 1):
            sum_ = A[i, j]
            for k in range(j):
                sum_ -= L[i, k] * L[j, k]
            
            if i == j:
                if sum_ <= 0.0:
                    return None, False  # Not positive definite
                L[i, j] = np.sqrt(sum_)
            else:
                L[i, j] = sum_ / L[j, j]
    
    return L, True

def nearest_pd(A):
    """Higham’s algorithm: returns nearest symmetric PD matrix"""
    B = (A + A.T) / 2
    eigvals, eigvecs = torch.linalg.eigh(B)
    eigvals_clipped = torch.clamp(eigvals, min=1e-6)
    A_pd = eigvecs @ torch.diag(eigvals_clipped) @ eigvecs.T
    return (A_pd + A_pd.T) / 2
def analyze_pd(matrix: torch.Tensor):
    print(f"\nMatrix shape: {matrix.shape}")
    if not torch.allclose(matrix, matrix.T, atol=1e-4):
        print("⚠️ Matrix is not symmetric. Cholesky decomposition requires symmetry.")

        print("Making matrix symmetric...")
        matrix = make_symmetric(matrix)
        print("✅ Matrix is now symmetric.")
        if not torch.allclose(matrix, matrix.T, atol=1e-4):
            print("❌ Failed to make matrix symmetric. Exiting.")
            sys.exit(1)

    print("✅ Matrix is symmetric.")

    #calculate the condition number
    cond_num = torch.linalg.cond(matrix)
    print(f"\nCondition number: {cond_num.item():.4e}")

    eigvals = torch.linalg.eigvalsh(matrix)
    min_eig = eigvals.min().item()
    num_neg = (eigvals <= 0).sum().item()
    print(f"\nEigenvalues:")
    print(f"  Min eigenvalue       : {min_eig:.6e}")
    print(f"  # of ≤ 0 eigenvalues : {num_neg} / {len(eigvals)}")

    if min_eig <= 0:
        print("⚠️ Matrix is not PD: min eigenvalue is ≤ 0.")

    print("\nLeading principal minors:")
    failed = False
    for k in range(1, matrix.shape[0] + 1):
        minor = torch.det(matrix[:k, :k])
        print(f"  Minor {k:2d}: {minor.item(): .4e}")
        if minor <= 0 and not failed:
            failed = True

    # Try PyTorch Cholesky
    try:
        L = torch.linalg.cholesky(matrix)
        print("\n✅ torch.linalg.cholesky succeeded.")
    except RuntimeError as e:
        print("\n❌ torch.linalg.cholesky failed:")
        print(f"  {e}")

    # Try C-style Cholesky
    print("\n▶️ Trying C-style Cholesky decomposition (numpy)...")
    A_np = matrix.numpy()
    L_cstyle, success = cholesky_decompose_cstyle(A_np)
    if success:
        print("✅ C-style Cholesky decomposition succeeded.")
        return
    else:
        print("❌ C-style Cholesky decomposition failed: matrix not positive definite.")

    # Try LU decomposition
    print("\n🔁 Trying LU decomposition")

    try:
        # Create dummy RHS vector (same size as matrix)
        y = torch.randn(matrix.shape[0], dtype=torch.float32)

        # Factor and solve
        LU, pivots = torch.linalg.lu_factor(matrix)

        print("✅ LU decomposition succeeded.")
    except Exception as e:
        print("❌ LU decomposition failed:")
        print(f"  {e}")

    # Distance from nearest PD
    A_pd = nearest_pd(matrix)
    frob_dist = torch.norm(matrix - A_pd, p='fro').item()
    print(f"\n📏 Frobenius distance to nearest PD matrix: {frob_dist:.4e}")
    print(f"→ Suggest shift: {abs(min_eig) + 1e-5:.4e} to make eigenvalues strictly positive.")


def compare_matrices(fp32, fp16):
    rel_diff = torch.abs(fp16 - fp32) / (torch.abs(fp32) + 1e-6)
    print(f"\n🔍 Relative difference statistics:")
    print(f"  Max  : {rel_diff.max().item():.3e}")
    print(f"  Mean : {rel_diff.mean().item():.3e}")
    print(f"  Std  : {rel_diff.std().item():.3e}")

    # Plot heatmap
    rel_diff_np = rel_diff.numpy()
    plt.figure(figsize=(8, 6))
    sns.heatmap(rel_diff_np, annot=False, cmap="viridis", cbar_kws={'label': 'Relative Difference'})
    plt.title("Heatmap of Relative Differences (FP16 vs FP32)")
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_pd_matrix.py <matrix_fp16.txt> [matrix_fp32.txt]")
        sys.exit(1)
    

    matrix_fp16 = load_matrix(sys.argv[1])
    analyze_pd(matrix_fp16)

    if len(sys.argv) > 2:
        matrix_fp32 = load_matrix(sys.argv[2])
        compare_matrices(matrix_fp32, matrix_fp16)

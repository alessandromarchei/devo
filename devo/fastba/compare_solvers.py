import sys
import numpy as np
import torch



'''
run this file by passing a matrix file as an argument:
python compare_solvers.py matrix.txt

'''
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


def make_symmetric(matrix: torch.Tensor):
    return (matrix + matrix.T) / 2

def solve_and_compare(matrix: torch.Tensor):
    matrix = make_symmetric(matrix)
    n = matrix.shape[0]
    y = torch.randn(n, dtype=matrix.dtype)

    chol_ok = False
    lu_ok = False

    # Try Cholesky
    try:
        L = torch.linalg.cholesky(matrix)
        x_chol = torch.cholesky_solve(y.unsqueeze(1), L).squeeze(1)
        chol_ok = True
        print("✅ Cholesky decomposition succeeded.")
    except Exception as e:
        print("❌ Cholesky decomposition failed:", e)

    # Try LU
    try:
        LU, piv = torch.linalg.lu_factor(matrix)
        x_lu = torch.linalg.lu_solve(LU, piv, y.unsqueeze(1)).squeeze(1)
        lu_ok = True
        print("✅ LU decomposition succeeded.")
    except Exception as e:
        print("❌ LU decomposition failed:", e)

    if chol_ok and lu_ok:
        print("\n🧪 Comparing solutions from Cholesky and LU:")
        rel_diff = torch.abs(x_chol - x_lu) / (torch.abs(x_lu) + 1e-6)
        print(f"  Max  relative difference: {rel_diff.max().item():.3e}")
        print(f"  Mean relative difference: {rel_diff.mean().item():.3e}")
        print(f"  Std  relative difference: {rel_diff.std().item():.3e}")
    else:
        print(" Solver comparison skipped: one or both methods failed.")


    #print visually the results, one column for each method, like chol[0] and lu[0]
    for i in range(n):
        print(f"Row {i}: Cholesky: {x_chol[i].item():.6f}, LU: {x_lu[i].item():.6f}")
        
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python compare_solvers.py <matrix.txt>")
        sys.exit(1)

    matrix_file = sys.argv[1]
    matrix = load_matrix(matrix_file)
    print(f"✅ Loaded matrix of shape {matrix.shape} from '{matrix_file}'")

    solve_and_compare(matrix)

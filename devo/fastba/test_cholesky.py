import torch
import time
import numpy as np

def benchmark_solvers(device, n=60, runs=500, warmup=30):
    torch.manual_seed(0)

    # Generate SPD matrix S = A @ A.T + εI
    A = torch.randn(n, n, device=device)
    epsilon = 1e-3
    S = A @ A.T + epsilon * torch.eye(n, device=device)
    y = torch.randn(n, 1, device=device)

    # Warm-up
    for _ in range(warmup):
        _ = torch.linalg.solve(S, y)
        L = torch.linalg.cholesky(S)
        _ = torch.cholesky_solve(y, L)

    # Timings
    solve_times = []
    chol_solve_times = []

    for _ in range(runs):
        # Direct solve
        start = time.perf_counter()
        x1 = torch.linalg.solve(S, y)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        solve_times.append(time.perf_counter() - start)

        # Cholesky + cholesky_solve
        start = time.perf_counter()
        L = torch.linalg.cholesky(S)
        x2 = torch.cholesky_solve(y, L)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        chol_solve_times.append(time.perf_counter() - start)

    solve_times = np.array(solve_times)
    chol_solve_times = np.array(chol_solve_times)

    print(f"\n--- Results on {device} ---")
    print(f"Matrix size: {n}x{n}, runs: {runs}, warmup: {warmup}")
    print(f"Direct solve:       {solve_times.mean()*1e6:.2f} ± {solve_times.std()*1e6:.2f} µs")
    print(f"Cholesky + solve:   {chol_solve_times.mean()*1e6:.2f} ± {chol_solve_times.std()*1e6:.2f} µs")
    print(f"Speedup:            {solve_times.mean() / chol_solve_times.mean():.2f}x")

if __name__ == "__main__":
    benchmark_solvers(torch.device("cpu"))
    if torch.cuda.is_available():
        benchmark_solvers(torch.device("cuda"))

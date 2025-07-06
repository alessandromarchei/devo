import torch
import ba_cpu

def BA_cpu(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations=2):
    return ba_cpu.forward(poses.data, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations)
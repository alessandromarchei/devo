import torch
import cuda_ba
import cuda_ba_double

neighbors = cuda_ba.neighbors
reproject = cuda_ba.reproject

def BA(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations=2):
    return cuda_ba.forward(poses.data, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations)

def BA_double(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations=2):
    return cuda_ba_double.forward(poses.data, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations)
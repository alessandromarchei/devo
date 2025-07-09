import torch
import cuda_ba
import cuda_ba_double
import cuda_ba_trunc
import cuda_ba_trunc_double
import cuda_ba_det
import cuda_ba_debug

neighbors = cuda_ba.neighbors
reproject = cuda_ba.reproject

def BA(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations=2):
    return cuda_ba.forward(poses.data, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations)

def BA_double(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations=2):
    return cuda_ba_double.forward(poses.data, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations)

def BA_trunc(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations=2, decimal_places=0):
    return cuda_ba_trunc.forward(poses.data, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations, decimal_places)

def BA_trunc_double(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations=2, decimal_places=0):
    return cuda_ba_trunc_double.forward(poses.data, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations, decimal_places)

def BA_det(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations=2):
    return cuda_ba_det.forward(poses.data, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations)

def BA_debug(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations=2):
    return cuda_ba_debug.forward(poses.data, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations)
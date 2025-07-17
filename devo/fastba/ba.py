import torch
import cuda_ba
import cuda_ba_double
import cuda_ba_trunc
import cuda_ba_trunc_double
import cuda_ba_det
import cuda_ba_debug
import cuda_ba_single_thread
import cuda_ba_red
import cuda_ba_red_cpu_fw
import cuda_ba_red_cpu_bw
import cuda_ba_red_cpu_fw_save
import cuda_ba_red_cpu_bw_save
import cuda_ba_kahan
import cuda_ba_red2


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

def BA_single_thread(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations=2):
    return cuda_ba_single_thread.forward(poses.data, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations)

def BA_red(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations=2):
    return cuda_ba_red.forward(poses.data, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations)

def BA_red_cpu_fw(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations=2):
    return cuda_ba_red_cpu_fw.forward(poses.data, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations)

def BA_red_cpu_bw(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations=2):
    return cuda_ba_red_cpu_bw.forward(poses.data, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations)

def BA_red_cpu_fw_save(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations=2):
    return cuda_ba_red_cpu_fw_save.forward(poses.data, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations)

def BA_red_cpu_bw_save(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations=2):
    return cuda_ba_red_cpu_bw_save.forward(poses.data, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations)

def BA_red_kahan(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations=2):
    return cuda_ba_kahan.forward(poses.data, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations)

def BA_red2(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations=2, reduction_config=None):
    if reduction_config is None:
        reduction_config = [1, 1, 1, 1, 1]
    return cuda_ba_red2.forward2(poses.data, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations, reduction_config)
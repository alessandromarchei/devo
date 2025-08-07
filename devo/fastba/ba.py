import torch
import cuda_ba
import cuda_ba_double
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
import cuda_ba_kahan_bw
import ba_cpu
import cuda_ba_kahan_db64
import ba_cpu_fp128
import ba_cpu_profile
import ba_cpu_debug 
import ba_cpu_fp16
import cuda_ba_fp16_chol
import cuda_ba_fp32_chol
import cuda_ba_fp16_lu
import cuda_ba_fp32_lu
import cuda_ba_fp32_chol2
import cuda_ba_bf16_chol
import cuda_ba_bf16_lu

neighbors = cuda_ba.neighbors
reproject = cuda_ba.reproject

def BA(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations=2):
    return cuda_ba.forward(poses.data, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations)

def BA_double(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations=2):
    return cuda_ba_double.forward(poses.data, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations)

# def BA_trunc(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations=2, decimal_places=0):
#     return cuda_ba_trunc.forward(poses.data, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations, decimal_places)

# def BA_trunc_double(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations=2, decimal_places=0):
#     return cuda_ba_trunc_double.forward(poses.data, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations, decimal_places)

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

def BA_kahan(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations=2):
    return cuda_ba_kahan.forward(poses.data, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations)

def BA_red_kahan_bw(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations=2):
    return cuda_ba_kahan_bw.forward(poses.data, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations)

def BA_red2(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations=2, reduction_config=None):
    if reduction_config is None:
        reduction_config = [1, 1, 1, 1, 1]
    return cuda_ba_red2.forward2(poses.data, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations, reduction_config)

def BA_cpu(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations=2):
    return ba_cpu.forward(poses.data, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations)

def BA_kahan_db64(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations=2):
    return cuda_ba_kahan_db64.forward(poses.data, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations)

def BA_cpu_fp128(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations=2):
    return ba_cpu_fp128.forward(poses.data, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations)

def BA_cpu_profile(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations=2):
    return ba_cpu_profile.forward(poses.data, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations)

def BA_cpu_debug(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations=2):
    return ba_cpu_debug.forward(poses.data, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations)

def BA_cpu_fp16(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations=2):
    #cast the input tensors to half precision, in case they are not already + convert to cpu tensor
    # poses = poses.to(torch.float16).cpu()
    # patches = patches.to(torch.float16).cpu()
    # intrinsics = intrinsics.to(torch.float16).cpu()
    # target = target.to(torch.float16).cpu()
    # weight = weight.to(torch.float16).cpu()
    # lmbda = lmbda.to(torch.float16).cpu()
    # ii = ii.cpu()
    # jj = jj.cpu()
    # kk = kk.cpu()
    # t0 = t0.cpu()
    # t1 = t1.cpu()

    
#convert the output tensors back to float32
    # poses = poses.to(torch.float32).cuda()
    # patches = patches.to(torch.float32).cuda()
    # intrinsics = intrinsics.to(torch.float32).cuda()
    # target = target.to(torch.float32).cuda()
    # weight = weight.to(torch.float32).cuda()
    # lmbda = lmbda.to(torch.float32).cuda()
    # ii = ii.cuda()
    # jj = jj.cuda()
    # kk = kk.cuda()
    # t0 = t0.cuda()
    # t1 = t1.cuda()

    return ba_cpu_fp16.forward(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations)

def BA_fp16_chol(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations=2):

    return cuda_ba_fp16_chol.forward(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations)

def BA_fp32_lu(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations=2):

    return cuda_ba_fp32_lu.forward(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations)

def BA_fp16_lu(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations=2):

    return cuda_ba_fp16_lu.forward(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations)

def BA_fp32_chol(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations=2):
    return cuda_ba_fp32_chol.forward(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations)

def BA_fp32_chol2(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations=2):
    return cuda_ba_fp32_chol2.forward(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations)

def BA_bf16_chol(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations=2):
    return cuda_ba_bf16_chol.forward(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations)

def BA_bf16_lu(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations=2):
    return cuda_ba_bf16_lu.forward(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations)

import torch
import torch.nn.functional as F


all_times = []

class Timer:
    def __init__(self, name, enabled=True):
        self.name = name
        self.enabled = enabled

        if self.enabled:
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        if self.enabled:
            self.start.record()
        
    def __exit__(self, type, value, traceback):
        global all_times
        if self.enabled:
            self.end.record()
            torch.cuda.synchronize()

            elapsed = self.start.elapsed_time(self.end)
            all_times.append(elapsed)
            print(self.name, elapsed)


def coords_grid(b, n, h, w, **kwargs):
    """ coordinate grid """
    x = torch.arange(0, w, dtype=torch.float, **kwargs)
    y = torch.arange(0, h, dtype=torch.float, **kwargs)
    coords = torch.stack(torch.meshgrid(y, x, indexing="ij"))
    return coords[[1,0]].view(1, 1, 2, h, w).repeat(b, n, 1, 1, 1)

def coords_grid_with_index(d, **kwargs):
    """
    coordinate grid with frame index
    
    Returns:
        coords (Tensor): grid of x-, y-coordinates & depth value for each frame (B,n_frames,3,H,W)
        index (Tensor): (B,n_frames,H,W)
    """
    b, n, h, w = d.shape
    i = torch.ones_like(d)
    x = torch.arange(0, w, dtype=torch.float, **kwargs)
    y = torch.arange(0, h, dtype=torch.float, **kwargs)

    y, x = torch.stack(torch.meshgrid(y, x, indexing="ij"))
    y = y.view(1, 1, h, w).repeat(b, n, 1, 1)
    x = x.view(1, 1, h, w).repeat(b, n, 1, 1)

    coords = torch.stack([x, y, d], dim=2)
    index = torch.arange(0, n, dtype=torch.float, **kwargs)
    index = index.view(1, n, 1, 1, 1).repeat(b, 1, 1, h, w)

    return coords, index

def patchify(x, patch_size=3):
    """ extract patches from video """
    b, n, c, h, w = x.shape
    x = x.view(b*n, c, h, w)
    y = F.unfold(x, patch_size)
    y = y.transpose(1,2)
    return y.reshape(b, -1, c, patch_size, patch_size)


def pyramidify(fmap, lvls=[1]):
    """ turn fmap into a pyramid """
    b, n, c, h, w = fmap.shape

    pyramid = []
    for lvl in lvls:
        gmap =  F.avg_pool2d(fmap.view(b*n, c, h, w), lvl, stride=lvl)
        pyramid += [ gmap.view(b, n, c, h//lvl, w//lvl) ]
        
    return pyramid

def all_pairs_exclusive(n, **kwargs):
    ii, jj = torch.meshgrid(torch.arange(n, **kwargs), torch.arange(n, **kwargs))
    k = ii != jj
    return ii[k].reshape(-1), jj[k].reshape(-1)

def set_depth(patches, depth):
    patches[...,2,:,:] = depth[...,None,None]
    return patches

def flatmeshgrid(*args, **kwargs):
    grid = torch.meshgrid(*args, **kwargs)
    return (x.reshape(-1) for x in grid)


def nms_image(image_tensor, kernel_size=3):
    """
    Performs non-maximum suppression on each channel of a 3D tensor representing an image.

    Args:
    - image_tensor: torch.Tensor of shape (C, H, W)
    - kernel_size: int, size of non maximum suppression around maximums

    Returns:
    - out_tensor: torch.Tensor of shape (C, H, W), float tensor, suppressed version of image_tensor
    """

    image_tensor = image_tensor.unsqueeze(0)
    padding = (kernel_size - 1) // 2

    # Max pool over height and width dimensions
    max_vals = torch.nn.functional.max_pool2d(
        image_tensor, kernel_size, stride=1, padding=padding
    )
    max_vals = max_vals.squeeze(0)

    # Perform non-maximum suppression
    mask = max_vals == image_tensor
    mask = mask.squeeze(0)
    image_tensor = image_tensor.squeeze(0)

    return image_tensor * mask.float()


def get_coords_from_topk_events(
    events,
    patches_per_image,
    border_suppression_size=0,
    non_max_supp_rad=0,
):
    positive_event_tensor = torch.abs(events.squeeze(0))
    downsampled_event_tensor = F.avg_pool2d(positive_event_tensor, 4, 4)
    event_in_xy_form = downsampled_event_tensor.transpose(3, 2)
    ev_mean = torch.mean(event_in_xy_form, dim=1)

    if border_suppression_size != 0:
        # set the borders to 0
        ev_mean[:, :border_suppression_size, :] = 0
        ev_mean[:, -border_suppression_size:, :] = 0
        ev_mean[:, :, :border_suppression_size] = 0
        ev_mean[:, :, -border_suppression_size:] = 0

    if non_max_supp_rad != 0:
        # perform non maximum suppression
        ev_mean = nms_image(ev_mean, kernel_size=non_max_supp_rad)

    event_mean_flat = torch.flatten(ev_mean, start_dim=1, end_dim=-1)
    values, indices = torch.topk(event_mean_flat, k=patches_per_image, dim=-1)

    # compute the row and column indices of the top k values in the flattened tensor
    row_indices = indices / ev_mean.shape[-1]
    col_indices = indices % ev_mean.shape[-1]

    # compute the batch indices of the top k values in the flattened tensor
    batch_indices = (
        torch.arange(ev_mean.shape[0], device="cuda")
        .view(-1, 1)
        .repeat(1, patches_per_image)
    )

    # combine the batch, row, and column indices to obtain the indices in the original 3D tensor
    orig_indices = torch.stack((batch_indices, row_indices, col_indices), dim=-1)

    coords = orig_indices[:, :, 1:]
    return coords


def write_tensor(f, name, tensor,max_streak=100):
    tensor = tensor.detach().cpu()
    shape_str = ",".join(str(s) for s in tensor.shape)
    f.write(f"### {name} {shape_str}\n")

    flat = tensor.contiguous().view(-1).numpy()
    zero_streak = 0
    for val in flat:
        if val == 0.0:
            zero_streak += 1
            if zero_streak >= max_streak:
                break
        else:
            zero_streak = 0
        f.write(f"{val:.8f}\n")


import numpy as np
import torch

def write_inputs_npz(file_path, **tensors):
    """
    Save multiple tensors into an npz file.

    Args:
        file_path (str): Output .npz file path.
        tensors (dict): Any number of named PyTorch tensors as keyword arguments.

    Example:
        write_inputs_npz("inputs.npz", gmap=gmap, pyramid=pyramid, coords=coords, ii1=ii1, jj1=jj1)
    """
    data_dict = {}
    for name, tensor in tensors.items():
        cpu_tensor = tensor.detach().cpu()
        data_dict[name] = cpu_tensor.numpy()
    
    np.savez_compressed(file_path, **data_dict)

def dump_ba_inputs(
    path, poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, n, iters
):
    #create file if not exists
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)


    with open(path, "w") as f:
        write_tensor(f, "poses", poses)
        write_tensor(f, "patches", patches)
        write_tensor(f, "intrinsics", intrinsics)
        write_tensor(f, "target", target)
        write_tensor(f, "weight", weight)
        write_tensor(f, "lmbda", lmbda)
        write_tensor(f, "ii", ii)
        write_tensor(f, "jj", jj)
        write_tensor(f, "kk", kk)
        f.write(f"### t0\n{t0}\n")
        f.write(f"### n\n{n}\n")
        f.write(f"### iters\n{iters}\n")

def dump_ba_inputs_npz(
    path, poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, n,
):  
    """
    Save ba input tensors into a .npz file.

    Args:
        path (str): Path to the .npz file to write.
        poses (Tensor): Poses tensor of shape (B, N, 7)
        patches (Tensor): Patches tensor of shape (B, N * M, 3, P, P)
        intrinsics (Tensor): Intrinsics tensor of shape (B, N, 4)
        target (Tensor): Target tensor of shape (B, E, 2)
        weight (Tensor): Weight tensor of shape (B, E, 2)
        lmbda (Tensor): Lambda tensor of shape (B, E, 2)
        ii (Tensor): Indices tensor of shape (E,)
        jj (Tensor): Indices tensor of shape (E,)
        kk (Tensor): Indices tensor of shape (E,)
        t0 (int): Start frame index
        n (int): Number of frames
        iters (int): Number of iterations
    """

    # Create folder if not exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    data_dict = {
        "poses": poses.detach().cpu().numpy(),
        "patches": patches.detach().cpu().numpy(),
        "intrinsics": intrinsics.detach().cpu().numpy(),
        "target": target.detach().cpu().numpy(),
        "weight": weight.detach().cpu().numpy(),
        "lmbda": lmbda.detach().cpu().numpy(),
        "ii": ii.detach().cpu().numpy(),
        "jj": jj.detach().cpu().numpy(),
        "kk": kk.detach().cpu().numpy(),
        "t0": t0,
        "n": n,
    }
    np.savez_compressed(path, **data_dict)
    #print(f"[✓] Saved BA inputs to {path} (keys: {list(data_dict.keys())})")


def dump_extracted_coords_npz(
    path, coords,
):  
    """
    Save extracted coordinates tensor into a .npz file.

    Args:
        path (str): Path to the .npz file to write.
        coords (Tensor): Coordinates tensor of shape (B, N patches, 2)
    """

    # Create folder if not exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    data_dict = {
        "coords": coords.detach().cpu().numpy(),
    }
    np.savez_compressed(path, **data_dict)
    #print(f"[✓] Saved extracted coordinates to {path} (keys: {list(data_dict.keys())})")



def dump_corr_inputs(path, gmap, pyramid, pyr_index, coords, ii1, jj1):
    #create file if not exists
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        write_tensor(f, "gmap", gmap, max_streak=100)  # gmap is large, no zero streak
        write_tensor(f, "pyramid", pyramid[pyr_index], max_streak=100)
        write_tensor(f, "coords", coords, max_streak=100)
        write_tensor(f, "ii1", ii1, max_streak=100)
        write_tensor(f, "jj1", jj1, max_streak=100)

import os


def dump_corr_inputs_npz(path, gmap, pyramid, pyr_index, coords, ii1, jj1):
    """
    Save correlation input tensors into a .npz file.

    Args:
        path (str): Path to the .npz file to write.
        gmap (Tensor)
        pyramid (tuple of Tensors)
        pyr_index (int): Index to select from pyramid tuple.
        coords (Tensor)
        ii1 (Tensor)
        jj1 (Tensor)
    """

    # Create folder if not exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    data_dict = {
        "gmap": gmap.detach().cpu().numpy(),
        "pyramid": pyramid[pyr_index].detach().cpu().numpy(),
        "coords": coords.detach().cpu().numpy(),
        "ii1": ii1.detach().cpu().numpy(),
        "jj1": jj1.detach().cpu().numpy(),
    }

    np.savez_compressed(path, **data_dict)
    print(f"[✓] Saved corr inputs to {path} (keys: {list(data_dict.keys())})")


LOG=False
def log_stats(tensor, name, logname, path="test_randomness/full_runs/", ignore_zeros=False):
    """Minimal per-variable one-line stat logging: name mean std (optionally ignoring zeros)."""
    if not LOG:
        return
    if logname is None:
        return
    fname = f"{path}{logname}.txt"
    tensor_cpu = tensor.detach().float().cpu()

    if ignore_zeros:
        non_zero_values = tensor_cpu[tensor_cpu != 0]
        if non_zero_values.numel() > 0:
            mean = non_zero_values.mean().item()
            std = non_zero_values.std().item()
        else:
            mean = 0.0
            std = 0.0
    else:
        mean = tensor_cpu.mean().item()
        std = tensor_cpu.std().item()

    import os
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, "a") as f:
        f.write(f"{name} {mean:.6f} {std:.6f}\n")



def svd_se3_align(src_points, tgt_points):
    """
    src_points: [N, 3] predicted points
    tgt_points: [N, 3] target points
    Returns: R (3x3), t (3,)
    """

    src_mean = src_points.mean(dim=0, keepdim=True)
    tgt_mean = tgt_points.mean(dim=0, keepdim=True)

    src_centered = src_points - src_mean
    tgt_centered = tgt_points - tgt_mean

    H = src_centered.T @ tgt_centered
    U, _, Vt = torch.linalg.svd(H)
    R = Vt.T @ U.T

    # Handle reflection
    if torch.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = tgt_mean.squeeze() - R @ src_mean.squeeze()

    return R, t


def matrix_to_quaternion(R):
    """
    Convert a 3x3 rotation matrix to a normalized quaternion (x, y, z, w).
    """
    r = torch.zeros(4, device=R.device)

    trace = R.trace()
    if trace > 0.0:
        s = torch.sqrt(trace + 1.0) * 2
        r[3] = 0.25 * s
        r[0] = (R[2, 1] - R[1, 2]) / s
        r[1] = (R[0, 2] - R[2, 0]) / s
        r[2] = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        r[3] = (R[2, 1] - R[1, 2]) / s
        r[0] = 0.25 * s
        r[1] = (R[0, 1] + R[1, 0]) / s
        r[2] = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        r[3] = (R[0, 2] - R[2, 0]) / s
        r[0] = (R[0, 1] + R[1, 0]) / s
        r[1] = 0.25 * s
        r[2] = (R[1, 2] + R[2, 1]) / s
    else:
        s = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        r[3] = (R[1, 0] - R[0, 1]) / s
        r[0] = (R[0, 2] + R[2, 0]) / s
        r[1] = (R[1, 2] + R[2, 1]) / s
        r[2] = 0.25 * s

    return F.normalize(r, dim=0)


import torch
import torch.nn.functional as F

def normalize_quaternion(q):
    return F.normalize(q, dim=-1)

def quaternion_to_matrix(q):
    """
    Convert a normalized quaternion to a 3x3 rotation matrix.
    q: (..., 4) tensor in (x, y, z, w) order
    """
    x, y, z, w = q.unbind(-1)
    B = q.shape[:-1]
    R = torch.empty(*B, 3, 3, device=q.device, dtype=q.dtype)

    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    R[..., 0, 0] = 1 - 2 * (yy + zz)
    R[..., 0, 1] = 2 * (xy - wz)
    R[..., 0, 2] = 2 * (xz + wy)

    R[..., 1, 0] = 2 * (xy + wz)
    R[..., 1, 1] = 1 - 2 * (xx + zz)
    R[..., 1, 2] = 2 * (yz - wx)

    R[..., 2, 0] = 2 * (xz - wy)
    R[..., 2, 1] = 2 * (yz + wx)
    R[..., 2, 2] = 1 - 2 * (xx + yy)

    return R

def ba_solver_py(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations=5):
    """
    Drop-in Python replacement for fastba.BA

    poses: [1, N, 7]
    patches: [1, N * M, 3, P, P]
    intrinsics: [1, N, 4]
    target: [1, E, 2]
    weight: [1, E, 2]
    ii, jj, kk: [E]
    t0: start frame
    t1: end frame
    iterations: number of optimization steps
    """

    # Remove batch dim
    poses_opt = poses[0, t0:t1].clone().detach().requires_grad_(True)
    N = poses_opt.shape[0]

    M = patches.shape[1]
    P = patches.shape[-1]
    patches_flat = patches[0, :M]

    # Use first intrinsics row (assuming all the same)
    intrin = intrinsics[0, 0]
    fx, fy, cx, cy = intrin

    # Remove batch dim
    target = target[0]
    weight = weight[0]

    optimizer = torch.optim.Adam([poses_opt], lr=1e-2)

    for itr in range(iterations):
        optimizer.zero_grad()
        residuals = []

        for e in range(ii.shape[0]):
            idx_i = ii[e] - t0
            idx_j = jj[e] - t0
            idx_patch = kk[e]

            if idx_i < 0 or idx_j < 0 or idx_i >= N or idx_j >= N:
                continue

            ti = poses_opt[idx_i][:3]
            qi = poses_opt[idx_i][3:]
            Ri = quaternion_to_matrix(normalize_quaternion(qi))

            tj = poses_opt[idx_j][:3]
            qj = poses_opt[idx_j][3:]
            Rj = quaternion_to_matrix(normalize_quaternion(qj))

            # Patch center point
            x_c = patches_flat[idx_patch, 0, P // 2, P // 2]
            y_c = patches_flat[idx_patch, 1, P // 2, P // 2]
            d = patches_flat[idx_patch, 2, P // 2, P // 2]

            X_cam = torch.stack([(x_c - cx) / fx, (y_c - cy) / fy, torch.ones(1, device=poses.device)], dim=-1).squeeze()
            X_w = (Ri @ (d * X_cam)) + ti
            X_j = (Rj.T @ (X_w - tj))

            if X_j[2] <= 1e-3:
                continue  # avoid invalid depth

            x_proj = fx * (X_j[0] / X_j[2]) + cx
            y_proj = fy * (X_j[1] / X_j[2]) + cy

            rx = target[e, 0] - x_proj
            ry = target[e, 1] - y_proj

            wx = weight[e, 0]
            wy = weight[e, 1]

            res = wx * rx ** 2 + wy * ry ** 2
            residuals.append(res)

        if len(residuals) == 0:
            print("Warning: no valid residuals, skipping BA iteration")
            break

        loss = torch.stack(residuals).mean()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            poses_opt[:, 3:] = normalize_quaternion(poses_opt[:, 3:])

        print(f"BA iteration {itr}: loss = {loss.item():.6f}")

    # Prepare output to match original shape
    updated_poses = poses.clone()
    updated_poses[0, t0:t1] = poses_opt.detach()
    return updated_poses



def save_ba_debug_inputs_as_c_pair(
    poses, patches, intrinsics, target, weight, lmbda,
    ii, jj, kk, t0, t1,
    base_name="ba_in_fp32",
    zero_thresh=1e-6,
    min_nonzero_fraction=0.05
):
    header_path = f"{base_name}.h"
    source_path = f"{base_name}.c"

    def extern_decl(type_, name, size):
        return f"extern {type_} {name}[{size}];\n"

    def tensor_to_definitions(name, tensor, type_="float"):
        flat = tensor.contiguous().cpu().numpy()
        if name == "patches":
            flat = flat[:, :, 1, 1]  # center only
        flat = flat.reshape(-1)
        values = ', '.join(f'{v:.8f}' for v in flat)
        return f"{type_} {name}[{len(flat)}] = {{{values}}};\n", len(flat)

    def int_tensor_to_definitions(name, tensor, type_="uint16_t"):
        flat = tensor.contiguous().view(-1).cpu().numpy()
        values = ', '.join(str(int(v)) for v in flat)
        return f"{type_} {name}[{len(flat)}] = {{{values}}};\n", len(flat)

    def intrinsics_to_definition(tensor):
        flat = tensor.contiguous().view(-1).cpu().numpy()
        intr = flat[0:4]
        return f"float intrinsics[4] = {{{intr[0]:.8f}, {intr[1]:.8f}, {intr[2]:.8f}, {intr[3]:.8f}}};\n"

    def find_last_non_identity_pose(pose_tensor):
        poses_np = pose_tensor[0].cpu().numpy()
        identity = np.array([0, 0, 0, 0, 0, 0, 1])
        for i in reversed(range(poses_np.shape[0])):
            if not np.allclose(poses_np[i], identity, atol=1e-6):
                return i + 1
        return 1

    def find_first_allzero_patch(patch_tensor):
        patches_np = patch_tensor[0].cpu().numpy()[:, :, 1, 1]
        for i in range(patches_np.shape[0]):
            if np.all(np.abs(patches_np[i]) < zero_thresh):
                return i
        return patches_np.shape[0]

    # Trim inputs
    n_poses = find_last_non_identity_pose(poses)
    poses = poses[:, :n_poses, :]

    n_patches = find_first_allzero_patch(patches)
    patches = patches[:, :n_patches]

    mask = torch.logical_not(torch.logical_and(ii < t0, jj < t0))
    ii_trimmed = ii[mask]
    jj_trimmed = jj[mask]
    kk_trimmed = kk[mask]
    target_trimmed = target[:, mask, :]
    weight_trimmed = weight[:, mask, :]

    # Save header (.h)
    with open(header_path, "w") as h, open(source_path, "w") as c:
        h.write("// Auto-generated header declarations\n\n")
        c.write("// Auto-generated variable definitions\n\n")

        h.write(f"#pragma once\n\n")
        h.write("typedef unsigned short uint16_t;\n\n")
        c.write("typedef unsigned short uint16_t;\n\n")

        h.write(f"extern uint16_t num_poses;\n")
        c.write(f"uint16_t num_poses = {n_poses};\n\n")

        poses_def, poses_len = tensor_to_definitions("poses", poses)
        h.write(extern_decl("float", "poses", poses_len))
        c.write(poses_def + "\n")

        h.write(f"extern uint16_t num_patches;\n")
        c.write(f"uint16_t num_patches = {n_patches};\n\n")

        patches_def, patches_len = tensor_to_definitions("patches", patches[0])
        h.write(extern_decl("float", "patches", patches_len))
        c.write(patches_def + "\n")

        h.write(extern_decl("float", "intrinsics", 4))
        c.write(intrinsics_to_definition(intrinsics) + "\n")

        num_edges = ii_trimmed.shape[0]
        h.write(f"extern uint16_t num_edges;\n")
        c.write(f"uint16_t num_edges = {num_edges};\n\n")

        for name, data, is_int in [("target", target_trimmed, False), ("weights", weight_trimmed, False)]:
            def_fn = tensor_to_definitions if not is_int else int_tensor_to_definitions
            val_def, val_len = def_fn(name, data)
            h.write(extern_decl("float", name, val_len))
            c.write(val_def + "\n")

        h.write(f"extern float lambda;\n")
        c.write(f"float lambda = {float(lmbda):.8f};\n\n")

        for name, tensor in [("ii", ii_trimmed), ("jj", jj_trimmed), ("kk", kk_trimmed)]:
            val_def, val_len = int_tensor_to_definitions(name, tensor)
            h.write(extern_decl("uint16_t", name, val_len))
            c.write(val_def + "\n")

        h.write("extern uint16_t t0;\n")
        h.write("extern uint16_t t1;\n")
        c.write(f"uint16_t t0 = {t0};\n")
        c.write(f"uint16_t t1 = {t1};\n")

        kx, ku, _ = torch.unique(kk_trimmed, return_inverse=True, return_counts=True)
        h.write(f"extern uint16_t unique_patches;\n")
        c.write(f"uint16_t unique_patches = {kx.shape[0]};\n\n")

        ku_def, ku_len = int_tensor_to_definitions("ku", ku)
        kx_def, kx_len = int_tensor_to_definitions("kx", kx)

        h.write(extern_decl("uint16_t", "ku", ku_len))
        h.write(extern_decl("uint16_t", "kx", kx_len))

        c.write(ku_def + "\n")
        c.write(kx_def + "\n")

    print(f"[INFO] Saved header to: {header_path}")
    print(f"[INFO] Saved source to: {source_path}")
    return poses, patches


def save_golden_outputs_as_c_pair(poses, patches, base_name="golden_outputs"):
    header_path = f"{base_name}.h"
    source_path = f"{base_name}.c"

    def extern_decl(type_, name, size):
        return f"extern {type_} {name}[{size}];\n"

    def tensor_to_definitions(name, tensor, type_="float"):
        flat = tensor.contiguous().view(-1).cpu().numpy()
        values = ', '.join(f'{v:.8f}' for v in flat)
        return f"{type_} {name}[{len(flat)}] = {{{values}}};\n", len(flat)

    with open(header_path, "w") as h, open(source_path, "w") as c:
        h.write("// Golden output header declarations\n\n")
        c.write("// Golden output variable definitions\n\n")

        h.write("typedef unsigned short uint16_t;\n\n")
        c.write("typedef unsigned short uint16_t;\n\n")

        n_poses = poses.shape[1]
        n_patches = patches.shape[1]

        h.write("extern uint16_t golden_num_poses;\n")
        c.write(f"uint16_t golden_num_poses = {n_poses};\n\n")

        poses_def, poses_len = tensor_to_definitions("golden_poses", poses)
        h.write(extern_decl("float", "golden_poses", poses_len))
        c.write(poses_def + "\n")

        h.write("extern uint16_t golden_num_patches;\n")
        c.write(f"uint16_t golden_num_patches = {n_patches};\n\n")

        # === NEW: Extract center (x, y, disp) from 3x3x3 patches ===
        center_patches = patches[:, :, 1, 1, :]  # shape: [1, N, 3]

        patches_def, patches_len = tensor_to_definitions("golden_patches", center_patches[0])
        h.write(extern_decl("float", "golden_patches", patches_len))
        c.write(patches_def + "\n")

    print(f"[INFO] Saved golden header to: {header_path}")
    print(f"[INFO] Saved golden source to: {source_path}")
    print(f"[INFO] Golden Poses: {n_poses}, Golden Patches: {n_patches} (center only)")


def save_corr_io_as_c_pair(
    gmap, fmap, coords, ii, jj, corr, R: int,
    base_name: str = "corr_io_fp16_floats"
):
    """
    Save correlation IO to .h/.c using __fp16 (half precision) values
    written in normal float notation (e.g., 0.02545) instead of hex or int.
    """

    header_path = f"{base_name}.h"
    source_path = f"{base_name}.c"

    #gmap shape [1, buf, channel, 3, 3]
    #change order to [1, buf, 3, 3, channel], so save by channel order
    gmap = gmap.permute(0, 1, 3, 4, 2)  # [B, N, H, W, C] -> [B, N, 3, 3, C]
    
    #fmap shape [1, buf, channel, H2, W2]
    #change order to [1, buf, H2, W2, channel]
    fmap = fmap.permute(0, 1, 3, 4, 2)

    def as_fp16_float_flat(t: torch.Tensor) -> np.ndarray:
        # Convert to float16 for rounding, then back to float32 for printing
        return t.detach().contiguous().to(torch.float16).cpu().numpy().astype(np.float32).reshape(-1)

    def as_u16_flat(t: torch.Tensor) -> np.ndarray:
        return t.detach().contiguous().view(-1).cpu().to(torch.int).numpy().astype(np.uint16)

    def float_list(fvals: np.ndarray, per_line=16) -> str:
        parts = [f"{v:.8g}" for v in fvals.tolist()]  # shorten but keep precision
        lines = []
        for i in range(0, len(parts), per_line):
            lines.append(", ".join(parts[i:i+per_line]))
        return ",\n  ".join(lines)

    # ---- Shapes ----
    Bg, Ng, PgH, PgW, Cg = gmap.shape
    Bf, Bbuf, H2, W2, Cf = fmap.shape
    Bc, M, two, PcH, PcW  = coords.shape
    Bco, Mo, D1, D2, PoH, PoW = corr.shape

    assert Bg == 1 and Bf == 1 and Bc == 1 and Bco == 1
    assert two == 2
    assert PgH == PgW == PcH == PcW == PoH == PoW == 3
    assert Cg == Cf
    assert M == Mo == ii.numel() == jj.numel()

    C = Cg
    D = D1
    Nedges = M

    # ---- Flatten ----
    gmap_f   = as_fp16_float_flat(gmap)
    fmap_f   = as_fp16_float_flat(fmap)
    coords_f = as_fp16_float_flat(coords)
    corr_f   = as_fp16_float_flat(corr)
    ii_u16   = as_u16_flat(ii)
    jj_u16   = as_u16_flat(jj)

    with open(header_path, "w") as h, open(source_path, "w") as c:
        # ---------- Header ----------
        h.write("// Auto-generated correlation IO (__fp16 floats)\n\n")
        h.write("#pragma once\n#include <stdint.h>\n\n")
        h.write(f"#define MATCH_FEATURES_CHANNELS        {C}\n")
        h.write(f"#define MATCH_FEATURES_HEIGHT          {H2}\n")
        h.write(f"#define MATCH_FEATURES_WIDTH           {W2}\n")
        h.write(f"#define BUF_SIZE                     {Bbuf}\n")
        h.write(f"#define NEDGES   {Nedges}\n")
        h.write(f"#define RADIUS   {R}\n")
        h.write(f"#define D        {D}\n\n")

        h.write(f"extern const fp16 patches[{gmap_f.size}];\n")
        h.write(f"extern const fp16 match_features[{fmap_f.size}];\n")
        h.write(f"extern const fp16 coords[{coords_f.size}];\n")
        h.write(f"extern const uint16_t ii[{ii_u16.size}];\n")
        h.write(f"extern const uint16_t jj[{jj_u16.size}];\n")
        h.write(f"extern const fp16 correlation[{corr_f.size}];\n\n")

        # ---------- Source ----------
        c.write("// Auto-generated correlation IO (fp16 floats)\n\n")
        c.write(f"#include \"{base_name}.h\"\n\n")

        c.write(f"const fp16 patches[{gmap_f.size}] = {{\n  {float_list(gmap_f)}\n}};\n\n")
        c.write(f"const fp16 match_features[{fmap_f.size}] = {{\n  {float_list(fmap_f)}\n}};\n\n")
        c.write(f"const fp16 coords[{coords_f.size}] = {{\n  {float_list(coords_f)}\n}};\n\n")
        c.write(f"const uint16_t ii[{ii_u16.size}] = {{\n  {', '.join(map(str, ii_u16))}\n}};\n\n")
        c.write(f"const uint16_t jj[{jj_u16.size}] = {{\n  {', '.join(map(str, jj_u16))}\n}};\n\n")
        c.write(f"const fp16 corr_out[{corr_f.size}] = {{\n  {float_list(corr_f)}\n}};\n")

    print(f"[INFO] Saved header to: {header_path}")
    print(f"[INFO] Saved source to: {source_path}")

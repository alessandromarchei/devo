import math
import numpy as np
import torch
import torchvision

from devo.lietorch import SE3


######## TRANSFORM FUNCTIONS WITH ASPECT RATIO NOT MAINTAINED ########
def transform_rescale(scale, voxels, disps=None, poses=None, intrinsics=None, square=False):
    """
    Resize voxel and disparity grids to a fixed target resolution,
    and update poses and intrinsics accordingly.
    
    Args:
        target_size: tuple (new_height, new_width)
        voxels:      Tensor (B, C, H, W)
        disps:       Tensor (B, 1, H, W)
        poses:       Tensor (B, n_frames, 7)
        intrinsics:  Tensor (B, 4) [fx, fy, cx, cy]
        
    Returns:
        Rescaled voxels, disps, poses, intrinsics
    """
    H, W = voxels.shape[-2:]

    if (scale == 0.5 or scale == 0.25) and square:
        #SQUARE only if specified
        nW = math.floor(scale * W)  #with scale = 0.25 we get W = 160
        #making it not square, so H = W
        nH = nW
    else:
        nH, nW = math.floor(scale * H), math.floor(scale * W)


    resize = torchvision.transforms.Resize((nH, nW))

    # Compute separate scaling factors for height and width
    scale_y = nH / H
    scale_x = nW / W

    # Resize voxels ONLY IF THE SIZE IS not the original
    if voxels.shape[2] != nH:
        #only check about the height, since we are making it square
        voxels = resize(voxels)


    #resize disps if they are provided
    if disps is not None:
        disps = resize(disps)

    # Scale intrinsics per axis (fx, fy, cx, cy)
    if intrinsics is not None:
        fx, fy, cx, cy = intrinsics[..., 0], intrinsics[..., 1], intrinsics[..., 2], intrinsics[..., 3]
        fx = fx * scale_x
        fy = fy * scale_y
        cx = cx * scale_x
        cy = cy * scale_y
        intrinsics = torch.stack([fx, fy, cx, cy], dim=-1)

    # Scale translation part of poses
    if poses is not None:
        if not square or (scale_x == scale_y):
            poses = transform_rescale_poses(scale_x, poses)
        else:
            # Apply non-uniform scaling to the poses
            poses = transform_rescale_poses_xy(scale_x, scale_y, poses)

    return voxels, disps, poses, intrinsics

def transform_rescale_poses_xy(scale_x, scale_y, poses):
    """
    Rescale SE3 poses by applying non-uniform scaling to the X and Y translation components.
    Rotation remains unchanged.
    
    Args:
        scale_x: float
        scale_y: float
        poses: Tensor of shape (..., 7), representing [tx, ty, tz, qx, qy, qz, qw]
    
    Returns:
        Tensor of same shape (..., 7), with scaled translations
    """

    if scale_x == scale_y:
        # If the scale factors are equal, no need to apply non-uniform scaling
        s = torch.tensor(scale_x)
        poses = SE3(poses).scale(s).data
        return poses

    else:
        #NON UNIFORM SCALING, INTRODUCING AMBIGUITY in the Z axis
        poses_se3 = SE3(poses)
        t, q = poses_se3.data.split([3, 4], dim=-1)

        # Scale only X and Y components of the translation
        t[..., 0] *= scale_x
        t[..., 1] *= scale_y
        # Leave t[..., 2] (Z) unchanged

        poses_scaled = torch.cat([t, q], dim=-1)
    return poses_scaled


######## ORIGINAL FUNCTION, SCALE MAINTAINS ASPECT RATIO ########
# def transform_rescale(scale, voxels, disps=None, poses=None, intrinsics=None):
#     """Transform voxels/images, depth maps, poses and intrinsics (n_frames,*)"""
#     H, W = voxels.shape[-2:]
#     nH, nW = math.floor(scale * H), math.floor(scale * W)
#     resize = torchvision.transforms.Resize((nH, nW))

#     voxels = resize(voxels)
#     if disps is not None:
#         disps = resize(disps)
#     if poses is not None:
#         poses = transform_rescale_poses(scale, poses)
#     if intrinsics is not None:
#         intrinsics = scale * intrinsics

#     return voxels, disps, poses, intrinsics

def transform_rescale_poses(scale, poses):
    s = torch.tensor(scale)
    poses = SE3(poses).scale(s).data
    return poses



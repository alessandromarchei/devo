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

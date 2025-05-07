import os
import h5py
import hdf5plugin

import numpy as np
import argparse
import yaml
import configargparse

import torch

from torch import cuda
import matplotlib
import matplotlib.pyplot as plt


import os
import torch
import numpy as np
import cv2

import os
import torch
import numpy as np
import cv2

from tqdm import tqdm

def visualize_voxel(*voxel_in, EPS=1e-3, save=False, folder="results/voxels",
                       window_name="Voxel View", max_size=1800):
    colors = {
        -1: [0, 0, 255],     # red
         0: [255, 255, 255], # white
         1: [255, 0, 0]      # blue
    }

    all_images = []

    for i, vox in enumerate(voxel_in):
        if vox.device != 'cpu':
            vox = vox.detach().cpu()
        voxel = vox.clone().numpy()

        bins, H, W = voxel.shape
        out_images = []

        for b in range(bins):
            v = voxel[b]
            v[np.bitwise_and(v < EPS, v > 0)] = 0
            v[np.bitwise_and(v > -EPS, v < 0)] = 0
            v[v < 0] = -1
            v[v > 0] = 1

            img = np.zeros((H, W, 3), dtype=np.uint8)
            for val, color in colors.items():
                img[v == val] = color

            out_images.append(img)

        combined = np.hstack(out_images)
        all_images.append(combined)

    final_image = np.vstack(all_images)

    # Resize if too large for screen
    h, w = final_image.shape[:2]
    scale = min(max_size / h, max_size / w, 1.0)  # only downscale
    if scale < 1.0:
        final_image = cv2.resize(final_image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)

    if save:
        os.makedirs(folder, exist_ok=True)
        cv2.imwrite(os.path.join(folder, "debug.png"), final_image)
    else:
        cv2.imshow(window_name, final_image)
        cv2.waitKey(1)  # short pause, non-blocking

    return final_image


    if save:
        os.makedirs(folder, exist_ok=True)
        cv2.imwrite(os.path.join(folder, "debug.png"), final_image)
    else:
        cv2.imshow(window_name, final_image)
        cv2.waitKey(1)  # show for a short time (non-blocking)

    return final_image  # useful for logging or tests




def process_slice(events, idx0, idx1, config):
    xs = events["x"][idx0:idx1]
    ys = events["y"][idx0:idx1]
    ts = events["t"][idx0:idx1]
    ps = events["p"][idx0:idx1]
    return xs, ys, ts, ps



def to_event_stack(xs, ys, ts, ps, H=480, W=640, nb_of_time_bins=5):
    """Returns safe int8-encoded event stack: shape (nb_of_time_bins, H, W)"""

    device = 'cuda' if cuda.is_available() else 'cpu'

    if len(ts) == 0:
        return torch.zeros(nb_of_time_bins, H, W, dtype=torch.int8, device=device)

    ps = ps.astype(np.int8)
    ps[ps == 0] = -1

    duration = ts[-1] - ts[0]
    if duration == 0:
        return torch.zeros(nb_of_time_bins, H, W, dtype=torch.int8, device=device)

    # Bin assignment
    bin_indices = ((ts - ts[0]) * nb_of_time_bins / duration).astype(np.int32)
    bin_indices = np.clip(bin_indices, 0, nb_of_time_bins - 1)

    xs = np.clip(xs, 0, W - 1).astype(np.int32)
    ys = np.clip(ys, 0, H - 1).astype(np.int32)

    # Torch conversion
    bin_indices = torch.from_numpy(bin_indices).to(device)
    xs = torch.from_numpy(xs).to(device)
    ys = torch.from_numpy(ys).to(device)
    ps = torch.from_numpy(ps).to(device).int()  # accumulate as int32

    linear_idx = bin_indices * (H * W) + ys * W + xs
    voxel_grid_flat = torch.zeros(nb_of_time_bins * H * W, dtype=torch.int32, device=device)
    voxel_grid_flat.index_add_(0, linear_idx, ps)

    voxel_grid = voxel_grid_flat.view(nb_of_time_bins, H, W)
    
    # Clip and cast to int8
    voxel_grid = torch.clamp(voxel_grid, -127, 127).to(torch.int8)

    return voxel_grid



def to_voxel_grid(xs, ys, ts, ps, H=480, W=640, nb_of_time_bins=5):
    """Returns voxel grid representation of event steam. (5, H, W)

    In voxel grid representation, temporal dimension is
    discretized into "nb_of_time_bins" bins. The events fir
    polarities are interpolated between two near-by bins
    using bilinear interpolation and summed up.

    If event stream is empty, voxel grid will be empty.
    """

    if cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'


    voxel_grid = torch.zeros(nb_of_time_bins,
                          H,
                          W,
                          dtype=torch.float32)

    voxel_grid_flat = voxel_grid.flatten()
    ps = ps.astype(np.int8)
    ps[ps == 0] = -1

    # Convert timestamps to [0, nb_of_time_bins] range.
    duration = ts[-1] - ts[0]
    start_timestamp = ts[0]
    features = torch.from_numpy(np.stack([xs.astype(np.float32), ys.astype(np.float32), ts, ps], axis=1))
    x = features[:, 0]
    y = features[:, 1]
    polarity = features[:, 3].float()
    t = (features[:, 2] - start_timestamp) * (nb_of_time_bins - 1) / duration  # torch.float64
    t = t.to(torch.float64)


    left_t, right_t = t.floor(), t.floor()+1
    left_x, right_x = x.floor(), x.floor()+1
    left_y, right_y = y.floor(), y.floor()+1

    for lim_x in [left_x, right_x]:
        for lim_y in [left_y, right_y]:
            for lim_t in [left_t, right_t]:
                mask = (0 <= lim_x) & (0 <= lim_y) & (0 <= lim_t) & (lim_x <= W-1) \
                       & (lim_y <= H-1) & (lim_t <= nb_of_time_bins-1)

                # we cast to long here otherwise the mask is not computed correctly
                lin_idx = lim_x.long() \
                          + lim_y.long() * W \
                          + lim_t.long() * W * H

                lin_idx = lin_idx

                weight = (polarity * (1-(lim_x-x).abs()) * (1-(lim_y-y).abs()) * (1-(lim_t-t).abs()))
                voxel_grid_flat.index_add_(dim=0, index=lin_idx[mask], source=weight[mask].float())

    return voxel_grid


def binary_search_array(array, x, left=None, right=None, side="left"):
    """
    Binary search through a sorted array.
    """

    left = 0 if left is None else left
    right = len(array) - 1 if right is None else right
    mid = left + (right - left) // 2

    if left > right:
        return left if side == "left" else right

    if array[mid] == x:
        return mid

    if x < array[mid]:
        return binary_search_array(array, x, left=left, right=mid - 1)

    return binary_search_array(array, x, left=mid + 1, right=right)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5', type=str, default='/usr/scratch/badile43/amarchei/TartanEvent/office2/Easy/P010/events.h5', help="Path to .h5 event file")
    parser.add_argument('--config', type=str, default="h5_converter/h5.yml", help="Path to YAML config file")
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    assert os.path.exists(args.h5), f"File not found: {args.h5}"
    dirname = os.path.dirname(args.h5)
    event_frames_dir = os.path.join(dirname, str("event_frames"))
    event_voxel_dir = os.path.join(dirname, str("event_voxel"))
    os.makedirs(event_frames_dir, exist_ok=True)
    os.makedirs(event_voxel_dir, exist_ok=True)

    output_dir = event_voxel_dir if config["representation"] == "voxel" else event_frames_dir
    resolution = tuple(config["resolution"])
    num_bins = config["num_bins"]

    voxel_count = 0

    with h5py.File(args.h5, "r") as f:
        ts_all = f["events/t"]
        events = {k: f["events/" + k] for k in ("x", "y", "t", "p")}

        # Total duration in seconds
        duration_ns = int(ts_all[-1]) - int(ts_all[0])
        duration_sec = duration_ns / 1e9

        print(f"Scene: {args.h5}")
        print(f"Total events: {len(ts_all)}")
        print(f"Scene duration: {duration_sec:.2f} seconds")

        # Try to locate flow_left
        h5_dir = os.path.dirname(args.h5)
        flow_dir = os.path.join(h5_dir, "flow")
        if os.path.exists(flow_dir):
            flow_files = [f for f in os.listdir(flow_dir) if f.endswith('.npy')]
            print(f"Found {len(flow_files)} flow .npy files in flow/")
            print(f"=> Number of flow pairs: {len(flow_files) // 2}")
        else:
            print("flow_left/ directory not found.")

        if config.get("slicing_mode") == 'file':
            ts_filename = config["slicing_timestamps_txt"]
            slicing_txt_path = os.path.join(h5_dir, ts_filename)
            slicing_ts = np.loadtxt(slicing_txt_path, dtype=np.int64, converters=float)
            print(f"Using slicing timestamps from: {slicing_txt_path}")
            print(f"Number of slicing windows: {len(slicing_ts)-1}")

            for i in tqdm(range(len(slicing_ts) - 1), desc="Slicing by timestamps"):
                start, end = slicing_ts[i], slicing_ts[i + 1]
                idx0 = binary_search_array(ts_all, start)
                idx1 = binary_search_array(ts_all, end)
                xs, ys, ts, ps = process_slice(events, idx0, idx1, config)

                if config["representation"] == "voxel":
                    data = to_voxel_grid(xs, ys, ts, ps, resolution[0], resolution[1], num_bins)
                else:
                    data = to_event_stack(xs, ys, ts, ps, resolution[0], resolution[1], num_bins)

                outname = f"ev{i:04d}_{i+1:04d}.npy"
                if args.debug:
                    visualize_voxel(data)
                else:
                    np.save(os.path.join(output_dir, outname), data.cpu())
                voxel_count += 1

        elif config.get("slicing_mode") == 'number':
            nevents = config["events_per_slice"]
            total_events = len(ts_all)
            print(f"Using slicing by number of events: {nevents} per slice")

            for i, idx0 in enumerate(tqdm(range(0, total_events, nevents), desc="Slicing by number of events")):
                idx1 = min(idx0 + nevents, total_events)
                xs, ys, ts, ps = process_slice(events, idx0, idx1, config)

                if config["representation"] == "voxel":
                    data = to_voxel_grid(xs, ys, ts, ps, resolution[0], resolution[1], num_bins)
                else:
                    data = to_event_stack(xs, ys, ts, ps, resolution[0], resolution[1], num_bins)

                outname = f"ev{i:04d}_{i+1:04d}.npy"
                if args.debug:
                    visualize_voxel(data)
                else:
                    np.save(os.path.join(output_dir, outname), data.cpu())
                voxel_count += 1

    print(f"\nFinished preprocessing.")
    print(f"Total voxel grids created: {voxel_count}")
    print(f"Saved in: {output_dir}/")

if __name__ == "__main__":
    main()

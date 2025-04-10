import os
import pickle
import numpy as np
import h5py
import glob
import torch
import hdf5plugin


from utils.event_utils import to_voxel_grid, to_event_stack
from utils.transform_utils import transform_rescale, transform_rescale_poses
from utils.viz_utils import *


def get_scene_path(dataroot, scene, modality="image_left"):
    seq_name = scene.split("/")[0]
    seq_diffic = scene.split("/")[2]
    seq_num = scene.split("/")[3]
    sp = os.path.join(dataroot, seq_name, seq_diffic, modality, seq_name, seq_name, seq_diffic, seq_num)
    assert os.path.exists(sp), sp
    return sp

def save_scene_info(scene_info, name):
    cur_path = os.path.dirname(os.path.abspath(__file__))
    fgraph_dest_dir = os.path.join(cur_path, '../../fgraph')
    if not os.path.exists(fgraph_dest_dir):
        os.makedirs(fgraph_dest_dir)
    os.chmod(fgraph_dest_dir, 0o777)
    
    fname = os.path.join(fgraph_dest_dir, f'{name}.pickle')

    with open(fname, 'wb') as cachefile:
        pickle.dump((scene_info,), cachefile)
    os.chmod(fgraph_dest_dir, 0o555)

def is_converted(scene):
    return os.path.isfile(os.path.join(scene.replace('evs_left', 'image_left'), 'converted.txt'))
#
#def scene_in_split(scene, train_split, verbose=True, root=None):
#    if not any(x in scene for x in os.path.join(root, train_split)):
#        if verbose:
#            print(f"Not adding {scene}, since scene is not in requested split")
#        return False
#    else:
#        return True

def scene_in_split(scene_path, train_split, verbose=True):
    # Normalize and split the path
    parts = os.path.normpath(scene_path).split(os.sep)

    # Find the last occurrence of the known pattern (e.g., "Hard", "Easy")
    try:
        i = parts.index("Hard")  # or "Easy", or detect both
    except ValueError:
        try:
            i = parts.index("Easy")
        except ValueError:
            if verbose:
                print(f"[WARN] Could not find 'Hard' or 'Easy' in path: {scene_path}")
            return False

    # Extract the relative identifier: e.g. "office2/Hard/P002"
    scene_root = parts[i-1]
    rel_id = f"{scene_root}/" + "/".join(parts[i-1:i+2])

    #now copy the first name of the path until the / twice
    # 
    if rel_id in train_split:
        return True
    else:
        #print(f"Not adding {rel_id}, since it is not in requested split.") if verbose else None
        return False

def check_train_val_split(train, val, strict=True, name=None):
    assert len(train) > 0
    assert len(val) > 0
    if strict:
        assert len(set(train).intersection(set(val))) == 0
    else:
        intersect = set(train).intersection(set(val))
        for s in intersect:
            if name is None:
                print(f"\nWARNING: {s} is in both train and val split!!!\n")
            else:
                print(f"\nWARNING: {s} is in both train and {name}-val split!!!\n")

def load_splitfile(splitfile):
    with open(splitfile, 'r') as f:
        split = f.read().split()
    assert len(split) > 0
    return split


def seqs_in_scene_info(split, scene_info):
    splits_in_sinfo = True

    if split is not None:
        for seq in split:
            sum = np.sum([seq in sinf for sinf in scene_info.keys()])
            if sum == 0:
                print(f"Sequence {seq} not in scene_info")
                splits_in_sinfo = False
                break
            else:
                assert sum == 1

    return splits_in_sinfo



# ##### DEEP COPY OF THE H5 FILE #####
# def h5_to_voxels(scenedir,nbins=5, H=640, W=480, use_event_stack=False):


#     #if not '.h5' in the name, add it
#     if not '.h5' in scenedir:
#         h5in = glob.glob(os.path.join(scenedir, f"*.h5"))
#         root_path = scenedir
#     else:
#         h5in = [scenedir] 
#         #remove the last part of the path, until /*.h5
#         root_path = os.path.dirname(scenedir)
    
#     assert len(h5in) == 1
#     datain = h5py.File(h5in[0], 'r')

#     img_timestamps = glob.glob(os.path.join(root_path, "timestamps.txt"))
#     assert len(img_timestamps) == 1

#     #load the timestamps from the text file, they are in nanoseconds, floating point
#     tss_imgs_ns = np.loadtxt(img_timestamps[0])
#     num_imgs = tss_imgs_ns.shape[0]


#     #the content of the h5 file this

#     #    Dataset: events/p
#     #  Shape: (1340866965,), Type: int8
#     #Dataset: events/t
#     #  Shape: (1340866965,), Type: int64
#     #Dataset: events/width
#     #  Shape: (), Type: int32
#     #Dataset: events/x
#     #  Shape: (1340866965,), Type: uint16
#     #Dataset: events/y
#     #  Shape: (1340866965,), Type: uint16

#     x = datain["events"]["x"][:]
#     y = datain["events"]["y"][:]
#     p = datain["events"]["p"][:]
#     t_ns = datain["events"]["t"][:] #in nanoseconds

#     p = np.where(p == 0, -1, p)

#     #build the event indices, so the events that have the closest timestamp to the image timestamp

#     # Ensure image timestamps are in the same unit (ns)
#     if tss_imgs_ns.max() < 1e6:
#         tss_imgs_ns = tss_imgs_ns * 1e9  # convert to ns

#     # Find the index of the event closest to each image timestamp
#     event_idxs = np.searchsorted(t_ns, tss_imgs_ns, side='left')
#     event_idxs = np.clip(event_idxs, 0, len(t_ns) - 1)

#     with open("indexes_full_batch.txt", "w") as f:
#         for idx in event_idxs:
#             f.write(f"{idx}\n")

#     evidx_left = 0
#     data_list = []
#     for img_i in range(num_imgs):        
#         #take the event id between the current image and the next image
#         evid_nextimg = event_idxs[img_i]

#         #take the events between the current image and the next image
#         x_batch = x[evidx_left:evid_nextimg][:]
#         y_batch = y[evidx_left:evid_nextimg][:]
#         p_batch = p[evidx_left:evid_nextimg][:]
#         t_batch = t_ns[evidx_left:evid_nextimg][:]

#         #update the event id for the next image
#         evidx_left = evid_nextimg

#         #rectify the events of the batch
#         if len(x_batch) == 0:
#             voxel = np.zeros((nbins, H, W), dtype=np.float32)
#             continue
#         #create the voxel grid, with 5 channels
#         if not use_event_stack : 
#             voxel = to_voxel_grid(xs=x_batch, ys=y_batch, ts=t_batch, ps=p_batch, H=H, W=W, nb_of_time_bins=nbins)
#         else : 
#             voxel = to_event_stack(xs=x_batch, ys=y_batch, ts=t_batch, ps=p_batch, H=H, W=W, nb_of_time_bins=nbins)

#         data_list.append(voxel)


#     datain.close()

#     return data_list


def h5_to_voxels(scenedir, nbins=5, H=640, W=480, use_event_stack=False):

    if not '.h5' in scenedir:
        h5in = glob.glob(os.path.join(scenedir, f"*.h5"))
        root_path = scenedir
    else:
        h5in = [scenedir]
        root_path = os.path.dirname(scenedir)

    assert len(h5in) == 1
    datain = h5py.File(h5in[0], 'r')

    # Load image timestamps
    img_timestamps = glob.glob(os.path.join(root_path, "timestamps.txt"))
    assert len(img_timestamps) == 1
    tss_imgs_ns = np.loadtxt(img_timestamps[0])
    if tss_imgs_ns.max() < 1e6:
        tss_imgs_ns *= 1e9  # ensure nanoseconds

    # Load all events fully into memory
    x = datain["events"]["x"][:]
    y = datain["events"]["y"][:]
    p = datain["events"]["p"][:]
    t_ns = datain["events"]["t"][:]

    p = np.where(p == 0, -1, p)

    total_events = len(t_ns)
    num_imgs = tss_imgs_ns.shape[0]

    # Compute event indexes for each timestamp
    event_idxs = np.searchsorted(t_ns, tss_imgs_ns, side='left')
    event_idxs = np.clip(event_idxs, 0, total_events - 1)

    with open("indexes_full_batch.txt", "w") as f:
        for idx in event_idxs:
            f.write(f"{idx}\n")

    # If no indexes provided, default to all image frames
    indexes = list(range(num_imgs))

    # Extract voxel batches
    voxel_list = []
    for i in indexes:
        start_idx = event_idxs[i - 1] if i > 0 else 0
        end_idx = event_idxs[i]

        x_batch = x[start_idx:end_idx]
        y_batch = y[start_idx:end_idx]
        p_batch = p[start_idx:end_idx]
        t_batch = t_ns[start_idx:end_idx]

        if len(x_batch) == 0:
            voxel = np.zeros((nbins, H, W), dtype=np.float32)
        else:
            voxel_fn = to_event_stack if use_event_stack else to_voxel_grid
            voxel = voxel_fn(xs=x_batch, ys=y_batch, ts=t_batch, ps=p_batch,
                             H=H, W=W, nb_of_time_bins=nbins)

        voxel_list.append(voxel)

    datain.close()
    return voxel_list


def h5_to_voxels_indexed(scenedir, nbins=5, H=640, W=480, use_event_stack=False, indexes=None, chunk_size=10000000):
    assert indexes is not None and len(indexes) > 0

    if not scenedir.endswith('.h5'):
        h5in = glob.glob(os.path.join(scenedir, "*.h5"))
        assert len(h5in) == 1
        h5_path = h5in[0]
        root_path = scenedir
    else:
        h5_path = scenedir
        root_path = os.path.dirname(scenedir)

    datain = h5py.File(h5_path, 'r')
    x_ds = datain["events"]["x"]
    y_ds = datain["events"]["y"]
    p_ds = datain["events"]["p"]
    t_ds = datain["events"]["t"]

    # Try to load timestamps.txt
    ts_path = os.path.join(root_path, "timestamps.txt")
    try:
        tss_imgs_ns = np.loadtxt(ts_path)
        if tss_imgs_ns.size == 0:
            raise ValueError("Empty timestamps file.")
    except Exception:
        # Fall back to counting images in images_left/
        images_folder = os.path.join(root_path, "image_left")
        try:
            all_files = sorted(os.listdir(images_folder))
            num_images = len(all_files)
            if num_images < 2:
                raise ValueError(f"Found too few images ({num_images}) in {images_folder}")
            N_synthetic = num_images - 1
            spacing_ns = 33_333_333.33  # 33.33ms in nanoseconds
            tss_imgs_ns = np.arange(N_synthetic) * spacing_ns
            print(f"[WARN] Using synthetic timestamps for {root_path} (timestamps.txt missing or empty, based on {num_images} images)")
        except Exception as e:
            print(f"[ERROR] Cannot synthesize timestamps from {images_folder}: {e}")
            raise RuntimeError(f"Could not get timestamps for scene {root_path}")

    if tss_imgs_ns.max() < 1e6:
        tss_imgs_ns *= 1e9

    # Preallocate result array
    event_idxs = np.zeros(len(tss_imgs_ns), dtype=np.int64)

    # Sort image timestamps for faster lookup
    sorted_indices = np.argsort(tss_imgs_ns)
    sorted_timestamps = tss_imgs_ns[sorted_indices]

    # Accumulate and merge searchsorted results across chunks
    total_events = len(t_ds)
    current_offset = 0
    ts_idx = 0

    while current_offset < total_events and ts_idx < len(sorted_timestamps):
        chunk_end = min(current_offset + chunk_size, total_events)
        t_chunk = t_ds[current_offset:chunk_end]

        ts_remaining = sorted_timestamps[ts_idx:]
        idxs_in_chunk = np.searchsorted(t_chunk, ts_remaining, side="left")

        valid = idxs_in_chunk < (chunk_end - current_offset)
        count_valid = valid.sum()

        global_idxs = current_offset + idxs_in_chunk[:count_valid]
        event_idxs[sorted_indices[ts_idx:ts_idx + count_valid]] = global_idxs

        ts_idx += count_valid
        current_offset = chunk_end

    while ts_idx < len(sorted_timestamps):
        event_idxs[sorted_indices[ts_idx]] = total_events - 1
        ts_idx += 1

    event_idxs = np.clip(event_idxs, 0, total_events - 1)

    with open("indexes.txt", "w") as f:
        for idx in event_idxs:
            f.write(f"{idx}\n")

    # Build voxels
    voxel_list = []
    for i in indexes:
        assert 0 <= i < len(event_idxs)
        start_idx = event_idxs[i - 1] if i > 0 else 0
        end_idx = event_idxs[i]

        x_batch = x_ds[start_idx:end_idx]
        y_batch = y_ds[start_idx:end_idx]
        p_batch = p_ds[start_idx:end_idx]
        t_batch = t_ds[start_idx:end_idx]
        p_batch = np.where(p_batch == 0, -1, p_batch)

        if len(x_batch) == 0:
            voxel = np.zeros((nbins, H, W), dtype=np.float32)
        else:
            if not use_event_stack:
                voxel = to_voxel_grid(xs=x_batch, ys=y_batch, ts=t_batch, ps=p_batch,
                                      H=H, W=W, nb_of_time_bins=nbins)
            else:
                voxel = to_event_stack(xs=x_batch, ys=y_batch, ts=t_batch, ps=p_batch,
                                       H=H, W=W, nb_of_time_bins=nbins)
        voxel_list.append(voxel)

    datain.close()
    return voxel_list



# ##### BATCHED VERSION OF THE H5 FILE #####
# def h5_to_voxels(scenedir, nbins=5, H=480, W=640, use_event_stack=False, batch_size=5, scale=1.0):
#     """
#     Efficiently loads an HDF5 event file and converts events to voxels per image frame.
#     Uses batched access to avoid loading the full file into memory.

#     Parameters:
#         scenedir (str): Path to the HDF5 file or directory.
#         nbins (int): Number of temporal bins in the voxel grid.
#         H, W (int): Sensor resolution.
#         use_event_stack (bool): Whether to use an event stack instead of voxel grid.
#         batch_size (int): Number of images to process per batch.

#     Returns:
#         List[np.ndarray]: List of voxel/event stack arrays.
#     """
    
#     # Determine .h5 file path
#     if not scenedir.endswith('.h5'):
#         h5in = glob.glob(os.path.join(scenedir, f"*.h5"))
#         assert len(h5in) == 1, f"Found {len(h5in)} HDF5 files in {scenedir}, expected 1."
#         h5_path = h5in[0]
#         root_path = scenedir
#     else:
#         h5_path = scenedir
#         root_path = os.path.dirname(scenedir)

#     # Load image timestamps
#     img_timestamps_files = glob.glob(os.path.join(root_path, "timestamps.txt"))
#     assert len(img_timestamps_files) == 1, "Missing timestamps.txt"
#     tss_imgs_ns = np.loadtxt(img_timestamps_files[0])  # shape (num_imgs,)
#     if tss_imgs_ns.max() < 1e6:
#         tss_imgs_ns = tss_imgs_ns * 1e9  # convert to ns if needed

#     data_list = []

#     with h5py.File(h5_path, 'r') as datain:
#         # Use lazy datasets
#         x = datain["events"]["x"]
#         y = datain["events"]["y"]
#         p = datain["events"]["p"]
#         t = datain["events"]["t"]

#         num_events = t.shape[0]
#         num_imgs = tss_imgs_ns.shape[0]

#         # For performance, load timestamps into RAM only (if they fit)
#         if num_events < 1e9:
#             t_arr = t[:]
#         else:
#             print("Large file detected, streaming timestamps...")
#             t_arr = None

#         # Compute event indices for each image timestamp
#         if t_arr is not None:
#             event_idxs = np.searchsorted(t_arr, tss_imgs_ns, side='left')
#         else:
#             event_idxs = []
#             for ts in tss_imgs_ns:
#                 # Manual binary search on lazy dataset
#                 lo, hi = 0, num_events - 1
#                 while lo < hi:
#                     mid = (lo + hi) // 2
#                     if t[mid] < ts:
#                         lo = mid + 1
#                     else:
#                         hi = mid
#                 event_idxs.append(lo)
#             event_idxs = np.clip(event_idxs, 0, num_events - 1)

#         # Batched processing
#         evidx_left = 0
#         for i in range(0, num_imgs, batch_size):
#             i_end = min(i + batch_size, num_imgs)
#             idx_start = evidx_left
#             idx_end = event_idxs[i_end - 1]  # last image in the batch

#             # Stream only necessary event slices
#             x_batch = x[idx_start:idx_end]
#             y_batch = y[idx_start:idx_end]
#             p_batch = p[idx_start:idx_end]
#             t_batch = t[idx_start:idx_end]

#             # polarity fix
#             p_batch = np.where(p_batch == 0, -1, p_batch)

#             # Loop through images in this batch
#             for j in range(i, i_end):
#                 ev_end = event_idxs[j]
#                 if evidx_left >= ev_end:
#                     continue

#                 x_slice = x_batch[evidx_left - idx_start : ev_end - idx_start]
#                 y_slice = y_batch[evidx_left - idx_start : ev_end - idx_start]
#                 p_slice = p_batch[evidx_left - idx_start : ev_end - idx_start]
#                 t_slice = t_batch[evidx_left - idx_start : ev_end - idx_start]

#                 if len(x_slice) == 0:
#                     evidx_left = ev_end
#                     continue

#                 if not use_event_stack:
#                     voxel = to_voxel_grid(xs=x_slice, ys=y_slice, ts=t_slice, ps=p_slice, H=H, W=W, nb_of_time_bins=nbins)
#                 else:
#                     voxel = to_event_stack(xs=x_slice, ys=y_slice, ts=t_slice, ps=p_slice, H=H, W=W, nb_of_time_bins=nbins)

#                 #here rescale the voxel in case the resolution is not 640x480
#                 if scale != 1.0:
#                     #hacky way for our 160x160 resolution
#                     voxel,_,_,_ = transform_rescale(scale, voxel)

#                 data_list.append(voxel)
#                 evidx_left = ev_end

#     return data_list


def h5_to_voxels_limit(scenedir, nbins=5, H=640, W=480, use_event_stack=False, max_events_loaded=1000000):
    import os.path as osp
    import glob
    import h5py
    import numpy as np

    # Load .h5 file
    if not scenedir.endswith('.h5'):
        h5in = glob.glob(os.path.join(scenedir, "*.h5"))
        root_path = scenedir
    else:
        h5in = [scenedir]
        root_path = os.path.dirname(scenedir)

    assert len(h5in) == 1, f"Expected 1 .h5 file, found {len(h5in)}"
    datain = h5py.File(h5in[0], 'r')

    # Load or synthesize timestamps
    ts_path = os.path.join(root_path, "timestamps.txt")
    try:
        tss_imgs_ns = np.loadtxt(ts_path)
        if tss_imgs_ns.size == 0:
            raise ValueError("Empty timestamps file.")
    except Exception:
        # Fallback: count number of images
        images_folder = os.path.join(root_path, "images_left")
        try:
            all_files = sorted(os.listdir(images_folder))
            num_images = len(all_files)
            if num_images < 2:
                raise ValueError(f"Found too few images ({num_images}) in {images_folder}")
            N_synthetic = num_images - 1
            spacing_ns = 33_333_333.33  # 33.33ms in nanoseconds
            tss_imgs_ns = np.arange(N_synthetic) * spacing_ns
            print(f"[WARN] Using synthetic timestamps for {root_path} (timestamps.txt missing or empty, based on {num_images} images)")
        except Exception as e:
            print(f"[ERROR] Cannot synthesize timestamps from {images_folder}: {e}")
            raise RuntimeError(f"Could not get timestamps for scene {root_path}")

    if tss_imgs_ns.max() < 1e6:
        tss_imgs_ns *= 1e9  # ensure timestamps are in ns

    # Get total number of events
    total_events = datain["events"]["x"].shape[0]
    max_events = min(max_events_loaded, total_events)

    # Load only the first N events
    x = datain["events"]["x"][:max_events]
    y = datain["events"]["y"][:max_events]
    p = datain["events"]["p"][:max_events]
    t_us = datain["events"]["t"][:max_events]
    t = t_us * 1  # convert to ns
    p[p == 0] = -1

    # Build event indices for voxel grid timestamps
    event_idxs = np.searchsorted(t, tss_imgs_ns, side='left')
    event_idxs = np.clip(event_idxs, 0, max_events)

    # Only keep timestamps that fall within the loaded events
    valid_idx_mask = event_idxs < max_events
    event_idxs = event_idxs[valid_idx_mask]
    tss_imgs_ns = tss_imgs_ns[valid_idx_mask]

    evidx_left = 0
    data_list = []

    for img_i in range(len(event_idxs)):
        evid_nextimg = event_idxs[img_i]

        x_batch = x[evidx_left:evid_nextimg]
        y_batch = y[evidx_left:evid_nextimg]
        p_batch = p[evidx_left:evid_nextimg]
        t_batch = t[evidx_left:evid_nextimg]

        evidx_left = evid_nextimg

        if len(x_batch) == 0:
            continue

        if not use_event_stack:
            voxel = to_voxel_grid(xs=x_batch, ys=y_batch, ts=t_batch, ps=p_batch, H=H, W=W, nb_of_time_bins=nbins)
        else:
            voxel = to_event_stack(xs=x_batch, ys=y_batch, ts=t_batch, ps=p_batch, H=H, W=W, nb_of_time_bins=nbins)

        data_list.append(voxel)

    datain.close()
    return data_list




# def h5_to_voxels_indexed(scenedir, nbins=5, H=640, W=480, use_event_stack=False, indexes=None, chunk_size=1000000):
#     import os, glob, h5py, numpy as np
#     from bisect import bisect_left
    
#     assert indexes is not None and len(indexes) > 0
    
#     if not scenedir.endswith('.h5'):
#         h5in = glob.glob(os.path.join(scenedir, f"*.h5"))
#         assert len(h5in) == 1, "Expected exactly one .h5 file"
#         h5_path = h5in[0]
#         root_path = scenedir
#     else:
#         h5_path = scenedir
#         root_path = os.path.dirname(scenedir)
    
#     datain = h5py.File(h5_path, 'r')
#     x_ds = datain["events"]["x"]
#     y_ds = datain["events"]["y"]
#     p_ds = datain["events"]["p"]
#     t_ds = datain["events"]["t"]
    
#     # Load image timestamps
#     tss_imgs_ns = np.loadtxt(os.path.join(root_path, "timestamps.txt"))
#     if tss_imgs_ns.max() < 1e6:
#         tss_imgs_ns *= 1e9  # convert to nanoseconds
    
#     # Find event indices corresponding to image timestamps using chunked processing
#     total_events = len(t_ds)
#     event_idxs = []
    
#     for ts in tss_imgs_ns:
#         # Binary search through chunks to find the right index
#         left, right = 0, total_events
#         while right - left > chunk_size:
#             mid = (left + right) // 2
#             if t_ds[mid] < ts:
#                 left = mid
#             else:
#                 right = mid
        
#         # Process the final chunk to find the exact index
#         chunk = t_ds[left:min(right, total_events)]
#         relative_idx = bisect_left(chunk, ts)
#         event_idxs.append(left + relative_idx)
    
#     event_idxs = np.array(event_idxs)
#     event_idxs = np.clip(event_idxs, 0, total_events - 1)
#     with open("indexes.txt", "w") as f:
#         for idx in event_idxs:
#             f.write(f"{idx}\n")
    
#     voxel_list = []
#     for i in indexes:
#         assert 0 <= i < len(event_idxs)
#         start_idx = event_idxs[i]
#         end_idx = event_idxs[i+1] if i+1 < len(event_idxs) else total_events
        
#         # Lazy slicing!
#         x_batch = x_ds[start_idx:end_idx]
#         y_batch = y_ds[start_idx:end_idx]
#         p_batch = p_ds[start_idx:end_idx]
#         t_batch = t_ds[start_idx:end_idx]
        
#         # Polarity conversion
#         p_batch = np.where(p_batch == 0, -1, p_batch)
        
#         if len(x_batch) == 0:
#             voxel = np.zeros((nbins, H, W), dtype=np.float32)
#         else:
#             if not use_event_stack:
#                 voxel = to_voxel_grid(xs=x_batch, ys=y_batch, ts=t_batch, ps=p_batch, 
#                                      H=H, W=W, nb_of_time_bins=nbins)
#             else:
#                 voxel = to_event_stack(xs=x_batch, ys=y_batch, ts=t_batch, ps=p_batch, 
#                                       H=H, W=W, nb_of_time_bins=nbins)


#         voxel_list.append(voxel)
#         # visualize_voxel(voxel)
    
#     datain.close()
#     return voxel_list


import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import csv
import os
import cv2
import math
import random
import json
import pickle
import os.path as osp
import h5py
import glob
import time

from .augmentation import RGBDAugmentor, EVSDAugmentor
from .rgbd_utils import *
from .utils import seqs_in_scene_info, save_scene_info, load_splitfile, check_train_val_split
from utils.load_utils import voxel_read
from utils.transform_utils import transform_rescale, transform_rescale_poses

from ..lietorch import SE3
from .utils import is_converted, scene_in_split, h5_to_voxels_indexed, h5_to_voxels
from utils.viz_utils import *

class RGBDDataset(data.Dataset):
    def __init__(self, name, datapath, n_frames=4, crop_size=[480,640], fmin=10.0, fmax=75.0, aug=True, sample=True,
                 fgraph_pickle=None, train_split=None, val_split=None, strict_split=True, return_fname=False, scale=1.0):
        """ Base class for RGBD dataset """
        self.aug = None
        self.root = datapath
        self.name = name
        self.return_fname = return_fname

        self.aug = aug
        self.sample = sample
        self.scale = scale

        self.n_frames = n_frames
        self.fmin = fmin # exclude very easy examples
        self.fmax = fmax # exclude very hard examples

        self.train_split = load_splitfile(train_split)
        val_split = load_splitfile(val_split) # do not need val split later on
        check_train_val_split(self.train_split, val_split, strict=strict_split)

        if self.aug:
            if self.scale != 1.0:
                crop_size = np.floor(self.scale * np.array(crop_size)).astype(int).tolist()
            self.aug = RGBDAugmentor(crop_size=crop_size)

        # building dataset is expensive, cache so only needs to be performed once
        if fgraph_pickle is None or fgraph_pickle == '':
            scene_info = self._build_dataset()
            save_scene_info(scene_info, name)
        else:
            scene_info = pickle.load(open(fgraph_pickle, 'rb'))[0]
            if not seqs_in_scene_info(self.train_split, scene_info):
                print(f"Loaded scene_info {fgraph_pickle} does NOT contain all requested scenes. Rebuilding dataset...")
                scene_info = self._build_dataset()
                save_scene_info(scene_info, name)

        self.scene_info = scene_info
        self._build_dataset_index()
                
    def _build_dataset_index(self):
        self.dataset_index = []  # is list of (scene, frameIndex)
        for scene in self.scene_info:
            graph = self.scene_info[scene]['graph']
            for i in graph:
                if i < len(graph) - self.n_frames:
                    self.dataset_index.append((scene, i))

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

    @staticmethod
    def depth_read(depth_file):
        return np.load(depth_file)

    def build_frame_graph(self, poses, depths, intrinsics, f=16, max_flow=256):
        """ compute optical flow distance between all pairs of frames """
        def read_disp(fn):
            depth = self.__class__.depth_read(fn)[f//2::f, f//2::f]
            depth[depth < 0.01] = np.mean(depth)
            return 1.0 / depth

        poses = np.array(poses)
        intrinsics = np.array(intrinsics) / f
        
        disps = np.stack(list(map(read_disp, depths)), 0)
        d = f * compute_distance_matrix_flow(poses, disps, intrinsics)
        
        # uncomment for nice visualization
        # import matplotlib.pyplot as plt
        # plt.imshow(d)
        # plt.show()

        graph = {}
        for i in range(d.shape[0]):
            j, = np.where(d[i] < max_flow)
            graph[i] = (j, d[i,j])

        return graph

    def __getitem__(self, index):
        """ return training video """

        index = index % len(self.dataset_index)
        scene_id, ix = self.dataset_index[index]

        frame_graph = self.scene_info[scene_id]['graph']
        images_list = self.scene_info[scene_id]['images']
        depths_list = self.scene_info[scene_id]['depths']
        poses_list = self.scene_info[scene_id]['poses']
        intrinsics_list = self.scene_info[scene_id]['intrinsics']

        # stride = np.random.choice([1,2,3])

        d = np.random.uniform(self.fmin, self.fmax)
        s = 1

        inds = [ ix ]

        while len(inds) < self.n_frames:
            # get other frames within flow threshold
            if self.sample:
                k = (frame_graph[ix][1] > self.fmin) & (frame_graph[ix][1] < self.fmax)
                frames = frame_graph[ix][0][k]

                # prefer frames forward in time
                if np.count_nonzero(frames[frames > ix]):
                    ix = np.random.choice(frames[frames > ix])

                elif ix + 1 < len(images_list):
                    ix = ix + 1 # prefer forward-time over flow < max_flow

                elif np.count_nonzero(frames):
                    ix = np.random.choice(frames)

            else:
                i = frame_graph[ix][0].copy()
                g = frame_graph[ix][1].copy()

                g[g > d] = -1
                if s > 0:
                    g[i <= ix] = -1
                else:
                    g[i >= ix] = -1

                if len(g) > 0 and np.max(g) > 0:
                    ix = i[np.argmax(g)]
                else:
                    if ix + s >= len(images_list) or ix + s < 0:
                        s *= -1

                    ix = ix + s
            
            inds += [ ix ]


        images, depths, poses, intrinsics = [], [], [], []
        for i in inds:
            images.append(self.__class__.image_read(images_list[i]))
            depths.append(self.__class__.depth_read(depths_list[i]))
            poses.append(poses_list[i])
            intrinsics.append(intrinsics_list[i])

        images = np.stack(images).astype(np.float32)
        depths = np.stack(depths).astype(np.float32)
        poses = np.stack(poses).astype(np.float32)
        intrinsics = np.stack(intrinsics).astype(np.float32)

        images = torch.from_numpy(images).float()
        images = images.permute(0, 3, 1, 2) # (n_frames, 480, 640, 3) => (n_frames, 3, 480, 640)

        disps = torch.from_numpy(1.0 / depths)
        poses = torch.from_numpy(poses)
        intrinsics = torch.from_numpy(intrinsics)

        if self.scale != 1.0:
            images, disps, poses, intrinsics = transform_rescale(self.scale, images, disps, poses, intrinsics)

        if self.aug:
            images, poses, disps, intrinsics = \
                self.aug(images, poses, disps, intrinsics)

        # normalize depth
        s = .7 * torch.quantile(disps, .98)
        disps = disps / s
        poses[...,:3] *= s

        if not self.return_fname:
            return images, poses, disps, intrinsics
        else:
            return images, poses, disps, intrinsics, scene_id

    def __len__(self):
        return len(self.dataset_index)

    def __imul__(self, x):
        self.dataset_index *= x
        return self


class EVSDDataset(data.Dataset):
    def __init__(self, name, datapath, n_frames=4, crop_size=[480,640], fmin=10.0, fmax=75.0, aug=True, sample=True,
                 fgraph_pickle=None, train_split=None, val_split=None, strict_split=True, return_fname=False, scale=1.0, args=None):
        """ Base class for Events + Depth dataset """
        self.aug = None
        self.root = datapath
        self.name = name
        self.return_fname = return_fname

        self.aug = aug
        self.sample = sample
        self.scale = scale
        self.args = args

        self.n_frames = n_frames
        self.fmin = fmin # exclude very easy examples
        self.fmax = fmax # exclude very hard examples

        self.train_split = load_splitfile(train_split)
        val_split = load_splitfile(val_split) # do not need val split later on
        check_train_val_split(self.train_split, val_split, strict=strict_split)

        if (scale == 0.25 or scale == 0.5) and self.args.square:
            #hacky way of modifying the crop size to be squared instead of 640x480
            crop_size = [640, 640]
            #now when performing augmentation, the crop size is 640x640

        if self.aug:
            if self.scale != 1.0:
                crop_size = np.floor(self.scale * np.array(crop_size)).astype(int).tolist()
            self.aug = EVSDAugmentor(crop_size=crop_size)

        # building dataset is expensive, cache so only needs to be performed once
        if fgraph_pickle is None or fgraph_pickle == '':
            scene_info = self._build_dataset()
            save_scene_info(scene_info, name)
        else:
            scene_info = pickle.load(open(fgraph_pickle, 'rb'))[0]
            if not seqs_in_scene_info(self.train_split, scene_info):
                print(f"Loaded scene_info {fgraph_pickle} does NOT contain all requested scenes. Rebuilding dataset...")
                scene_info = self._build_dataset()
                save_scene_info(scene_info, name)

        self.scene_info = scene_info
        self._build_dataset_index()

    def _build_dataset_index(self):
        self.dataset_index = []
        for scene in self.scene_info:
            graph = self.scene_info[scene]['graph']
            for i in graph: # graph is dict of {frameIdx: (co-visible frames, distance)}
                if len(graph[i][0]) > self.n_frames:
                    self.dataset_index.append((scene, i))

    @staticmethod
    def voxel_read(voxel_file):
        h5 = h5py.File(voxel_file, 'r')
        voxel = h5['voxel'][:] # (5, 480, 640) np.float32
        h5.close()
        return voxel

    @staticmethod
    def depth_read(depth_file):
        return np.load(depth_file)

    def build_frame_graph(self, poses, depths, intrinsics, f=16, max_flow=256):
        """ compute optical flow distance between all pairs of frames """
        def read_disp(fn):
            depth = self.__class__.depth_read(fn)[f//2::f, f//2::f]
            depth[depth < 0.01] = np.mean(depth)
            return 1.0 / depth

        poses = np.array(poses)                 # (N, 7)
        intrinsics = np.array(intrinsics) / f   # (N, 4) 

        disps = np.stack(list(map(read_disp, depths)), 0)  # (N, 480/f, 640/f) = (N, 30, 40)
        d = f * compute_distance_matrix_flow(poses, disps, intrinsics) # (N, N)

        # uncomment for nice visualization
        # import matplotlib.pyplot as plt
        # plt.imshow(d)
        # plt.show()

        graph = {}
        for i in range(d.shape[0]):
            j, = np.where(d[i] < max_flow)
            graph[i] = (j, d[i,j])

        return graph

    def __getitem__(self, index):
        """ return training video """

        #len(self.dataset_index) contains the NUMBER OF FRAMES of the scene (NVOXELS = 5*len(self.dataset_index))
        index = index % len(self.dataset_index)
        scene_id, ix = self.dataset_index[index] # scene, frame_ix in graph

        #frame_graph 1 PER FRAME (OF)
        #DEPHTS 1 PER FRAME
        frame_graph = self.scene_info[scene_id]['graph']
        voxel_list = self.scene_info[scene_id]['voxels']
        depths_list = self.scene_info[scene_id]['depths']
        poses_list = self.scene_info[scene_id]['poses']
        intrinsics_list = self.scene_info[scene_id]['intrinsics']

        # stride = np.random.choice([1,2,3])

        #d is a random number BETWEEN THE OF threshold MAX and MIN
        d = np.random.uniform(self.fmin, self.fmax)
        s = np.random.choice([1,2,3]) # default: constant 1

        inds = [ ix ]
        

        while len(inds) < self.n_frames:
            # get other frames within flow threshold
            if self.sample:     #sample = True by default
                k = (frame_graph[ix][1] > self.fmin) & (frame_graph[ix][1] < self.fmax)
                frames = frame_graph[ix][0][k] # visible frames within flow thres.

                # prefer frames forward in time
                if np.count_nonzero(frames[frames > ix]):
                    ix = np.random.choice(frames[frames > ix])

                elif ix + 1 < len(voxel_list):
                    ix = ix + 1

                elif np.count_nonzero(frames):
                    ix = np.random.choice(frames)

            else:
                i = frame_graph[ix][0].copy() # co-visible frame ixs
                g = frame_graph[ix][1].copy() # distances between ix and frames from i

                g[g > d] = -1 # exclude distances greater than d
                if s > 0:
                    g[i <= ix] = -1 # exclude frames backward in time
                else:
                    g[i >= ix] = -1 # exclude frames forward in time

                if len(g) > 0 and np.max(g) > 0: # frame ix found
                    ix = i[np.argmax(g)]
                else:
                    if ix + s >= len(voxel_list) or ix + s < 0:
                        s *= -1 # go back if end of frame

                    ix = ix + s

            inds += [ ix ]

        #ONLY LOAD THE VOXEL INDICES
        all_voxels = h5_to_voxels_indexed(voxel_list[0], 5, W=640, H=480, indexes=inds)

        voxels, depths, poses, intrinsics = [], [], [], []
        for i in inds:
            #moved out voxels append since we have a function to genereate voxels, not .h5 file for each voxel
            #voxels.append(self.__class__.voxel_read(voxel_list[i]))
            # voxels.append(all_voxels[i])
            depths.append(self.__class__.depth_read(depths_list[i]))
            poses.append(poses_list[i])
            intrinsics.append(intrinsics_list[i])

        # voxels = np.stack(all_voxels).astype(np.float32)
        voxels = np.stack(all_voxels).astype(np.float32)

        # voxels = np.stack(voxels).astype(np.float32)
        depths = np.stack(depths).astype(np.float32)
        poses = np.stack(poses).astype(np.float32)
        intrinsics = np.stack(intrinsics).astype(np.float32)

        voxels = torch.from_numpy(voxels).float()
        disps = torch.from_numpy(1.0 / depths)
        poses = torch.from_numpy(poses)
        intrinsics = torch.from_numpy(intrinsics)

        # visualize_voxel(voxels[0])

        if self.scale != 1.0:
            #check if the voxel size is already 160x160
            voxels, disps, poses, intrinsics = transform_rescale(self.scale, voxels, disps, poses, intrinsics, self.args.square)

        if self.aug:
            voxels, poses, disps, intrinsics = \
                self.aug(voxels, poses, disps, intrinsics)

        # normalize depth
        s = .7 * torch.quantile(disps, .98)
        disps = disps / s
        poses[...,:3] *= s

        if not self.return_fname:
            return voxels, poses, disps, intrinsics
        else:
            return voxels, poses, disps, intrinsics, scene_id

    def __len__(self):
        return len(self.dataset_index)

    def __imul__(self, x):
        self.dataset_index *= x
        return self
    



# class MVSECDataset(data.Dataset):
#     def __init__(self, name, datapath, n_frames=4, crop_size=[260,346], fmin=10.0, fmax=75.0, aug=True, sample=True,
#                  fgraph_pickle=None, train_split=None, val_split=None, strict_split=True, return_fname=False, scale=1.0):
#         """ Base class for Events + Depth dataset """
#         self.aug = None
#         self.root = datapath
#         self.name = name
#         self.return_fname = return_fname
#         self.resolution = crop_size

#         DEPTH_SCALE = 5.0

#         self.aug = aug
#         self.sample = sample
#         self.scale = scale

#         self.n_frames = n_frames
#         self.fmin = fmin # exclude very easy examples
#         self.fmax = fmax # exclude very hard examples

#         self.train_split = load_splitfile(train_split)
#         val_split = load_splitfile(val_split) # do not need val split later on
#         check_train_val_split(self.train_split, val_split, strict=strict_split)

#         if self.aug:
#             if self.scale != 1.0:
#                 crop_size = np.floor(self.scale * np.array(crop_size)).astype(int).tolist()
#             self.aug = EVSDAugmentor(crop_size=crop_size)

#         # building dataset is expensive, cache so only needs to be performed once
#         if fgraph_pickle is None or fgraph_pickle == '':
#             scene_info = self._build_dataset()
#             save_scene_info(scene_info, name)
#         else:
#             scene_info = pickle.load(open(fgraph_pickle, 'rb'))[0]
#             if not seqs_in_scene_info(self.train_split, scene_info):
#                 print(f"Loaded scene_info {fgraph_pickle} does NOT contain all requested scenes. Rebuilding dataset...")
#                 scene_info = self._build_dataset()
#                 save_scene_info(scene_info, name)

#         self.scene_info = scene_info
#         self._build_dataset_index()

#     def _build_dataset_index(self):
#         self.dataset_index = []
#         for scene in self.scene_info:
#             graph = self.scene_info[scene]['graph']
#             for i in graph: # graph is dict of {frameIdx: (co-visible frames, distance)}
#                 if len(graph[i][0]) > self.n_frames:
#                     self.dataset_index.append((scene, i))

#     @staticmethod
#     def voxel_read(voxel_file):
#         h5 = h5py.File(voxel_file, 'r')
#         voxel = h5['voxel'][:] # (5, 480, 640) np.float32
#         h5.close()
#         return voxel

#     @staticmethod
#     def depth_read(depth_file):
#         return np.load(depth_file)

#     # def build_frame_graph(self, poses, depths, intrinsics, f=16, max_flow=256):
#     #     """ compute optical flow distance between all pairs of frames """
#     #     def read_disp(fn):
#     #         depth = self.__class__.depth_read(fn)[f//2::f, f//2::f]
#     #         depth[depth < 0.01] = np.mean(depth)
#     #         return 1.0 / depth

#     #     poses = np.array(poses)                 # (N, 7)
#     #     intrinsics = np.array(intrinsics) / f   # (N, 4) 

#     #     disps = np.stack(list(map(read_disp, depths)), 0)  # (N, 480/f, 640/f) = (N, 30, 40)
#     #     d = f * compute_distance_matrix_flow(poses, disps, intrinsics) # (N, N)

#     #     # uncomment for nice visualization
#     #     # import matplotlib.pyplot as plt
#     #     # plt.imshow(d)
#     #     # plt.show()

#     #     graph = {}
#     #     for i in range(d.shape[0]):
#     #         j, = np.where(d[i] < max_flow)
#     #         graph[i] = (j, d[i,j])

#     #     return graph

#     def __getitem__(self, index):
#         """ return training video """

#         index = index % len(self.dataset_index)
#         scene_id, ix = self.dataset_index[index] # scene, frame_ix in graph

#         frame_graph = self.scene_info[scene_id]['graph']
#         voxel_list = self.scene_info[scene_id]['voxels']
#         depths_list = self.scene_info[scene_id]['depths']
#         poses_list = self.scene_info[scene_id]['poses']
#         intrinsics_list = self.scene_info[scene_id]['intrinsics']

#         # stride = np.random.choice([1,2,3])

#         d = np.random.uniform(self.fmin, self.fmax)
#         s = np.random.choice([1,2,3]) # default: constant 1

#         inds = [ ix ]
        

#         while len(inds) < self.n_frames:
#             # get other frames within flow threshold
#             if self.sample:
#                 k = (frame_graph[ix][1] > self.fmin) & (frame_graph[ix][1] < self.fmax)
#                 frames = frame_graph[ix][0][k] # visible frames within flow thres.

#                 # prefer frames forward in time
#                 if np.count_nonzero(frames[frames > ix]):
#                     ix = np.random.choice(frames[frames > ix])

#                 elif ix + 1 < len(voxel_list):
#                     ix = ix + 1

#                 elif np.count_nonzero(frames):
#                     ix = np.random.choice(frames)

#             else:
#                 i = frame_graph[ix][0].copy() # co-visible frame ixs
#                 g = frame_graph[ix][1].copy() # distances between ix and frames from i

#                 g[g > d] = -1 # exclude distances greater than d
#                 if s > 0:
#                     g[i <= ix] = -1 # exclude frames backward in time
#                 else:
#                     g[i >= ix] = -1 # exclude frames forward in time

#                 if len(g) > 0 and np.max(g) > 0: # frame ix found
#                     ix = i[np.argmax(g)]
#                 else:
#                     if ix + s >= len(voxel_list) or ix + s < 0:
#                         s *= -1 # go back if end of frame

#                     ix = ix + s

#             inds += [ ix ]

#         #here create the voxels with h5_to_voxel function
#         all_voxels = h5_to_voxels(voxel_list[0], 5, H=self.resolution[0], W=self.resolution[1])

#         #print(f"number of voxels {len(all_voxels)}")

#         voxels, depths, poses, intrinsics = [], [], [], []
#         for i in inds:
#             #moved out voxels append since we have a function to genereate voxels, not .h5 file for each voxel
#             #voxels.append(self.__class__.voxel_read(voxel_list[i]))
#             voxels.append(all_voxels[i])
#             depths.append(self.__class__.depth_read(depths_list[i]))
#             poses.append(poses_list[i])
#             intrinsics.append(intrinsics_list[i])




#         voxels = np.stack(voxels).astype(np.float32)
#         depths = np.stack(depths).astype(np.float32)
#         poses = np.stack(poses).astype(np.float32)
#         intrinsics = np.stack(intrinsics).astype(np.float32)

#         voxels = torch.from_numpy(voxels).float()
#         disps = torch.from_numpy(1.0 / depths)
#         poses = torch.from_numpy(poses)
#         intrinsics = torch.from_numpy(intrinsics)

#         if self.scale != 1.0:
#             voxels, disps, poses, intrinsics = transform_rescale(self.scale, voxels, disps, poses, intrinsics)

#         if self.aug:
#             voxels, poses, disps, intrinsics = \
#                 self.aug(voxels, poses, disps, intrinsics)

#         # normalize depth
#         s = .7 * torch.quantile(disps, .98)
#         disps = disps / s
#         poses[...,:3] *= s

#         if not self.return_fname:
#             return voxels, poses, disps, intrinsics
#         else:
#             return voxels, poses, disps, intrinsics, scene_id

#     def __len__(self):
#         return len(self.dataset_index)

#     def __imul__(self, x):
#         self.dataset_index *= x
#         return self


#     def _build_dataset(self):
#         from tqdm import tqdm
#         print("Building MVSEC dataset")

#         scene_info = {}
#         scenes = glob.glob(osp.join(self.root, '*/*/'))
#         print(f"Found {len(scenes)} scenes")
#         print(f"example scene: {scenes[0]}")


#         for scene in tqdm(sorted(scenes)):
#             #if not is_converted(scene):
#             #    print(f"Skipping {scene}. Not fully converted")
#             #    continue

#             if not scene_in_split(scene, self.train_split):
#                 continue
            
#             #save the path for the event file
#             data = sorted(glob.glob(osp.join(scene, '*_data.h5')))
#             gt = sorted(glob.glob(osp.join(scene, '*_gt.h5')))

#             #this creates a set of directories or files named image_left located three levels deep under self.root, which is /scratch/amarchei/TartanEvent/
#             scene = '/'.join(scene.split('/'))
#             scene_info[scene] = {'data': data, 'gt': gt}
            
#             print(f"Added {scene} to TartanAir EVDS dataset")

#         return scene_info

#     @staticmethod
#     def calib_read():
#         return np.array([320.0, 320.0, 320.0, 240.0])

#     @staticmethod
#     def voxel_read(voxel_file):
#         h5 = h5py.File(voxel_file, 'r')
#         voxel = h5['voxel'][:]
#         # assert voxel.dtype == np.float32 # (5, 480, 640)
#         h5.close()
#         return voxel

#     @staticmethod
#     def depth_read(depth_file):
#         depth = np.load(depth_file) / TartanAirEVS.DEPTH_SCALE
#         depth[depth==np.nan] = 1.0
#         depth[depth==np.inf] = 1.0
#         # visualize_depth_map(depth)
#         return depth
    
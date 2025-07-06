import numpy as np
import torch
import glob
import cv2
import os
import os.path as osp
import functools
import operator
import h5py
import hdf5plugin
import time
from ..lietorch import SE3
from .base import RGBDDataset, EVSDDataset, EVSDDataset_fake
from .utils import is_converted, scene_in_split, h5_to_voxels_indexed

class TartanAir(RGBDDataset):
    """ Derived class for TartanAir RGBD dataset """
    # scale depths to balance rot & trans
    DEPTH_SCALE = 5.0

    def __init__(self, mode='training', **kwargs):
        self.mode = mode
        self.n_frames = 2
        super(TartanAir, self).__init__(name='TartanAir', **kwargs)

    def _build_dataset(self):
        from tqdm import tqdm
        print("Building TartanAir RGBD dataset")

        scene_info = {}

        #creates a set of directories or files named image_left located three levels deep under self.root, which is /scratch/amarchei/TartanEvent/
        scenes = glob.glob(osp.join(self.root, '*/*/*/image_left'))
        print(f"Found {len(scenes)} scenes")
        print(f"example scene: {scenes[0]}")

        for scene in tqdm(sorted(scenes)):
            if not scene_in_split(scene, self.train_split, verbose=False):
                continue
                        
            images = sorted(glob.glob(osp.join(scene, '*.png')))
            assert len(images) > 0
            depths = sorted(glob.glob(osp.join(scene.replace("image_left", "depth_left"), '*.npy')))
            assert len(images) == len(depths)

            poses = np.loadtxt(osp.join(scene, '../pose_left.txt'), delimiter=' ')
            poses = poses[:, [1, 2, 0, 4, 5, 3, 6]] # NED (z,x,y) to (x,y,z) camera frame
            poses[:,:3] /= TartanAir.DEPTH_SCALE
            intrinsics = [TartanAir.calib_read()] * len(images)
            assert poses.shape[0] == len(images)

            # graph of co-visible frames based on flow
            graph = self.build_frame_graph(poses, depths, intrinsics) # graph is dict of {frameIdx: (co-visible frames, distance)}

            scene = '/'.join(scene.split('/'))
            scene_info[scene] = {'images': images, 'depths': depths, 
                'poses': poses, 'intrinsics': intrinsics, 'graph': graph}

            print(f"Added {scene} to TartanAir RGBD dataset")

        return scene_info

    @staticmethod
    def calib_read():
        return np.array([320.0, 320.0, 320.0, 240.0])

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

    @staticmethod
    def depth_read(depth_file):
        depth = np.load(depth_file) / TartanAir.DEPTH_SCALE
        depth[depth==np.nan] = 1.0
        depth[depth==np.inf] = 1.0
        # visualize_depth_map(depth)
        return depth


class TartanAirE2VID(RGBDDataset):
    """ Derived class for TartanAir e2v dataset """
    # scale depths to balance rot & trans
    DEPTH_SCALE = 5.0

    def __init__(self, mode='training', **kwargs):
        self.mode = mode
        self.n_frames = 2
        super(TartanAirE2VID, self).__init__(name='TartanAirE2VID', **kwargs)

    def _build_dataset(self):
        from tqdm import tqdm
        print("Building TartanAirE2VID dataset")

        scene_info = {}
        scenes = glob.glob(osp.join(self.root, '*/*/e2v'))
        scenes = [glob.glob(osp.join(s, '*/*/*/*')) for s in scenes]
        scenes = functools.reduce(operator.concat, scenes)
        for scene in tqdm(sorted(scenes)):
            if not scene_in_split(scene, self.train_split):
                continue

            images = sorted(glob.glob(osp.join(scene, 'e2calib/*.png')))
            assert len(images) > 0
            depthdir = scene.replace("/e2v/", "/depth_left/").replace("/datasets/tartan-e2v/", "/datasets/tartan/")
            depths = sorted(glob.glob(osp.join(depthdir, 'depth_left/*.npy')))[1:]
            assert len(images) == len(depths)

            scene_tartan = scene.replace("/e2v/", "/image_left/").replace("/datasets/tartan-e2v/", "/datasets/tartan/")
            poses = np.loadtxt(osp.join(scene_tartan, 'pose_left.txt'), delimiter=' ')
            poses = poses[1:, [1, 2, 0, 4, 5, 3, 6]] # NED (z,x,y) to (x,y,z) camera frame
            poses[:,:3] /= TartanAir.DEPTH_SCALE
            intrinsics = [TartanAir.calib_read()] * len(images)
            assert poses.shape[0] == len(images)

            # graph of co-visible frames based on flow
            graph = self.build_frame_graph(poses, depths, intrinsics) # graph is dict of {frameIdx: (co-visible frames, distance)}

            scene = '/'.join(scene.split('/'))
            scene_info[scene] = {'images': images, 'depths': depths,
                'poses': poses, 'intrinsics': intrinsics, 'graph': graph}

            print(f"Added {scene} to TartanAir RGBD dataset")

        return scene_info

    @staticmethod
    def calib_read():
        return np.array([320.0, 320.0, 320.0, 240.0])

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

    @staticmethod
    def depth_read(depth_file):
        depth = np.load(depth_file) / TartanAir.DEPTH_SCALE
        depth[depth==np.nan] = 1.0
        depth[depth==np.inf] = 1.0
        # visualize_depth_map(depth)
        return depth


class TartanAirEVS(EVSDDataset):
    """ Derived class for TartanAir event + depth dataset (EVSD) """
    # scale depths to balance rot & trans
    DEPTH_SCALE = 5.0

    def __init__(self, mode='training', scale=1.0, args=None, **kwargs):
        self.mode = mode
        self.scale = scale 
        self.n_frames = 2
        self.args = args
        super(TartanAirEVS, self).__init__(name='TartanAirEVS', scale=scale, args=args, **kwargs)

    def _build_dataset(self):
        from tqdm import tqdm
        print("Building TartanAir EVSD dataset")
        
        #training script should be already laoded in parent class
        #with open(self.train_split, "r") as f:
        #    #read each line of the training split file, example : /usr/scratch/badile43/amarchei/TartanEvent/seasonsforest/Easy/P002
#
        #    lines = f.readlines()
        #    scenes = []
        #    for line in lines:
        #        line = line.strip()
        #        scenes.append(line)
        scenes = self.train_split
        scene_info = {}

        if self.args.precomputed_inputs == True:


            
            #start the loop loading every path of each of the depth event and flow folders
            for scene in tqdm(scenes):
                
                #create list of voxels (although they are stack)
                #load the paths of the event stack containing each teh .npy files
                if self.args.event_representation == "stack":
                    events = os.path.join(scene, "event_frames")
                elif self.args.event_representation == "voxel":
                    events = os.path.join(scene, "event_voxel")

                #replace the badile43 with badile44
                events = events.replace("badile43", "badile44")
                print("Loading events from: ", events)
                voxels = sorted(glob.glob(osp.join(events, '*.npy')))
                assert len(voxels) > 0

                n_voxels = len(voxels)

                depths = sorted(glob.glob(osp.join(scene, 'depth_left/*.npy')))[1:] # No event voxel at first timestamp t=0

                diff = len(depths) - n_voxels

                if diff > 0:
                    #remove the last diff elements from the depths
                    depths = depths[:-diff]
                assert len(voxels) == len(depths)

                # [simon] poses are c2w, did thorough viz and data_type.md]
                poses = np.loadtxt(osp.join(scene, 'pose_left.txt'), delimiter=' ')[1:] # No event voxel at first timestamp t=0
                poses = poses[:, [1, 2, 0, 4, 5, 3, 6]] # NED (z,x,y) to (x,y,z) camera frame
                poses[:,:3] /= TartanAirEVS.DEPTH_SCALE
                intrinsics = [TartanAirEVS.calib_read()] * len(depths)

                diff = len(poses) - len(depths)
                if diff > 0:
                    #remove the last diff elements from the poses
                    poses = poses[:-diff]
                assert poses.shape[0] == len(depths)

                # graph of co-visible frames based on flow
                #OPTICAL FLOW COMPUTATION BETWEEN EACH FRAME PAIR
                # OF = f(GT_POSE,DEPTH)
                
                #GOAl : skip frames that have too much magnitude or not
                graph = self.build_frame_graph(poses, depths, intrinsics) # graph is dict of {frameIdx: (co-visible frames, distance)}


                #this creates a set of directories or files named image_left located three levels deep under self.root, which is /scratch/amarchei/TartanEvent/
                scene = '/'.join(scene.split('/'))
                scene_info[scene] = {'voxels': voxels, 'depths': depths,
                    'poses': poses, 'intrinsics': intrinsics, 'graph': graph}
                
                print(f"Added {scene} to TartanAir EVDS dataset")

            return scene_info
            
        else:
            
            #CLASSICAL and previous implementation, where the processing happens at runtime, so only load the events.h5

            scenes = self.train_split
            print(f"Found {len(scenes)} scenes")
            print(f"example scene: {scenes[0]}")


            for scene in tqdm(sorted(scenes)):
                #if not is_converted(scene):
                #    print(f"Skipping {scene}. Not fully converted")
                #    continue
                
                #save the path for the event file
                voxels = sorted(glob.glob(osp.join(scene, '*.h5')))

                assert len(voxels) > 0
                depths = sorted(glob.glob(osp.join(scene, 'depth_left/*.npy')))[1:] # No event voxel at first timestamp t=0
                #assert len(voxels) == len(depths)

                # [simon] poses are c2w, did thorough viz and data_type.md]
                poses = np.loadtxt(osp.join(scene, 'pose_left.txt'), delimiter=' ')[1:] # No event voxel at first timestamp t=0
                poses = poses[:, [1, 2, 0, 4, 5, 3, 6]] # NED (z,x,y) to (x,y,z) camera frame
                poses[:,:3] /= TartanAirEVS.DEPTH_SCALE
                intrinsics = [TartanAirEVS.calib_read()] * len(depths)
                assert poses.shape[0] == len(depths)

                # graph of co-visible frames based on flow
                #OPTICAL FLOW COMPUTATION BETWEEN EACH FRAME PAIR
                # OF = f(GT_POSE,DEPTH)
                
                #GOAl : skip frames that have too much magnitude or not
                graph = self.build_frame_graph(poses, depths, intrinsics) # graph is dict of {frameIdx: (co-visible frames, distance)}


                #this creates a set of directories or files named image_left located three levels deep under self.root, which is /scratch/amarchei/TartanEvent/
                scene = '/'.join(scene.split('/'))
                scene_info[scene] = {'voxels': voxels, 'depths': depths,
                    'poses': poses, 'intrinsics': intrinsics, 'graph': graph}
                
                print(f"Added {scene} to TartanAir EVDS dataset")

            return scene_info

    @staticmethod
    def calib_read():
        return np.array([320.0, 320.0, 320.0, 240.0])

    @staticmethod
    def voxel_read(voxel_file):
        h5 = h5py.File(voxel_file, 'r')
        voxel = h5['voxel'][:]
        # assert voxel.dtype == np.float32 # (5, 480, 640)
        h5.close()
        return voxel


    #PREVIOUS IMPLEMENTATION, THROWED SOME ERRORS WHEN FILE TEMPORARILY NOT FOUND
    # @staticmethod
    # def depth_read(depth_file):
    #     depth = np.load(depth_file) / TartanAirEVS.DEPTH_SCALE
    #     depth[depth==np.nan] = 1.0
    #     depth[depth==np.inf] = 1.0
    #     # visualize_depth_map(depth)
    #     return depth
    
    @staticmethod
    def depth_read(path, max_retries=3, delay=1.0):
        for attempt in range(max_retries):
            if os.path.exists(path):
                try:
                    return np.load(path)
                except Exception as e:
                    print(f"Retry {attempt+1}/{max_retries}: failed to load {path}: {e}")
            else:
                print(f"Retry {attempt+1}/{max_retries}: {path} not found, waiting {delay} seconds...")
            time.sleep(delay)
        raise FileNotFoundError(f"Failed to load file after {max_retries} retries: {path}")




class TartanAirEVS_fake(EVSDDataset_fake):
    """ Derived class for TartanAir event + depth dataset (EVSD) """
    # scale depths to balance rot & trans
    DEPTH_SCALE = 5.0

    def __init__(self, mode='training', scale=1.0, args=None, **kwargs):
        self.mode = mode
        self.scale = scale 
        self.n_frames = 2
        self.args = args
        super(TartanAirEVS_fake, self).__init__(name='TartanAirEVS', scale=scale, args=args, **kwargs)

    def _build_dataset(self):
        from tqdm import tqdm
        print("Building TartanAir EVSD dataset")
        
        #training script should be already laoded in parent class
        #with open(self.train_split, "r") as f:
        #    #read each line of the training split file, example : /usr/scratch/badile43/amarchei/TartanEvent/seasonsforest/Easy/P002
#
        #    lines = f.readlines()
        #    scenes = []
        #    for line in lines:
        #        line = line.strip()
        #        scenes.append(line)
        scenes = self.train_split
        scene_info = {}

        if self.args.precomputed_inputs == True:


            
            #start the loop loading every path of each of the depth event and flow folders
            for scene in tqdm(scenes):
                
                #create list of voxels (although they are stack)
                #load the paths of the event stack containing each teh .npy files
                if self.args.event_representation == "stack":
                    events = os.path.join(scene, "event_frames")
                elif self.args.event_representation == "voxel":
                    events = os.path.join(scene, "event_voxel")

                #replace the badile43 with badile44
                events = events.replace("badile43", "badile44")
                print("Loading events from: ", events)
                voxels = sorted(glob.glob(osp.join(events, '*.npy')))
                assert len(voxels) > 0

                n_voxels = len(voxels)

                depths = sorted(glob.glob(osp.join(scene, 'depth_left/*.npy')))[1:] # No event voxel at first timestamp t=0

                diff = len(depths) - n_voxels

                if diff > 0:
                    #remove the last diff elements from the depths
                    depths = depths[:-diff]
                assert len(voxels) == len(depths)

                # [simon] poses are c2w, did thorough viz and data_type.md]
                poses = np.loadtxt(osp.join(scene, 'pose_left.txt'), delimiter=' ')[1:] # No event voxel at first timestamp t=0
                poses = poses[:, [1, 2, 0, 4, 5, 3, 6]] # NED (z,x,y) to (x,y,z) camera frame
                poses[:,:3] /= TartanAirEVS.DEPTH_SCALE
                intrinsics = [TartanAirEVS.calib_read()] * len(depths)

                diff = len(poses) - len(depths)
                if diff > 0:
                    #remove the last diff elements from the poses
                    poses = poses[:-diff]
                assert poses.shape[0] == len(depths)

                # graph of co-visible frames based on flow
                #OPTICAL FLOW COMPUTATION BETWEEN EACH FRAME PAIR
                # OF = f(GT_POSE,DEPTH)
                
                #GOAl : skip frames that have too much magnitude or not
                if self.args.sequence_mode == 'covisibility_off':
                    max_flow = 10000
                    print("Using covisibility_off mode, setting max_flow to 10000")
                    graph = self.build_frame_graph(poses, depths, intrinsics, max_flow=max_flow) # graph is dict of {frameIdx: (co-visible frames, distance)}
                else:
                    graph = self.build_frame_graph(poses, depths, intrinsics)

                #this creates a set of directories or files named image_left located three levels deep under self.root, which is /scratch/amarchei/TartanEvent/
                scene = '/'.join(scene.split('/'))
                scene_info[scene] = {'voxels': voxels, 'depths': depths,
                    'poses': poses, 'intrinsics': intrinsics, 'graph': graph}
                
                print(f"Added {scene} to TartanAir EVDS dataset")

            return scene_info
            
        else:
            
            #CLASSICAL and previous implementation, where the processing happens at runtime, so only load the events.h5

            scenes = self.train_split
            print(f"Found {len(scenes)} scenes")
            print(f"example scene: {scenes[0]}")


            for scene in tqdm(sorted(scenes)):
                #if not is_converted(scene):
                #    print(f"Skipping {scene}. Not fully converted")
                #    continue
                
                #save the path for the event file
                voxels = sorted(glob.glob(osp.join(scene, '*.h5')))

                assert len(voxels) > 0
                depths = sorted(glob.glob(osp.join(scene, 'depth_left/*.npy')))[1:] # No event voxel at first timestamp t=0
                #assert len(voxels) == len(depths)

                # [simon] poses are c2w, did thorough viz and data_type.md]
                poses = np.loadtxt(osp.join(scene, 'pose_left.txt'), delimiter=' ')[1:] # No event voxel at first timestamp t=0
                poses = poses[:, [1, 2, 0, 4, 5, 3, 6]] # NED (z,x,y) to (x,y,z) camera frame
                poses[:,:3] /= TartanAirEVS.DEPTH_SCALE
                intrinsics = [TartanAirEVS.calib_read()] * len(depths)
                assert poses.shape[0] == len(depths)

                # graph of co-visible frames based on flow
                #OPTICAL FLOW COMPUTATION BETWEEN EACH FRAME PAIR
                # OF = f(GT_POSE,DEPTH)
                
                #GOAl : skip frames that have too much magnitude or not
                if self.args.sequence_mode == 'covisibility_off':
                    max_flow = 10000
                    print("Using covisibility_off mode, setting max_flow to 10000")
                    graph = self.build_frame_graph(poses, depths, intrinsics, max_flow=max_flow) # graph is dict of {frameIdx: (co-visible frames, distance)}
                else:
                    graph = self.build_frame_graph(poses, depths, intrinsics)

                #this creates a set of directories or files named image_left located three levels deep under self.root, which is /scratch/amarchei/TartanEvent/
                scene = '/'.join(scene.split('/'))
                scene_info[scene] = {'voxels': voxels, 'depths': depths,
                    'poses': poses, 'intrinsics': intrinsics, 'graph': graph}
                
                print(f"Added {scene} to TartanAir EVDS dataset")

            return scene_info

    @staticmethod
    def calib_read():
        return np.array([320.0, 320.0, 320.0, 240.0])

    @staticmethod
    def voxel_read(voxel_file):
        h5 = h5py.File(voxel_file, 'r')
        voxel = h5['voxel'][:]
        # assert voxel.dtype == np.float32 # (5, 480, 640)
        h5.close()
        return voxel


    #PREVIOUS IMPLEMENTATION, THROWED SOME ERRORS WHEN FILE TEMPORARILY NOT FOUND
    # @staticmethod
    # def depth_read(depth_file):
    #     depth = np.load(depth_file) / TartanAirEVS.DEPTH_SCALE
    #     depth[depth==np.nan] = 1.0
    #     depth[depth==np.inf] = 1.0
    #     # visualize_depth_map(depth)
    #     return depth
    
    @staticmethod
    def depth_read(path, max_retries=3, delay=1.0):
        for attempt in range(max_retries):
            if os.path.exists(path):
                try:
                    return np.load(path)
                except Exception as e:
                    print(f"Retry {attempt+1}/{max_retries}: failed to load {path}: {e}")
            else:
                print(f"Retry {attempt+1}/{max_retries}: {path} not found, waiting {delay} seconds...")
            time.sleep(delay)
        raise FileNotFoundError(f"Failed to load file after {max_retries} retries: {path}")

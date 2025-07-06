import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from collections import OrderedDict
import contextlib
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from devo.data_readers.factory import dataset_factory
import warnings

from devo.lietorch import SE3
from devo.logger import Logger
import torch.nn.functional as F

# from devo.net import VONet # TODO add net.py
from devo.enet import eVONet
from devo.selector import SelectionMethod

# DDP training
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm

import sys
import os
import argparse
import yaml
from types import SimpleNamespace

import json
from torch.distributed import all_gather_object

DEBUG_PLOT_PATCHES = False
NUM_WORKERS = 0

#clear cache
torch.cuda.empty_cache()


torch.manual_seed(10)


def dict2namespace(d):
    return argparse.Namespace(**d)

def load_config(path):
    with open(path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    return dict2namespace(cfg_dict)


def setup_ddp(rank, args):
    os.environ['MASTER_ADDR'] = 'localhost'

    #if args.port is an integer, convert it to string
    if isinstance(args.port, int):
        args.port = str(args.port)

    os.environ['MASTER_PORT'] = args.port
        
    dist.init_process_group(                                   
    	backend='nccl',                                         
   		init_method='env://',     
    	world_size=args.gpu_num,                              
    	rank=rank)

    torch.manual_seed(0)
    torch.cuda.set_device(rank)

def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey()

def image2gray(image):
    image = image.mean(dim=0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey()


def initialize_voxel_list_debug(db):
    print("Debug mode enabled. Saving voxel usage and scene information.")
    #save information about the dataloader, logging information about the occurrences fo each frame in each scene
    voxel_usage_counter = db.datasets[0].get_dataset_lenght() # returns a dict with scene_id as key and number of frames as value

    return voxel_usage_counter

def save_scene_info(voxel_usage_counter, scene_info, inds):
    """ Update the scene_info with the current iteration index """
    scene_info = scene_info[0] if isinstance(scene_info, list) else scene_info
    for ix in inds:
        voxel_usage_counter[scene_info][ix] += 1

def merge_voxel_counters(voxel_usage_counter_all_ranks):
    merged = {}
    for rank_dict in voxel_usage_counter_all_ranks:
        for scene, counts in rank_dict.items():
            if scene not in merged:
                merged[scene] = counts.copy()
            else:
                merged[scene] = [a + b for a, b in zip(merged[scene], counts)]
    return merged

def create_single_json(voxel_usage_counter, rank, args):
    # Save voxel debug info after 100 steps
    if args.ddp:
        gathered_voxels = [None for _ in range(args.gpu_num)]
        all_gather_object(gathered_voxels, voxel_usage_counter)

        if rank == 0:
            merged_voxels = merge_voxel_counters(gathered_voxels)
            output_path = f'voxel_usage_{args.steps}k_n{args.n_frames}.json'
            if getattr(args, 'sequence_mode', None) == 'fw_mode':

                output_path = f'voxel_usage_{args.steps}k_n{args.n_frames}_covisibility_off.json'

            with open(output_path, 'w') as f:
                json.dump(merged_voxels, f, indent=2)
    else:
        output_path = f'voxel_usage_{args.steps}k_n{args.n_frames}.json'
        if getattr(args, 'sequence_mode', None) == 'fw_mode':
            output_path = f'voxel_usage_{args.steps}k_n{args.n_frames}_covisibility_off.json'

        with open(output_path, 'w') as f:
            json.dump(voxel_usage_counter, f, indent=2)

#def kabsch_umeyama(A, B):
#    """ compute optimal scaling (SIM3) that minimizing RMSD between two sets of points """
#    n, m = A.shape
#    EA = torch.mean(A, axis=0)
#    EB = torch.mean(B, axis=0)
#    VarA = torch.mean((A - EA).norm(dim=1)**2)
#
#    H = ((A - EA).T @ (B - EB)) / n
#    U, D, VT = torch.linalg.svd(H.cpu(), full_matrices=False)       #HERE ERROR SOMETIMES, does not converge
#    #U, D, VT = torch.svd(H) instead of the above line
#
#    c = VarA / torch.trace(torch.diag(D))
#    return c


def kabsch_umeyama(A, B, default_scale=1.0):
    """
    Compute optimal SIM3 scale minimizing RMSD between point sets A and B.
    If computation fails, a warning is emitted and default_scale is returned.
    """
    default_scale = torch.tensor(default_scale, dtype=A.dtype, device=A.device)  # <- aggiunta
    try:
        n, m = A.shape
        EA = torch.mean(A, dim=0)
        EB = torch.mean(B, dim=0)
        VarA = torch.mean((A - EA).norm(dim=1)**2)
        VarB = torch.mean((B - EB).norm(dim=1)**2)

        if VarA < 1e-6 or VarB < 1e-6:
            warnings.warn("Variance too low; using default scale.")
            return default_scale

        H = ((A - EA).T @ (B - EB)) / n

        if torch.isnan(H).any() or torch.isinf(H).any() or H.norm() < 1e-6:
            warnings.warn("Covariance matrix is invalid; using default scale.")
            return default_scale

        try:
            U, D, VT = torch.linalg.svd(H.cpu(), full_matrices=False)
        except RuntimeError as e:
            warnings.warn(f"SVD failed: {e}. Trying torch.svd fallback.")
            try:
                U, S, V = torch.svd(H.cpu())
                D = S
                VT = V.T
            except RuntimeError as e2:
                warnings.warn(f"Fallback SVD also failed: {e2}. Using default scale.")
                return default_scale

        trace_D = torch.sum(D)
        if trace_D < 1e-8:
            warnings.warn("Trace too small; using default scale.")
            return default_scale

        c = VarA / trace_D
        return c

    except Exception as ex:
        warnings.warn(f"Unhandled exception in kabsch_umeyama: {ex}. Using default scale.")
        return default_scale





def train(rank, args):
    """ main training loop """
    
    print("args.scale: ", args.scale)
    # coordinate multiple GPUs
    if args.ddp:
        setup_ddp(rank, args)

    # fetch dataset
    if args.evs:
        print("Using EVS dataset for debugging.")
        db = dataset_factory(['tartan_evs_fake'], datapath=args.datapath, n_frames=args.n_frames,
                             fgraph_pickle=args.fgraph_pickle, train_split=args.train_split,
                             val_split=args.val_split, strict_split=False, sample=True, return_fname=True, scale=args.scale, args=args)
    # elif args.e2vid:
    #     db = dataset_factory(['tartan_e2vid'], datapath=args.datapath, n_frames=args.n_frames,
    #                          fgraph_pickle=args.fgraph_pickle, train_split=args.train_split,
    #                          val_split=args.val_split, strict_split=False, sample=True, return_fname=True, scale=args.scale)  
    # elif args.evs and args.e2vid:
    #     db = dataset_factory(['tartan'], datapath=args.datapath, n_frames=args.n_frames,
    #                          fgraph_pickle=args.fgraph_pickle, train_split=args.train_split, 
    #                          val_split=args.val_split, strict_split=False, sample=True, return_fname=True, scale=args.scale)
    else:
        #raise error
        raise ValueError("Unknown dataset")
    
    # setup dataloader
    if args.ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            db, shuffle=True, num_replicas=args.gpu_num, rank=rank)
        train_loader = DataLoader(db, batch_size=args.batch, sampler=train_sampler, num_workers=NUM_WORKERS, pin_memory=True)
    else:
        train_loader = DataLoader(db, batch_size=args.batch, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Initial VOnet
    kwargs_net = {"ctx_feat_dim": args.ctx_feat_dim, "match_feat_dim": args.match_feat_dim, "dim": args.dim}
    net = VONet(**kwargs_net, patch_selector=args.patch_selector.lower()) if not args.evs else \
    eVONet(**kwargs_net, patch_selector=args.patch_selector.lower(), norm=args.norm, randaug=args.randaug, args=args)


    net.train()
    net.cuda()

    # if args.torch_compile:
    #     net = torch.compile(net)

    P = net.P # patch size (squared)
        
    if args.ddp:
        net = DDP(net, device_ids=[rank], find_unused_parameters=False)

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
        args.lr, args.steps, pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')
    
    total_steps = 0
    if args.checkpoint is not None and args.checkpoint == True:
        print(f"Loading from last checkpoint file in checkpoints/{args.name}/")

        checkpoint_files = [f for f in os.listdir(f'checkpoints/{args.name}') if f.endswith('.pth')]

        print(f"checkpoint_files: {checkpoint_files}")

        if len(checkpoint_files) == 0:
            print("No checkpoint files found, starting from scratch.")
            total_steps = 0
            #exit from the if block
            
        else:
            #the name of the checkpint is like this  005000.pth  015000.pth  025000.pth  035000.pth  045000.pth  055000.pth  065000.pth  075000.pth 
            #extract the latest and highest number from the checkpoint files
            last_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('.')[0]))

            last_checkpoint = f'checkpoints/{args.name}/{last_checkpoint}'
            print(f"Loading last checkpoint file: {last_checkpoint}")


            checkpoint = torch.load(last_checkpoint)
            model = net.module if args.ddp else net
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # legacy
                new_state_dict = OrderedDict()
                for k, v in checkpoint.items():
                    new_state_dict[k.replace('module.', '')] = v
                # with RGB pretraining
                update_dict = { k: v for k, v in new_state_dict.items() if k in model.state_dict() and model.state_dict()[k].shape == v.shape }
                print(update_dict.keys())
                state = model.state_dict()
                state.update(update_dict)
                # keys with different shape: ['patchify.fnet.conv1.weight', 'patchify.inet.conv1.weight']
                # corresponding values: [torch.Size([32, 3, 7, 7]), torch.Size([32, 3, 7, 7])]
                model.load_state_dict(state, strict=False)
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'steps' in checkpoint:
                total_steps = checkpoint['steps']
                print(f"Resuming from {total_steps} steps within single GPU")
                if args.ddp:
                    total_steps_debug = total_steps * args.gpu_num
                    print(f"Resuming from {total_steps_debug} globally across {args.gpu_num} GPUs")

    elif args.checkpoint is not None and args.checkpoint != '':
        print(f"Loading SPECIFIC checkpoint file from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint)
        model = net.module if args.ddp else net
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # legacy
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                new_state_dict[k.replace('module.', '')] = v
            # with RGB pretraining
            update_dict = { k: v for k, v in new_state_dict.items() if k in model.state_dict() and model.state_dict()[k].shape == v.shape }
            print(update_dict.keys())
            state = model.state_dict()
            state.update(update_dict)
            # keys with different shape: ['patchify.fnet.conv1.weight', 'patchify.inet.conv1.weight']
            # corresponding values: [torch.Size([32, 3, 7, 7]), torch.Size([32, 3, 7, 7])]
            model.load_state_dict(state, strict=False)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'steps' in checkpoint:
            total_steps = checkpoint['steps']
            print(f"Resuming from {total_steps} steps within single GPU")
            if args.ddp:
                total_steps_debug = total_steps * args.gpu_num
                print(f"Resuming from {total_steps_debug} globally across {args.gpu_num} GPUs")

    if rank == 0:
        print("Rank 0 : initialize logger")
        logger = Logger(args.name, scheduler, args.gpu_num * total_steps, args.gpu_num, args.tensorboard_update_step, args_config=args)


    #check if the checkpoint path contains a number in the last file name
    # if args.checkpoint is not None and args.checkpoint != '':
    #     checkpoint_path = args.checkpoint.split("/")[-1]
    #     checkpoint_path = checkpoint_path.split(".")[0]
    #     if checkpoint_path.isdigit():
    #         checkpoint_path = int(checkpoint_path)
            
    #         total_steps = checkpoint_path
    #         print(f"Starting from {total_steps} steps")

    if args.name == 'debug':
        #create the json files containing
        print("Debug mode enabled. Initializing voxel list for debugging. getting information directly from the dataloader")
        voxel_usage_counter = initialize_voxel_list_debug(db)

    avg_in_range_cov_ratio = 0.0
    with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(skip_first=1997, wait=1, warmup=1, active=2, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f'runs/{args.name}', rank),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) if args.profiler else contextlib.nullcontext() as prof:
        while True:
            for data_blob in tqdm(train_loader):
                scene_id = data_blob.pop()

                if args.name == 'debug':
                    #used for debugging purposes, getting statistics on the processed data
                    images, poses, disps, intrinsics, inds,in_range_cov = [
                        x.cuda().float() if torch.is_tensor(x) else x for x in data_blob
                    ]

                    #transform the in_range_cov to a single value
                    in_range_cov = in_range_cov.float()  # ensure it's a float tensor
                    avg_in_range_cov_ratio += in_range_cov.mean().item()  # accumulate the average covisibility ratio

                    #update the voxel usage counter
                    #if total_steps % 5000 == 0 : print(f"scene_id: {scene_id}, inds: {inds}")
                    save_scene_info(voxel_usage_counter, scene_id, inds) 
                
                else:
                    images, poses, disps, intrinsics = [x.cuda().float() for x in data_blob] # images: (B,n_frames,C,H,W), poses: (B,n_frames,7), disps: (B,n_frames,H,W) (all float32)
                
                
                optimizer.zero_grad(set_to_none=True) # TODO set_to_none=True
                
                total_steps += 1
                
                if args.name == 'debug' and total_steps % 10000 == 0:
                    #print(f"[DEBUG] Step {total_steps} - Saving voxel usage and scene information.")
                    create_single_json(voxel_usage_counter, rank, args)

                            
            
                if total_steps >= args.steps:
                    avg_in_range_cov_ratio /= args.steps  # average over all batches
                    print(f"[DEBUG] Average in-range covisibility ratio: {avg_in_range_cov_ratio:.4f}")
                    break
            else:
                continue
            break
            
    if rank == 0:
        if logger.writer is not None:
            logger.close()
    if args.ddp:
        dist.destroy_process_group()


def assert_config(args):
    assert os.path.isdir(args.datapath)
    
    assert args.gpu_num > 0 and args.gpu_num <= 10
    if args.gpu_num > 1:
        assert args.ddp
    assert args.batch > 0 and args.batch <= 1024
    assert args.steps > 0 and args.steps <= 4800000
    assert args.steps % args.gpu_num == 0
    assert args.iters >= 2 and args.iters <= 50
    assert args.lr > 0 and args.lr < 1
    assert args.n_frames > 7 and args.n_frames < 100 #  The first 8 frames are used for initialization while the next n_frames-8 frames are added one at a time
    assert args.pose_weight >= 0 and args.pose_weight <= 100 and args.flow_weight >= 0 and args.flow_weight <= 100

    if args.checkpoint is not None and args.checkpoint != '' and args.checkpoint != True:
        assert os.path.isfile(args.checkpoint)
        assert ".pth" in args.checkpoint or ".pt" in args.checkpoint 
    if args.fgraph_pickle is not None and args.fgraph_pickle != '':
        assert os.path.isfile(args.fgraph_pickle)
        assert ".pickle" in args.fgraph_pickle
    
    assert os.path.isfile(args.train_split)
    assert os.path.isfile(args.val_split)

    if args.ddp:
        assert DEBUG_PLOT_PATCHES == False


if __name__ == '__main__':
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config/debug.yaml'
    args = load_config(config_path)
    assert_config(args)

    print("----- CONFIGURATION -----")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("-------------------------")

    os.makedirs(f'checkpoints/{args.name}', exist_ok=True)

    #add different names to the variables for portability

    os.system("nvidia-smi")
    print(f"CUDA Version: {torch.version.cuda}")

    os.system("ulimit -n 2000000")

    args.steps = args.steps // args.gpu_num

    if args.ddp:
        mp.spawn(train, nprocs=args.gpu_num, args=(args,))
    else:
        train(0, args)

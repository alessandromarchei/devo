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


DEBUG_PLOT_PATCHES = False
NUM_WORKERS = 16

#clear cache
torch.cuda.empty_cache()



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
        db = dataset_factory(['tartan_evs'], datapath=args.datapath, n_frames=args.n_frames,
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
        train_loader = DataLoader(db, batch_size=args.batch, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

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

                images, poses, disps, intrinsics = [x.cuda().float() for x in data_blob] # images: (B,n_frames,C,H,W), poses: (B,n_frames,7), disps: (B,n_frames,H,W) (all float32)
                
                
                optimizer.zero_grad(set_to_none=True) # TODO set_to_none=True

                # print(f"scene_id: {scene_id}")
                #visualize_voxel(images[0][0])

                
                # fix poses to gt for first 1k steps
                ##### POSES = GT for the first N STEPS #####
                ##### ONLY TRAIN ON THE DEPTHS #####
                #print(f"total_steps: {total_steps}, scene_id: {scene_id}, poses.shape: {poses.shape}, disps.shape: {disps.shape}")
                so = total_steps < (args.first_gt_poses // args.gpu_num) and (args.checkpoint is None or args.checkpoint == "")

                poses = SE3(poses).inv() # [simon]: this does c2w -> w2c (which dpvo predicts&stores internally)


                #INFERENCE 
                traj = net(images, poses, disps, intrinsics, M=1024, STEPS=args.iters, structure_only=so, plot_patches=DEBUG_PLOT_PATCHES, patches_per_image=args.patches_per_image)
                
                
                # list [valid, p_ij, p_ij_gt, poses, poses_gt, kl] of iters (update operator)
                if DEBUG_PLOT_PATCHES:
                    patch_data = traj.pop()
                # valid (B,edges)
                # p_ij, p_ij_gt (B,edges,P,P,2), float32
                # poses, poses_gt (SE3.data of dim (B,n_frames,7))

                # Compute loss and metrics
                loss = 0.0
                pose_loss = 0.0
                flow_loss = 0.0
                scores_loss = torch.as_tensor(0.0)
                for i, data in enumerate(traj):
                    if args.patch_selector == SelectionMethod.SCORER:
                        (v, x, y, P1, P2, kl, scores, v_full, x_full, y_full, ba_weights, kk, dij) = data
                    else:
                        #valid, coords, coords_gt, Gs[:,:n], Ps[:,:n], kl
                        # v = valid
                        # x = coords
                        # y = coords_gt
                        # P1 = ABSOLUTE ESTIMATED POSE (WORLD COORDINATE)
                        # P2 = ABSOLUTE GT POSE (WORLD COORDINATE)
                        # kl = 0
                        (v, x, y, P1, P2, kl) = data
                    valid = (v > 0.5).reshape(-1) 
                    e = (x - y).norm(dim=-1) # residual (p_ij - p_ij_gt)
                    ef = e.reshape(-1, P**2)[valid].min(dim=-1).values # e.shape: (B*edges,P^2) -> (B*edges)
                    flow_loss = ef.mean()
                    
                    start_scorer = (i == (len(traj)-1)) and (total_steps // args.gpu_num) >= 1e+4
                    start_scorer = (i == (len(traj)-1))

                    if args.patch_selector == SelectionMethod.SCORER and start_scorer:
                        import math
                        valid_full = (v_full >= 0.5).reshape(-1)

                        kk = kk[valid_full]
                        e_full = (x_full - y_full).norm(dim=-1) # residual (p_ij - p_ij_gt)
                        e_full = e_full.reshape(-1, P**2)[valid_full].min(dim=-1).values # e.shape: (B*edges,P^2) -> (B*edges)
                        # scorer (flow only)
                        # scores_loss = (scores.view(-1)[kk] * e_full).mean()
                        # scorer (flow + ba)
                        scores_loss = ((-0.5*(ba_weights.view(-1,2)[valid_full].mean(dim=-1)).log() + 1) * scores.view(-1)[kk] * e_full).mean()
                        
                        scores = torch.max(scores, torch.as_tensor(1e-6))  
                        scores = -scores.log()
                        scores_loss += scores.mean()
                    else:
                        scores_loss = torch.as_tensor(0.0)
                    
                    N = P1.shape[1] # number frames (n_frames)
                    ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N))
                    ii = ii.reshape(-1).cuda()
                    jj = jj.reshape(-1).cuda()

                    k = ii != jj # not same frame
                    ii = ii[k]
                    jj = jj[k]

                    P1 = P1.inv() # because of (SE3(poses).inv())
                    P2 = P2.inv() # w2c => c2w

                    t1 = P1.matrix()[...,:3,3] #ABSOLUTE  predicted translation # TODO with detach()?
                    t2 = P2.matrix()[...,:3,3] #ABSOLUTE  gt translation # TODO with detach()?

                    s = kabsch_umeyama(t2[0], t1[0]).detach().clamp(max=10.0) # how to handle batch greater than 1?
                    P1 = P1.scale(s.view(1, 1))

                    dP = P1[:,ii].inv() * P1[:,jj] # predicted poses from frame i to j (G_ij)
                    dG = P2[:,ii].inv() * P2[:,jj] # gt poses from frame i to j (T_ij)

                    e1 = (dP * dG.inv()).log() # poses loss for each pair of frames
                    tr = e1[...,0:3].norm(dim=-1) # tx ty tz
                    ro = e1[...,3:6].norm(dim=-1) # qx qy qz

                    loss += args.flow_weight * flow_loss
                    loss += args.scores_weight * scores_loss
                    pose_loss = tr.mean() + ro.mean()
                    if not so and i >= 2:
                        loss += args.pose_weight * pose_loss

                # if rank == 0 and DEBUG_PLOT_PATCHES:
                #     plot_patch_following_all(images, patch_data, evs=args.evs, outdir=f"../viz/patches_all/name_{args.name}/step_{total_steps}/")
                #     plot_patch_following(images, patch_data, evs=args.evs, outdir=f"../viz/patches/name_{args.name}/step_{total_steps}/")
                #     plot_patch_depths_all(images, patch_data, disps, evs=args.evs, outdir=f"../viz/patches_depths_all/name_{args.name}/step_{total_steps}/")
                
                if torch.isnan(loss):
                    print(f"nan at {total_steps}: {scene_id}")
                
                loss.backward()

                torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
                optimizer.step()   
                scheduler.step()

                total_steps += 1
                
                metrics = {
                    "loss/train": loss.item(),                          # total loss value (pose_weight * pose_loss + flow_weight * flow_floss)
                    "loss/pose_train": pose_loss.item(),                # pose loss (rotation + translation)
                    "loss/rotation_train": ro.float().mean().item(),    # rotation loss
                    "loss/translation_train": tr.float().mean().item(), # translation loss
                    "loss/flow_train": flow_loss.item(),                # flow loss
                    "loss/scores_train": scores_loss.item(),            # scores loss
                    "px1": (e < .25).float().mean().item(),             # AUC
                    "r1": (ro < .001).float().mean().item(),            # fraction of frames rotation error < 0.001 degrees
                    "r2": (ro < .01).float().mean().item(),             # fraction of frames rotation error < 0.01 degrees
                    "t1": (tr < .001).float().mean().item(),            # fraction of frames translation error < 0.001 m
                    "t2": (tr < .01).float().mean().item(),             # fraction of frames translation error < 0.01 m
                }

                if rank == 0:
                    # log to tensorboard
                    #RESULTS ON TRAINING LOSSES UPDATE AT every SUM_FREQ (inside logger class)
                    logger.push(metrics)


                #SAVING CHECKPOINTS AND EVALUATION ON TARTANEVENT
                if total_steps % args.eval_step == 0:
                    torch.cuda.empty_cache()
                        
                    if args.eval:
                        print("Evaluating...")
                        if args.evs:
                            if "tartan" in args.val_split:
                                #default : use tartan air for validation set
                                from evals.eval_evs.eval_tartan_evs import evaluate as eval_tartan_evs
                                val_results, val_figures = eval_tartan_evs(None, args, net.module if args.ddp else net, total_steps,
                                                                        args.datapath, args.val_split, return_figure=True, plot=True, rpg_eval=False,
                                                                        scale=args.scale, expname=args.name, **kwargs_net)
                            if "mvsec" in args.val_split:
                                #use mvsec as validation set, if we are using the full training set of tartanair
                                from evals.eval_evs.eval_mvsec_evs_validation import evaluate as eval_mvsec_evs
                                #for compatibility, add the state args.model that is args.patchifier_model
                                args.model = args.patchifier_model
                                trials = 3
                                mvsec_datapath = "/capstor/scratch/cscs/amarchei/mvsec/indoor_flying"
                                val_results, val_figures = eval_mvsec_evs(None, args, net.module if args.ddp else net, total_steps,
                                                                        mvsec_datapath, args.val_split, trials, return_figure=True, plot=False, rpg_eval=False,
                                                                        expname=args.name, **kwargs_net)

                        if rank == 0:
                            logger.write_dict(val_results)
                            logger.write_figures(val_figures)
                        
                    torch.cuda.empty_cache()
                    net.train()
                
                if total_steps % args.checkpoint_step == 0:
                    if rank == 0:
                        PATH = 'checkpoints/%s/%06d.pth' % (args.name, args.gpu_num * total_steps)
                        torch.save({
                            'steps': total_steps,
                            'model_state_dict': net.module.state_dict() if args.ddp else net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict()}, PATH)
                
                if args.profiler:
                    prof.step()
            
                if total_steps >= args.steps:
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
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config/DEVO_base_update.yaml'
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

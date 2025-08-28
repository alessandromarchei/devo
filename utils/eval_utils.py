
import os
import torch
from devo.devo import DEVO
from devo.utils import Timer
from pathlib import Path
import datetime
import numpy as np
import yaml
import glob
from itertools import chain
from natsort import natsorted
import copy
import math
import shutil
from scipy.spatial.transform import Rotation as R
from tabulate import tabulate
import matplotlib

from devo.plot_utils import plot_trajectory, fig_trajectory
from devo.plot_utils import save_trajectory_tum_format, plot_trajectory_zx

from utils.viz_utils import show_image, visualize_voxel, save_event_frame_png

from evo.tools import file_interface
import evo.main_ape as main_ape
from evo.core import sync, metrics
from evo.core.trajectory import PoseTrajectory3D
from tqdm import tqdm

# [DEBUG]
# import matplotlib.pyplot as plt
# plt.switch_backend('Qt5Agg')
# plt.figure()
# plt.grid(False)
# plt.imshow(image.detach().cpu().numpy().transpose(1,2,0))
# plt.show()


"""
    landing and takeoff frames of MVSEC:
    indoor 1 : takeoff 150, landing 2060 (total frames 2205) ==> skip_end = 145
    indoor 2 : takeoff 280, landing 2500 (total frames 2664) ==> skip_end = 164
    indoor 3 : takeoff 250, landing 2800 (total frames 2951) ==> skip_end = 151
    indoor 4 : takeoff 250, landing 550 (total frames 622) ==> skip_end = 75
"""

#CREATE A LIST WITH THE SKIP START AND SKIP END BASED ON THE SCENE NAME
skip_ranges_mvsec = {
    "indoor_flying1": (150, 145),
    "indoor_flying2": (280, 164),
    "indoor_flying3": (250, 151),
    "indoor_flying4": (250, 75)
}

@torch.no_grad()
def run_rgb(imagedir, cfg, network, viz=False, iterator=None, timing=False, H=480, W=640, viz_flow=False): 
    slam = DEVO(cfg, network, ht=H, wd=W, viz=viz, viz_flow=viz_flow)
    
    for i, (image, intrinsics, t) in enumerate(iterator):
        if timing and i == 0:
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()

        if viz: 
            show_image(image, 1)
        
        with Timer("DPVO", enabled=False):
            slam(t, image, intrinsics)

    for _ in range(12):
        slam.update()

    poses, tstamps = slam.terminate()

    if timing:
        t1.record()
        torch.cuda.synchronize()
        dt = t0.elapsed_time(t1)/1e3
        print(f"{imagedir}\nDPVO Network {i+1} frames in {dt} sec, e.g. {(i+1)/dt} FPS")
    
    flowdata = slam.flow_data if viz_flow else None
    return poses, tstamps, flowdata


@torch.no_grad()
def run_voxel_debug(voxeldir, cfg, network, viz=False, iterator=None, timing=False, H=480, W=640, viz_flow=False, scale=1.0, model="DEVO",use_pyramid=True, **kwargs): 
    
    print(f"use_pyramid: {use_pyramid}")
    from devo.devo_debug import DEVO
    
    slam = DEVO(cfg, network, evs=True, ht=H, wd=W, viz=viz, viz_flow=viz_flow, model=model, pyramid=use_pyramid, **kwargs)
    
    for i, (voxel, intrinsics, t) in enumerate(tqdm(iterator)):
        if timing and i == 0:
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()

        #if viz: 
        #    # import matplotlib.pyplot as plt
        #    # plt.switch_backend('Qt5Agg')
        #    visualize_voxel(voxel.detach().cpu(), save=True, index=i)
        
        with Timer("DEVO", enabled=timing):
            slam(t, voxel, intrinsics, scale=scale)

    for _ in range(12):
        slam.update()

    poses, tstamps, max_nedges = slam.terminate()

    if timing:
        t1.record()
        torch.cuda.synchronize()
        dt = t0.elapsed_time(t1)/1e3
        print(f"{voxeldir}\nDEVO Network {i+1} frames in {dt} sec, e.g. {(i+1)/dt} FPS")
    
    flowdata = slam.flow_data if viz_flow else None
    return poses, tstamps, flowdata, max_nedges


@torch.no_grad()
def run_voxel_norm_seq(voxeldir, cfg, network, viz=False, iterator=None, timing=False, H=480, W=640, viz_flow=False, scale=1.0, N_norm=15, **kwargs): 
    slam = DEVO(cfg, network, evs=True, ht=H, wd=W, viz=viz, viz_flow=viz_flow, **kwargs)
    
    voxels = []
    tss = []
    for i, (voxel, intrinsics, t) in enumerate(iterator):
        if i == 0 or i % N_norm != 0:
            voxels.append(voxel)
            tss.append(t)
            continue
        else:
            voxels = [v.unsqueeze(0) for v in voxels]
            voxels = torch.cat(voxels, dim=0)
            n, ch, h, w = voxels.shape

            flatten_image = torch.clone(voxels).view(n,-1)
            pos = flatten_image > 0.0
            neg = flatten_image < 0.0
            vx_max = torch.Tensor([1]).to("cuda") if pos.sum().item() == 0 else flatten_image[pos].max(dim=-1, keepdim=True)[0]
            vx_min = torch.Tensor([1]).to("cuda") if neg.sum().item() == 0 else flatten_image[neg].min(dim=-1, keepdim=True)[0]
            if vx_min.item() == 0.0 or vx_max.item() == 0.0:
                print(f"empty voxel at {t}!")
            flatten_image[pos] = flatten_image[pos] / vx_max
            flatten_image[neg] = flatten_image[neg] / -vx_min
            voxels = flatten_image.view(n,ch,h,w)
            
            for t, vox in zip(tss, voxels):
                slam(t, vox, intrinsics, scale=scale)
            voxels = []
            tss = []

    for _ in range(12):
        slam.update()

    poses, tstamps = slam.terminate()

    flowdata = slam.flow_data if viz_flow else None
    return poses, tstamps, flowdata

@torch.no_grad()
def run_voxel(voxeldir, cfg, network, viz=False, iterator=None, timing=False, H=480, W=640, viz_flow=False, scale=1.0, model="DEVO",use_pyramid=True, **kwargs): 
    
    print(f"use_pyramid: {use_pyramid}")

    slam = DEVO(cfg, network, evs=True, ht=H, wd=W, viz=False, viz_flow=viz_flow, model=model, pyramid=use_pyramid, **kwargs)
    
    for i, (voxel, intrinsics, t) in enumerate(tqdm(iterator)):
        if timing and i == 0:
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()

        if viz: 
            # import matplotlib.pyplot as plt
            # plt.switch_backend('Qt5Agg')
            visualize_voxel(voxel.detach().cpu(), save=True, index=i)
        
        with Timer("DEVO", enabled=timing):
            slam(t, voxel, intrinsics, scale=scale)

    for _ in range(12):
        slam.update()

    poses, tstamps, max_nedges = slam.terminate()

    if timing:
        t1.record()
        torch.cuda.synchronize()
        dt = t0.elapsed_time(t1)/1e3
        print(f"{voxeldir}\nDEVO Network {i+1} frames in {dt} sec, e.g. {(i+1)/dt} FPS")
    
    flowdata = slam.flow_data if viz_flow else None
    return poses, tstamps, flowdata, max_nedges



@torch.no_grad()
def run_voxel_advanced(voxeldir, cfg, network, viz=False, iterator=None, timing=False, H=480, W=640, viz_flow=False, scale=1.0, model="DEVO",use_pyramid=True, **kwargs): 
    
    if kwargs.get("devo_debug", False):
        from devo.devo_debug import DEVO
    else:
        from devo.devo import DEVO


    print(f"use_pyramid: {use_pyramid}")
    skip_start = kwargs.get("skip_start", 0)
    skip_end = kwargs.get("skip_end", 0)

    kwargs['skip_start'] = skip_start
    kwargs['skip_end'] = skip_end    

    print(f"Voxel dir : {voxeldir}")
    #check if mvsec is in the name of the voxeldir
    if ('mvsec' in voxeldir) and (skip_end != 0 or skip_start != 0):
        #check which scene is being processed
        if('indoor_flying1' in voxeldir):
            skip_start, skip_end = skip_ranges_mvsec["indoor_flying1"]
            print("Processing indoor_flying1, skipping first 150 frames and last 145 frames")
        elif('indoor_flying2' in voxeldir):
            skip_start, skip_end = skip_ranges_mvsec["indoor_flying2"]
            print("Processing indoor_flying2, skipping first 280 frames and last 164 frames")
        elif('indoor_flying3' in voxeldir):
            skip_start, skip_end = skip_ranges_mvsec["indoor_flying3"]
            print("Processing indoor_flying3, skipping first 250 frames and last 151 frames")
        elif('indoor_flying4' in voxeldir):
            skip_start, skip_end = skip_ranges_mvsec["indoor_flying4"]
            print("Processing indoor_flying4, skipping first 250 frames and last 75 frames")

    data_list = list(iterator)
    total_len = len(data_list)
    print(f"Total frames in iterator: {total_len}, skipping first {skip_start} and last {skip_end} frames")
    
    slam = DEVO(cfg, network, evs=True, ht=H, wd=W, viz=False, viz_flow=viz_flow, model=model, pyramid=use_pyramid, **kwargs)
    
    for i, (voxel, intrinsics, t) in enumerate(tqdm(data_list)):
        if i < skip_start or i >= (total_len - skip_end):
            #print(f"Skipping frame {i} at timestamp {t}, total frames: {total_len}")
            continue

        if timing and i == 0:
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
        
        with Timer("DEVO", enabled=timing):
            slam(t, voxel, intrinsics, scale=scale)

    for _ in range(12):
        slam.update()

    poses, tstamps, max_nedges = slam.terminate()

    if timing:
        t1.record()
        torch.cuda.synchronize()
        dt = t0.elapsed_time(t1)/1e3
        print(f"{voxeldir}\nDEVO Network {i+1} frames in {dt} sec, e.g. {(i+1)/dt} FPS")
    
    flowdata = slam.flow_data if viz_flow else None
    return poses, tstamps, flowdata, max_nedges


@torch.no_grad()
def save_sequence(iterator=None,  save=True): 
        
    for i, (voxel, intrinsics, t) in enumerate(tqdm(iterator)):
        print(f"Saving voxel {i} at {t}")
        if save:
            save_event_frame_png(voxel.detach().cpu(), save=True, folder="event_frames", index=i)





@torch.no_grad()
def run_voxel_validation(voxeldir, cfg, network, viz=False, iterator=None, timing=False, H=480, W=640, viz_flow=False, scale=1.0, model="DEVO",use_pyramid=True, **kwargs): 
    
    print(f"use_pyramid: {use_pyramid}")

    slam = DEVO(cfg, network, evs=True, ht=H, wd=W, viz=False, viz_flow=viz_flow, model=model, pyramid=use_pyramid, **kwargs)
    
    for i, (voxel, intrinsics, t) in enumerate(tqdm(iterator)):
        if timing and i == 0:
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()

        if viz: 
            # import matplotlib.pyplot as plt
            # plt.switch_backend('Qt5Agg')
            visualize_voxel(voxel.detach().cpu(), save=True, index=i)
        
        with Timer("DEVO", enabled=timing):
            slam(t, voxel, intrinsics, scale=scale)

    for _ in range(12):
        slam.update()

    poses, tstamps, _ = slam.terminate()

    if timing:
        t1.record()
        torch.cuda.synchronize()
        dt = t0.elapsed_time(t1)/1e3
        print(f"{voxeldir}\nDEVO Network {i+1} frames in {dt} sec, e.g. {(i+1)/dt} FPS")
    
    flowdata = slam.flow_data if viz_flow else None
    return poses, tstamps, flowdata



def assert_eval_config(args):
    assert os.path.isfile(args.weights) and (".pth" in args.weights or ".pt" in args.weights)
    assert os.path.isfile(args.val_split)
    assert args.trials > 0

def ate(traj_ref, traj_est, timestamps):
    import evo
    import evo.main_ape as main_ape
    from evo.core.trajectory import PoseTrajectory3D
    from evo.core.metrics import PoseRelation

    traj_est = PoseTrajectory3D(
        positions_xyz=traj_est[:,:3],
        orientations_quat_wxyz=traj_est[:,3:], # TODO wrong format: EVO uses wxyz, we use xyzw
        timestamps=timestamps)

    traj_ref = PoseTrajectory3D(
        positions_xyz=traj_ref[:,:3],
        orientations_quat_wxyz=traj_ref[:,3:],  # TODO wrong format: EVO uses wxyz, we use xyzw
        timestamps=timestamps)
    
    result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
        pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)

    return result.stats["rmse"]

def get_alg(n):
    if n == "eds" or n == "tumvie" or n == "tartanair":
        return "rgb"
    elif n == "eds_evs" or n == "tumvie_evs" or n == "tartanair_evs":
        return "evs"
    elif n == "eds_evs_viz" or n == "tumvie_evs_viz" or n == "tartanair_evs_viz":
        return "evs_viz"

def make_outfolder(outdir, dataset_name, expname, scene_name, trial, train_step, stride, calib1_eds, camID_tumvie):
    date = datetime.datetime.today().strftime('%Y-%m-%d') # TODO improve output folder
    outfolder = os.path.join(f"{outdir}/{dataset_name}/{date}_{expname}/{scene_name}_trial_{trial}_step_{train_step}")
    if stride != 1:
        outfolder = outfolder + f"_stride_{stride}"
    if calib1_eds != None:
        outfolder = outfolder + f"_calib1" if calib1_eds else outfolder + f"_calib0"
    if camID_tumvie != None:
        outfolder = outfolder + f"_camID_{camID_tumvie}"
    outfolder = os.path.abspath(outfolder)
    os.makedirs(outfolder, exist_ok=True)
    return outfolder

def make_outfolder_outdir(outdir, dataset_name, expname, scene_name, trial, cfg):
    outfolder = os.path.join(f"{outdir}/{dataset_name}/{expname}/{scene_name}_trial_{trial}_cfg_{cfg['PATCHES_PER_FRAME']}_{cfg['REMOVAL_WINDOW']}_{cfg['PATCH_LIFETIME']}")

    outfolder = os.path.abspath(outfolder)
    os.makedirs(outfolder, exist_ok=True)
    return outfolder


def run_rpg_eval(outfolder, traj_ref, tss_ref_us, traj_est, tstamps):
    p = f"{outfolder}/"
    p = os.path.abspath(p)
    os.makedirs(p, exist_ok=True)

    fnameGT = os.path.join(p, "stamped_groundtruth.txt")
    f = open(fnameGT, "w")
    f.write("# timestamp[secs] tx ty tz qx qy qz qw\n")
    for i in range(len(traj_ref)):
        f.write(f"{tss_ref_us[i]/1e6} {traj_ref[i,0]} {traj_ref[i,1]} {traj_ref[i,2]} {traj_ref[i,3]} {traj_ref[i,4]} {traj_ref[i,5]} {traj_ref[i,6]}\n")
    f.close()

    fnameEst = os.path.join(p, "stamped_traj_estimate.txt")
    f = open(fnameEst, "w")
    f.write("# timestamp[secs] tx ty tz qx qy qz qw\n")
    for i in range(len(traj_est)):
        f.write(f"{tstamps[i]/1e6} {traj_est[i,0]} {traj_est[i,1]} {traj_est[i,2]} {traj_est[i,3]} {traj_est[i,4]} {traj_est[i,5]} {traj_est[i,6]}\n")
    f.close()
    
    # cmd = f"python thirdparty/rpg_trajectory_evaluation/scripts/analyze_trajectory_single.py --result_dir {p} --recalculate_errors --png --plot"
    cmd = f"python thirdparty/rpg_trajectory_evaluation/scripts/analyze_trajectory_single.py {p} --recalculate_errors --png --plot"
    os.system(cmd)

    return fnameGT, fnameEst

def load_stats_rpg_results(outfolder):
    rpg_fspath = os.path.join(outfolder, "saved_results/traj_est")

    absfile = natsorted(glob.glob(os.path.join(rpg_fspath, "absolute_err_stat*.yaml")))[-1]
    with open(absfile, 'r') as file:
        abs_stats = yaml.safe_load(file)

    last_relfile = natsorted(glob.glob(os.path.join(rpg_fspath, "relative_error_statistics_*.yaml")))[-1]
    with open(last_relfile, 'r') as file:
        rel_stats = yaml.safe_load(file)

    # last_relfile_time = natsorted(glob.glob(os.path.join(rpg_fspath, "Time_relative_error_statistics_*.yaml")))[-1]
    # with open(last_relfile_time, 'r') as file:
    #     rel_stats_time = yaml.safe_load(file)
    rel_stats_time = copy.deepcopy(rel_stats) 
    
    return abs_stats, rel_stats, rel_stats_time

def remove_all_patterns_from_str(s, patterns):
    for pattern in patterns:
        if pattern in s:
            s = s.replace(pattern, "")
    return s

def remove_row_from_table(table_string, row_index):
    rows = table_string.split('\n')
    if row_index < len(rows):
        del rows[row_index]
    return '\n'.join(rows)

def dict_to_table(data, scene, header=True):
    table_data = [["Scene", *data.keys()], [f"{scene}", *data.values()]]
    table_data = [row + ["\\\\"] for row in table_data]

    table = tabulate(table_data, tablefmt="plain")

    if not header:
        table = remove_row_from_table(table, 0)

    return table

def write_res_table(outfolder, res_str, scene_name, trial):
    res = res_str.split("|")
    res_dict = {}
    for r in res:
        k = r.split(":")[0]
        patterns_to_remove = ["\n", " ", ")", "("]
        k = remove_all_patterns_from_str(k, patterns_to_remove)

        v = r.split(":")[1]
        v = remove_all_patterns_from_str(v, patterns_to_remove)
        res_dict[k] = float(v)

    summtable_fnmae = os.path.join(outfolder, "../0_res.txt")
    if not os.path.isfile(summtable_fnmae): 
        f = open(summtable_fnmae, "w")
    else:
        f = open(summtable_fnmae, "a")
    if trial == 0:
        f.write("\n")

    table = dict_to_table(res_dict, scene_name, trial==0)
    f.write(table)
    f.write("\n")
    f.close()


def ate_real(traj_ref, tss_ref_us, traj_est, tstamps):
    evoGT = PoseTrajectory3D(
        positions_xyz=traj_ref[:,:3],
        orientations_quat_wxyz=traj_ref[:,3:], # TODO wrong format: EVO uses wxyz, we use xyzw
        timestamps=tss_ref_us/1e6)

    evoEst = PoseTrajectory3D(
        positions_xyz=traj_est[:,:3],
        orientations_quat_wxyz=traj_est[:,3:], # TODO wrong format: EVO uses wxyz, we use xyzw
        timestamps=tstamps/1e6)

    if traj_ref.shape == traj_est.shape:
        assert np.all(tss_ref_us == tstamps)
        return ate(traj_ref, traj_est, tstamps)*100, evoGT, evoEst
    
    evoGT, evoEst = sync.associate_trajectories(evoGT, evoEst, max_diff=1)
    ape_trans = main_ape.ape(evoGT, evoEst, pose_relation=metrics.PoseRelation.translation_part, align=True, correct_scale=True)
    evoATE = ape_trans.stats["rmse"]*100
    return evoATE, evoGT, evoEst


def make_evo_traj(poses_N_x_7, tss_us):
    assert poses_N_x_7.shape[1] == 7
    assert poses_N_x_7.shape[0] > 10
    assert tss_us.shape[0] == poses_N_x_7.shape[0]

    traj_evo = PoseTrajectory3D(
        positions_xyz=poses_N_x_7[:,:3],
        orientations_quat_wxyz=poses_N_x_7[:,3:],
        timestamps=tss_us/1e6)
    return traj_evo

@torch.no_grad()            
def log_results(data, hyperparam, all_results, results_dict_scene, figures, 
                plot=False, save=True, return_figure=False, rpg_eval=False, stride=1, 
                calib1_eds=None, camID_tumvie=None, outdir=None, expname="", max_diff_sec=0.01, save_csv=False, cfg=None, name=None, step=None):
    # results: dict of (scene, list of results)
    # all_results: list of all raw_results

    # unpack data
    traj_GT, tss_GT_us, traj_est, tss_est_us = data
    train_step, net, dataset_name, scene, trial, cfg, args, max_nedges = hyperparam

    ####### SINCE EVALUATION DATA HAS BEEN SHORTENED DUE TO MEMORY ISSUES,
    ####### THE GT DATA should be truncated as the traj_est length
    ####### ONLY FOR TARTANAIR DATASET
    # if 'tartan' in dataset_name.lower():
    #     traj_GT = traj_GT[:len(traj_est)]
    #     tss_GT_us = tss_GT_us[:len(traj_est)]
    

    # create folders
    if train_step is None:
        if isinstance(net, str) and ".pth" in net:
            train_step = os.path.basename(net.split(".")[0])
        else:
            train_step = -1
    scene_name = '_'.join(scene.split('/')[1:]).title() if "/P0" in scene else scene.title()
    if outdir is None:
        outdir = "results"

    if outdir is None:
        outfolder = make_outfolder(outdir, dataset_name, expname, scene_name, trial, train_step, stride, calib1_eds, camID_tumvie)
    else:
        outfolder = make_outfolder_outdir(outdir, dataset_name, expname, scene_name, trial, cfg)

    # save cfg & args to outfolder
    if cfg is not None:
        with open(f"{outfolder}/cfg.yaml", 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)
    if args is not None:
        if args is not None:
            with open(f"{outfolder}/args.yaml", 'w') as f:
                yaml.dump(vars(args), f, default_flow_style=False)

    # compute ATE
    ate_score, evoGT, evoEst = ate_real(traj_GT, tss_GT_us, traj_est, tss_est_us)
    all_results.append(ate_score)
    results_dict_scene[scene].append(ate_score)
    
    # following https://github.com/arclab-hku/Event_based_VO-VIO-SLAM/issues/5
    evoGT = make_evo_traj(traj_GT, tss_GT_us)
    evoEst = make_evo_traj(traj_est, tss_est_us)
    gtlentraj = evoGT.get_infos()["path length (m)"]
    evoGT, evoEst = sync.associate_trajectories(evoGT, evoEst, max_diff=1)
    ape_trans = main_ape.ape(copy.deepcopy(evoGT), copy.deepcopy(evoEst), pose_relation=metrics.PoseRelation.translation_part, align=True, correct_scale=True)
    MPE = ape_trans.stats["mean"] / gtlentraj * 100
    evoATE = ape_trans.stats["rmse"]*100
    assert abs(evoATE-ate_score) < 1e-5
    R_rmse_deg = -1.0

    if save:
        Path(f"{outfolder}").mkdir(exist_ok=True)
        save_trajectory_tum_format((traj_est, tss_est_us), f"{outfolder}/{scene_name}_Trial{trial+1:02d}.txt")

    if rpg_eval:

        fnamegt, fnameest = run_rpg_eval(outfolder, traj_GT, tss_GT_us, traj_est, tss_est_us)
        abs_stats, rel_stats, _ = load_stats_rpg_results(outfolder)

        # abs errs
        ate_rpg = abs_stats["trans"]["rmse"]*100
        print(f"ate_rpg: {ate_rpg:.04f}, ate_real (EVO): {ate_score:.04f}")
        # assert abs(ate_rpg-ate_score)/ate_rpg < 0.1 # 10%
        R_rmse_deg = abs_stats["rot"]["rmse"]
        MTE_m = abs_stats["trans"]["mean"]

        # traj_GT_inter = interpolate_traj_at_tss(traj_GT, tss_GT_us, tss_est_us)
        # ate_inter, _, _ = ate_real(traj_GT_inter, tss_est_us, traj_est, tss_est_us)
        
        res_str = f"\nATE[cm]: {ate_score:.03f} | R_rmse[deg]: {R_rmse_deg:.03f} | MPE[%/m]: {MPE:.03f} \n"
        # res_str += f"MTE[m]: {MTE_m:.03f} | (ATE_int[cm]: {ate_inter:.02f} | ATE_rpg[cm]: {ate_rpg:.02f}) \n"
        
                    #save on results.csv file, in append mode, the following information based on the scene
            #dataset, scene, patches, opt_window,Rem_window,patch_lt, ate, rot_error_x,rot_error_y, rot_error_z
        if cfg is not None and save_csv is not False:
            if name is None:
                name = "out_eval.csv"
            with open(name, "a") as f:
                f.write(
                    f"{'mvsec'},{scene_name},{cfg['PATCHES_PER_FRAME']},{cfg['OPTIMIZATION_WINDOW']},{cfg['REMOVAL_WINDOW']},{cfg['PATCH_LIFETIME']},{max_nedges},{ate_score},{R_rmse_deg},{MPE}\n")




        write_res_table(outfolder, res_str, scene_name, trial)
    else:
        p = f"{outfolder}/"
        p = os.path.abspath(p)
        os.makedirs(p, exist_ok=True)

        
        fnameGT = os.path.join(p, "stamped_groundtruth.txt")
        f = open(fnameGT, "w")
        f.write("# timestamp[secs] tx ty tz qx qy qz qw\n")
        for i in range(len(traj_GT)):
            f.write(f"{tss_GT_us[i]/1e6} {traj_GT[i,0]} {traj_GT[i,1]} {traj_GT[i,2]} {traj_GT[i,3]} {traj_GT[i,4]} {traj_GT[i,5]} {traj_GT[i,6]}\n")
        f.close()

        fnameEst = os.path.join(p, "stamped_traj_estimate.txt")
        f = open(fnameEst, "w")
        f.write("# timestamp[secs] tx ty tz qx qy qz qw\n")
        for i in range(len(traj_est)):
            f.write(f"{tss_est_us[i]/1e6} {traj_est[i,0]} {traj_est[i,1]} {traj_est[i,2]} {traj_est[i,3]} {traj_est[i,4]} {traj_est[i,5]} {traj_est[i,6]}\n")
        f.close()


        res_str = f"\nATE[cm]: {ate_score:.03f} | MPE[%/m]: {MPE:.03f}"


        write_res_table(outfolder, res_str, scene_name, trial)

    if plot and outdir is None:
        Path(f"{outfolder}/").mkdir(exist_ok=True)
        pdfname = f"{outfolder}/../{scene_name}_Trial{trial+1:02d}_exp_{expname}_step_{train_step}_stride_{stride}.pdf"
        plot_trajectory((traj_est, tss_est_us/1e6), (traj_GT, tss_GT_us/1e6), 
                        f"{dataset_name} {expname} {scene_name.replace('_', ' ')} Trial #{trial+1} {res_str}",
                        pdfname, align=True, correct_scale=True, max_diff_sec=max_diff_sec)
        shutil.copy(pdfname, f"{outfolder}/{scene_name}_Trial{trial+1:02d}_step_{train_step}_stride_{stride}.pdf")
        # [DEBUG]
        #pdfname = f"{outfolder}/GT_{scene_name}_Trial{trial+1:02d}_exp_{expname}_step_{train_step}_stride_{stride}.pdf"
        #plot_trajectory((traj_GT, tss_GT_us/1e6), (traj_GT, tss_GT_us/1e6), 
        #                f"{dataset_name} {expname} {scene_name.replace('_', ' ')} Trial #{trial+1} {res_str}",
        #                pdfname, align=True, correct_scale=True, max_diff_sec=max_diff_sec)
    
    elif plot and outdir is not None:
        Path(f"{outfolder}/").mkdir(exist_ok=True)
        pdfname = f"{outfolder}/{scene_name}_Trial{trial+1:02d}_exp_{expname}_cfg_{cfg['PATCHES_PER_FRAME']}_{cfg['REMOVAL_WINDOW']}_{cfg['PATCH_LIFETIME']}.pdf"
        plot_trajectory((traj_est, tss_est_us/1e6), (traj_GT, tss_GT_us/1e6), 
                        f"{dataset_name} {expname} {scene_name.replace('_', ' ')} Trial #{trial+1} {res_str}",
                        pdfname, align=True, correct_scale=True, max_diff_sec=max_diff_sec)
        shutil.copy(pdfname, f"{outfolder}/{scene_name}_Trial{trial+1:02d}_cfg_{cfg['PATCHES_PER_FRAME']}_{cfg['REMOVAL_WINDOW']}_{cfg['PATCH_LIFETIME']}.pdf")

    if return_figure:
        fig = fig_trajectory((traj_est, tss_est_us/1e6), (traj_GT, tss_GT_us/1e6), f"{dataset_name} {scene_name.replace('_', ' ')} {res_str})",
                            return_figure=True, max_diff_sec=max_diff_sec)
        figures[f"{dataset_name}_{scene_name}"] = fig
    
    print(f"Results for {dataset_name} {scene_name} Trial #{trial+1} {res_str}")

    return all_results, results_dict_scene, figures, outfolder



@torch.no_grad()
def write_raw_results(all_results, outfolder):
    # all_results: list of all raw_results
    os.makedirs(os.path.join(f"{outfolder}/../raw_results"), exist_ok=True)
    with open(os.path.join(f"{outfolder}/../raw_results", datetime.datetime.now().strftime('%m-%d-%I%p.txt')), "w") as f:
        f.write(','.join([str(x) for x in all_results]))

@torch.no_grad()
def compute_median_results(results, all_results, dataset_name, outfolder=None):
    # results: dict of (scene, list of results)
    # all_results: list of all raw_results
        
    results_dict = dict([(f"{dataset_name}/{k}", np.median(v)) for (k, v) in results.items()])
    results_dict["AUC"] = np.maximum(1 - np.array(all_results), 0).mean()

    xs = []
    for scene in results:
        x = np.median(results[scene])
        xs.append(x)
    results_dict["AVG"] = np.mean(xs) / 100.0 # cm -> m

    if outfolder is not None:
        with open(os.path.join(f"{outfolder}/../results_dict_latex_{datetime.datetime.now().strftime('%m-%d-%I%p.txt')}"), 'w') as f:
            k0 = list(results.keys())[0]
            num_runs = len(results[k0])
            f.write(' & '.join([str(k) for k in results.keys()]))
            f.write('\n')

 
            for i in range(num_runs):
                print(f"{[str(v[i]) for v in results.values()]}")
                f.write(' & '.join([str(v[i]) for v in results.values()]))
                f.write('\n')

            f.write(f"Medians\n")
            for i in range(num_runs):
                print(f"{[str(v[i]) for v in results.values()]}")
                f.write(' & '.join([str(np.median(v)) for v in results.values()]))
                f.write('\n')

            f.write('\n\n')

    return results_dict
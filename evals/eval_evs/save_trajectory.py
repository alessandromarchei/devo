import os
import math
import numpy as np
import os.path as osp

import torch
from devo.config import cfg

from utils.eval_utils import run_voxel, assert_eval_config
from utils.load_utils import voxel_iterator_parallel, voxel_iterator
from utils.eval_utils import log_results, write_raw_results, compute_median_results
from utils.transform_utils import transform_rescale_poses
from utils.viz_utils import viz_flow_inference

@torch.no_grad()
def evaluate(config, args, net, train_step=None, datapath="", split_file=None,
             trials=1, stride=1, plot=False, save=False, return_figure=False, viz=False, timing=False, viz_flow=False, scale=1.0,
             rpg_eval=True, expname="", est_file="pose_left.txt", gt_file="pose_left.txt", use_pyramid=True, **kwargs):
    dataset_name = "tartanair_evs"

    if config is None:
        config = cfg
        config.merge_from_file("config/configs_eval_training.yaml")

    scenes = open(split_file).read().split()

    results_dict_scene, loss_dict_scene, figures = {}, {}, {}
    all_results = []
    for i, scene in enumerate(scenes):
        print(f"Eval on {scene}")
        # scene_name = '_'.join(scene.split('/')[1:]).title() if "/P0" in scene else scene.title()
        # results_dict_scene[f"{dataset_name}/{scene_name}"] = [] # TODO use dataset_name/scene_name?
        #loss_dict_scene[f"{dataset_name}_{scene_name}"] = []
        results_dict_scene[scene] = []

# estimated trajectory
        scene_path = datapath_val
        traj_ref = osp.join(datapath_val, "pose_left.txt")

        #compute path of the evaluation set
        #datapath_val = '/usr/scratch/badile43/amarchei/TartanEvent/abandonedfactory/Easy'
        #scene = 'abandonedfactory/abandonedfactory/Easy/P011'
        # we want scene_path = /usr/scratch/badile43/amarchei/TartanEvent/abandonedfactory/Easy/P011'
        
        traj_est = np.loadtxt(os.path.join(est_file))
        traj_ref = np.loadtxt(os.path.join(gt_file))

        traj_est, tstamps = traj_est[:, 1:], traj_est[:, 0]
        traj_ref, tstamps = traj_ref[:, 1:], traj_ref[:, 0]

        PERM = [1, 2, 0, 4, 5, 3, 6] # ned -> xyz
        # events between two adjacent frames t-1 and t are accumulated in event voxel t -> ignore first pose (t=0)

        FREQ = 50
        # do evaluation 
        data = (traj_ref, tstamps*1e6/FREQ, traj_est, tstamps*1e6/FREQ)
        all_results, results_dict_scene, figures, outfolder = log_results(data, hyperparam, all_results, results_dict_scene, figures, 
                                                                plot=plot, save=save, return_figure=return_figure, rpg_eval=rpg_eval, stride=stride,
                                                                expname=args.expname, save_csv=args.save_csv, cfg=config, name=args.csv_name)
        

    # write output to file with timestamp
    write_raw_results(all_results, outfolder)
    # results_dict = compute_results(results_dict_scene, all_results, loss_dict_scene)
    results_dict = compute_median_results(results_dict_scene, all_results, dataset_name)
        
    if return_figure:
        return results_dict, figures
    return results_dict, None


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="config/default.yaml")
    parser.add_argument('--datapath', default='/usr/scratch/badile43/amarchei/TartanEvent', help='path to dataset directory')
    parser.add_argument('--weights', default="DEVO.pth")
    parser.add_argument('--val_split', type=str, default="splits/tartan/tartan_default_val.txt")
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--save_trajectory', action="store_true")
    parser.add_argument('--return_figs', action="store_true")
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--timing', action="store_true")
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--viz_flow', action="store_true")
    parser.add_argument('--expname', type=str, default="viz_scorer")
    parser.add_argument('--rpg_eval', action="store_true", help='advanced eval')
    parser.add_argument('--est_file', type=str, default="pose_left.txt", help='file with estimated poses')
    parser.add_argument('--gt_file', type=str, default="pose_left.txt", help='file with ground truth poses')

    args = parser.parse_args()
    assert_eval_config(args)

    cfg.merge_from_file(args.config)
    print("Running eval_tartan_evs.py with config...")
    print(cfg) 

    torch.manual_seed(1234)
    
    val_results, val_figures = evaluate(cfg, args, args.weights, datapath=args.datapath, split_file=args.val_split, trials=args.trials, \
                        plot=args.plot, save=args.save_trajectory, return_figure=args.return_figs, viz=args.viz, timing=args.timing, \
                        stride=args.stride, viz_flow=args.viz_flow, rpg_eval=args.rpg_eval, expname=args.expname, est_file=args.est_file, gt_file=args.gt_file)
    
    print("val_results= \n")
    for k in val_results:
        print(k, val_results[k])

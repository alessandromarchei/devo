import os
import torch
from devo.config import cfg

from utils.load_utils import load_mvsec_traj, mvsec_evs_iterator
from utils.eval_utils import assert_eval_config, run_voxel_validation
from utils.eval_utils import log_results, write_raw_results, compute_median_results
# from utils.viz_utils import viz_flow_inference

H, W = 260, 346

@torch.no_grad()
def evaluate(config, args, net, train_step=None, datapath="", split_file=None, 
             trials=1, stride=1, plot=False, save=False, return_figure=False, viz=False, timing=False, side='left', viz_flow=False, use_pyramid=True,
             expname='',**kwargs):
    dataset_name = "mvsec_evs"
    assert side == "left" or side == "right"
    assert stride == 1 # != does not work yet

    if config is None:
        config = cfg
        config.merge_from_file("config/eval_mvsec.yaml")
        
    scenes = open(split_file).read().split()

    results_dict_scene, figures = {}, {}
    all_results = []
    for i, scene in enumerate(scenes):
        print(f"Eval on {scene}")
        results_dict_scene[scene] = []

        for trial in range(trials):
            # estimated trajectory
            datapath_val = os.path.join(datapath, scene)

            # run the slam system
            traj_est, tstamps, flowdata = run_voxel_validation(datapath_val, config, net, viz=viz, 
                                          iterator=mvsec_evs_iterator(datapath_val, side=side, stride=stride, timing=timing, H=H, W=W),
                                          timing=timing, H=H, W=W, viz_flow=viz_flow, use_pyramid=use_pyramid, model=args.model, **kwargs)

            # load traj
            tss_traj_us, traj_hf = load_mvsec_traj(datapath_val)


            # do evaluation 
            #extract the number from the net word. an example is this : checkpoints/baseline_320x240/200000.pth, so extract 200000
            if train_step is None:
                train_step = int(net.split("/")[-1].split(".")[0])
                if train_step == 0:
                    train_step = 1
                print("train_step", train_step)
                


            data = (traj_hf, tss_traj_us, traj_est, tstamps)
            max_edges = 0  # max_nedges is not used in this context
            hyperparam = (train_step, net, dataset_name, scene, trial, cfg, args, max_edges)  # max_nedges is not used in this context
            all_results, results_dict_scene, figures, outfolder = log_results(data, hyperparam, all_results, results_dict_scene, figures, 
                                                                   plot=plot, save=save, return_figure=return_figure, stride=stride,
                                                                   expname=expname, step=train_step)
            
            # if viz_flow:
            #     viz_flow_inference(outfolder, flowdata)
            
        print(scene, sorted(results_dict_scene[scene]))

    # write output to file with timestamp
    write_raw_results(all_results, outfolder)
    results_dict = compute_median_results(results_dict_scene, all_results, dataset_name)
        
    if return_figure:
        return results_dict, figures
    return results_dict, None


if __name__ == '__main__': 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="config/eval_mvsec.yaml")
    parser.add_argument('--datapath', default='/usr/scratch/badile13/amarchei/mvsec/indoor_flying', help='path to dataset directory')
    parser.add_argument('--weights', default="DEVO.pth")
    parser.add_argument('--val_split', type=str, default="splits/mvsec/mvsec_val.txt")
    parser.add_argument('--trials', type=int, default=2)
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--save_trajectory', action="store_true")
    parser.add_argument('--return_figs', action="store_true")
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--timing', action="store_true")
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--side', type=str, default="left")
    parser.add_argument('--viz_flow', action="store_true")
    parser.add_argument('--expname', type=str, default="")
    parser.add_argument('--save_csv', action="store_true")
    parser.add_argument('--csv_name', type=str, default="")
    parser.add_argument('--model', type=str, default="original")
    parser.add_argument('--dim_inet', type=int, default=384, help='channel dimension of hidden state')
    parser.add_argument('--dim_fnet', type=int, default=128, help='channel dimension of last layer fnet')
    parser.add_argument(
        '--use_pyramid',
        type=lambda x: x.lower() == 'true',
        default=True,
        help='use pyramid (default: True)'
    )
    parser.add_argument(
        '--use_softagg',
        type=lambda x: x.lower() == 'true',
        default=True,
        help='use softagg (default: True)'
    )
    parser.add_argument(
        '--use_tempconv',
        type=lambda x: x.lower() == 'true',
        default=True,
        help='use tempconv (default: True)'
    )
    args = parser.parse_args()
    assert_eval_config(args)

    cfg.merge_from_file(args.config)
    print("Running eval_MVSEC_evs.py with config...")
    print(cfg) 

    torch.manual_seed(1234)

    # args.save_trajectory = True
    # args.plot = True
    kwargs = {"dim_inet": args.dim_inet, "dim_fnet": args.dim_fnet}

    val_results, val_figures = evaluate(cfg, args, args.weights, datapath=args.datapath, split_file=args.val_split, trials=args.trials, \
                       plot=args.plot, save=args.save_trajectory, return_figure=args.return_figs, viz=args.viz,timing=args.timing, \
                        stride=args.stride, side=args.side, viz_flow=args.viz_flow, use_pyramid=args.use_pyramid,**kwargs)
    
    print("val_results= \n")
    for k in val_results:
        print(k, val_results[k])
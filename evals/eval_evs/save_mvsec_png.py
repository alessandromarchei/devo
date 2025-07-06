import os
import torch
from devo.config import cfg

from utils.load_utils import mvsec_evs_iterator
from utils.eval_utils import assert_eval_config, save_sequence
# from utils.viz_utils import viz_flow_inference

H, W = 260, 346

if __name__ == '__main__': 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', default='/usr/scratch/badile13/amarchei/mvsec/indoor_flying', help='path to dataset directory')
    parser.add_argument('--val_split', type=str, default="splits/mvsec/mvsec_val.txt")
    parser.add_argument('--side', type=str, default="left")
    parser.add_argument('--stride', type=int, default=1)

    args = parser.parse_args()

    assert args.side == "left" or args.side == "right"
    stride = 1
    
    scenes = open(args.val_split).read().split()

    for i, scene in enumerate(scenes):
        print(f"Eval on {scene}")


        datapath_val = os.path.join(args.datapath, scene)

        # run the slam system
        save_sequence(iterator=mvsec_evs_iterator(datapath_val, side=args.side, stride=stride, timing=False, H=H, W=W))

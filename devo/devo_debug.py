import torch
import numpy as np
import torch.nn.functional as F
import csv
from . import fastba
from . import altcorr
from . import lietorch
from .lietorch import SE3, SO3
import time
from .ba import BA
# from .net import VONet # TODO add net.py
from .enet_debug import eVONet
from .utils import log_stats

from .utils import *
from . import projective_ops as pops

autocast = torch.cuda.amp.autocast
Id = SE3.Identity(1, device="cuda")

from utils.viz_utils import visualize_voxel, show_voxel_coordinates
from scripts.draw_graph import draw_patch_graph
import os



# torch.manual_seed(0)
class DEVO:
    def __init__(self, cfg, network, evs=False, ht=480, wd=640, viz=False, viz_flow=False, ctx_feat_dim=384, match_feat_dim=128, dim=32, model=None, pyramid=True,use_softagg=True, use_tempconv=True,**kwargs):
        
        self.cfg = cfg
        self.evs = evs

        self.kwargs = kwargs
        #extract the logname from the kwargs
        self.logname = kwargs.get("logname", None)
        #check existence of logname and eventually remove the file
        if self.logname is not None and os.path.exists(self.logname):
            print(f"Log file {self.logname} already exists, deleting it.")
            os.remove(self.logname)

        self.ctx_feat_dim = kwargs.get("dim_inet", ctx_feat_dim)
        self.match_feat_dim = kwargs.get("dim_fnet", match_feat_dim)
        print(f"ctx_feat_dim: {self.ctx_feat_dim}, match_feat_dim: {self.match_feat_dim}")

        self.trial = kwargs.get("trial", 0)
        
        self.load_path = self.kwargs.get("path", None)
        self.load_ba = self.kwargs.get("load_ba", False)
        self.load_coords = self.kwargs.get("load_coords", False)
        
        if self.load_ba or self.load_coords:
            self.trial = kwargs.get("trial_num", 1)
            print(f"Loading BA inputs from {self.load_path} for trial {self.trial}")

        self.iteration = kwargs.get("skip_start", 0)

        
        self.dim = dim
        # TODO add patch_selector

        self.model = model
        self.use_pyramid = kwargs.get("use_pyramid", pyramid)
        self.use_softagg = kwargs.get("use_softagg", use_softagg)
        self.use_tempconv = kwargs.get("use_tempconv", use_tempconv)

        print(f"Using pyramid: {self.use_pyramid}")
        print(f"Using softagg: {self.use_softagg}")
        print(f"Using tempconv: {self.use_tempconv}")

        self.load_weights(network)
        self.is_initialized = False
        self.enable_timing = False # TODO timing in param

        self.viz_flow = viz_flow
        
        self.n = 0      # active keyframes/frames (every frames == keyframe)
        self.m = 0      # number active patches
        self.M = self.cfg.PATCHES_PER_FRAME     # (default: 96)
        self.N = self.cfg.BUFFER_SIZE           # max number of keyframes (default: 2048)

        self.ht = ht    # image height
        self.wd = wd    # image width

        RES = self.RES

        ### state attributes ###
        self.tlist = []
        self.counter = 0 # how often this network is called __call__()

        self.flow_data = {}

        # dummy image for visualization
        self.image_ = torch.zeros(self.ht, self.wd, 3, dtype=torch.uint8, device="cpu")

        self.tstamps_ = torch.zeros(self.N, dtype=torch.long, device="cuda")

        self.poses_ = torch.zeros(self.N, 7, dtype=torch.float, device="cuda")
        self.patches_ = torch.zeros(self.N, self.M, 3, self.P, self.P, dtype=torch.float, device="cuda") # 3 channels = (x, y, depth)
        self.patches_gt_ = torch.zeros(self.N, self.M, 3, self.P, self.P, dtype=torch.float, device="cuda")
        self.intrinsics_ = torch.zeros(self.N, 4, dtype=torch.float, device="cuda")

        self.points_ = torch.zeros(self.N * self.M, 3, dtype=torch.float, device="cuda")
        self.colors_ = torch.zeros(self.N, self.M, 3, dtype=torch.uint8, device="cuda")

        self.index_ = torch.zeros(self.N, self.M, dtype=torch.long, device="cuda")
        self.index_map_ = torch.zeros(self.N, dtype=torch.long, device="cuda")

        ### network attributes ###
        self.mem = 32

        if self.cfg.MIXED_PRECISION:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.half}
        else:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.float}
        
        self.imap_ = torch.zeros(self.mem, self.M, self.ctx_feat_dim, **kwargs)
        self.gmap_ = torch.zeros(self.mem, self.M, self.match_feat_dim, self.P, self.P, **kwargs)

        ht = int(ht // RES)
        wd = int(wd // RES)

        self.fmap1_ = torch.zeros(1, self.mem, self.match_feat_dim, int(ht // 1), int(wd // 1), **kwargs)   #MATCHING FEATURES 1/4
        self.fmap2_ = torch.zeros(1, self.mem, self.match_feat_dim, int(ht // 4), int(wd // 4), **kwargs)   #MATCHING FEATURES 1/16

        #STORE THE REFERENCES, so gets updated when fmap1_ and fmap2_ are updated
        self.pyramid = (self.fmap1_, self.fmap2_)

        self.net = torch.zeros(1, 0, self.ctx_feat_dim, **kwargs)
        self.ii = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.jj = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.kk = torch.as_tensor([], dtype=torch.long, device="cuda")
        
        # initialize poses to identity matrix
        self.poses_[:,6] = 1.0

        # store relative poses for removed frames
        self.delta = {}

        self.pose_folder = "ba_poses"

        #self.load_precomputed_poses = self.cfg.get("LOAD_PRECOMPUTED_POSES", True)
        if self.logname is not None and "write" in self.logname:
            import glob
            self.load_precomputed_poses = False
            if os.path.exists(self.pose_folder):
                # Clean up existing files
                files = glob.glob(os.path.join(self.pose_folder, "*"))
                for f in files:
                    #os.removedirs(f)
                    pass
            else:
                os.makedirs(self.pose_folder)
        elif self.logname is not None and "read" in self.logname:
            self.load_precomputed_poses = True
        else:
            self.load_precomputed_poses = False


        self.pose_file_path = "poses_log.txt"
        if os.path.exists(self.pose_file_path) and self.load_precomputed_poses == False:
            print(f"Pose file {self.pose_file_path} already exists, deleting it.")
            os.remove(self.pose_file_path)


        self.ba_inner_counter = 0
        self.pose_file_read_pointer = 0  # number of lines already read


        self.viz = viz

    def load_weights(self, network):
        # load network from checkpoint file
        if isinstance(network, str):
            print(f"Loading from {network}")
            checkpoint = torch.load(network)
            # TODO infer ctx_feat_dim=self.ctx_feat_dim, match_feat_dim=self.match_feat_dim, dim=self.dim
            self.network = eVONet(ctx_feat_dim=self.ctx_feat_dim, match_feat_dim=self.match_feat_dim, dim=self.dim, patch_selector=self.cfg.PATCH_SELECTOR, model=self.model, use_pyramid=self.use_pyramid,
                       use_softagg=self.use_softagg, use_tempconv=self.use_tempconv, **self.kwargs)
                
            if 'model_state_dict' in checkpoint:
                if self.cfg.PATCH_SELECTOR.lower() == "scorer":
                    #default case
                    self.network.load_state_dict(checkpoint['model_state_dict']) 
                else:
                    print("Loading model with patch selector:", self.cfg.PATCH_SELECTOR)
                    print("Loading weights from checkpoint with NOT STRICT mode, so to avoid errors due to the abscence of the patch selector in the checkpoint")
                    self.network.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                # legacy
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint.items():
                    if "update.lmbda" not in k:
                        new_state_dict[k.replace('module.', '')] = v
                self.network.load_state_dict(new_state_dict)

        else:
            self.use_pyramid = network.use_pyramid
            self.use_softagg = network.use_softagg
            self.use_tempconv = network.use_tempconv
            self.network = network

        # steal network attributes
        self.ctx_feat_dim = self.network.ctx_feat_dim
        self.match_feat_dim = self.network.match_feat_dim
        self.dim = self.network.dim
        self.RES = self.network.RES
        self.P = self.network.P

        self.network.cuda()
        self.network.eval()

        # if self.cfg.MIXED_PRECISION:
        #     self.network.half()


    @property
    def poses(self):
        return self.poses_.view(1, self.N, 7)

    @property
    def patches(self):
        return self.patches_.view(1, self.N*self.M, 3, 3, 3)
    
    @property
    def patches_gt(self):
        return self.patches_gt_.view(1, self.N*self.M, 3, 3, 3)

    @property
    def intrinsics(self):
        return self.intrinsics_.view(1, self.N, 4)

    @property
    def ix(self):
        return self.index_.view(-1)

    @property
    def imap(self):
        return self.imap_.view(1, self.mem * self.M, self.ctx_feat_dim)

    @property
    def gmap(self):
        return self.gmap_.view(1, self.mem * self.M, self.match_feat_dim, 3, 3)

    def get_pose(self, t):
        if t in self.traj:
            return SE3(self.traj[t])

        t0, dP = self.delta[t]
        return dP * self.get_pose(t0)

    def terminate(self):
        """ interpolate missing poses """
        # print("keyframes", self.n)
        self.traj = {}
        for i in range(self.n):
            self.traj[self.tstamps_[i].item()] = self.poses_[i]

        if self.is_initialized:
            poses = [self.get_pose(t) for t in range(self.counter)]
            poses = lietorch.stack(poses, dim=0)
            poses = poses.inv().data.cpu().numpy()
        else:
            print(f"Warning: Model is not initialized. Using Identity.") # eval still runs bug
            id = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            poses = np.array([id for t in range(self.counter)])
            poses[:, :3] = poses[:, :3] + np.random.randn(self.counter, 3) * 0.01 # small random trans

        tstamps = np.array(self.tlist, dtype=np.float64)


        #write into a txt file the index, one per line, of the keys of self.traj

        return poses, tstamps, self.network.update.max_nedges
    
    def corr(self, coords, indicies=None):
        """ local correlation volume """
        ii, jj = indicies if indicies is not None else (self.kk, self.jj)
        ii1 = ii % (self.M * self.mem)
        jj1 = jj % (self.mem)

        #dump_corr_inputs_npz("test_randomness/", self.gmap, self.pyramid, 0, coords, ii1, jj1)
        # compute the correlation operation with PATCHES FROM MATCHING FEATURES
        corr1 = altcorr.corr(self.gmap, self.pyramid[0], coords / 1, ii1, jj1, 3)
        
        if not self.use_pyramid:
            # if use_pyramid is False, only use corr1
            corr_volume = corr1.reshape(1, len(ii), -1)
        else:
            # otherwise, use both corr1 and corr2
            #dump_corr_inputs_npz("test_randomness/", self.gmap, self.pyramid, 1, coords, ii1, jj1)
            corr2 = altcorr.corr(self.gmap, self.pyramid[1], coords / 4, ii1, jj1, 3)
            # stack together the two correlation volumes
            corr_volume = torch.stack([corr1, corr2], -1).reshape(1, len(ii), -1)
        
        return corr_volume

    def reproject(self, indicies=None):
        """ reproject patch k from i -> j """
        (ii, jj, kk) = indicies if indicies is not None else (self.ii, self.jj, self.kk)
        coords = pops.transform(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk)
        return coords.permute(0, 1, 4, 2, 3).contiguous()

    def append_factors(self, ii, jj):
        self.jj = torch.cat([self.jj, jj])
        self.kk = torch.cat([self.kk, ii])
        self.ii = torch.cat([self.ii, self.ix[ii]]) 
        # TODO: self.ix.shape = self.M*self.N
        # self.ix is filled dynamically

        net = torch.zeros(1, len(ii), self.ctx_feat_dim, **self.kwargs)
        self.net = torch.cat([self.net, net], dim=1)

    def remove_factors(self, m):
        self.ii = self.ii[~m]
        self.jj = self.jj[~m]
        self.kk = self.kk[~m]
        self.net = self.net[:,~m]

    def motion_probe(self):
        """ kinda hacky way to ensure enough motion for initialization """
        kk = torch.arange(self.m-self.M, self.m, device="cuda")
        jj = self.n * torch.ones_like(kk)
        ii = self.ix[kk]

        net = torch.zeros(1, len(ii), self.ctx_feat_dim, **self.kwargs)
        coords = self.reproject(indicies=(ii, jj, kk))

        with autocast(enabled=self.cfg.MIXED_PRECISION):
            corr = self.corr(coords, indicies=(kk, jj))
            ctx = self.imap[:,kk % (self.M * self.mem)]
            net, (delta, weight, _) = \
                self.network.update(net, ctx, corr, None, ii, jj, kk)

        return torch.quantile(delta.norm(dim=-1).float(), 0.5)

    def motionmag(self, i, j):
        k = (self.ii == i) & (self.jj == j)
        ii = self.ii[k]
        jj = self.jj[k]
        kk = self.kk[k]

        flow = pops.flow_mag(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk, beta=0.5)
        return flow.mean().item()

    def keyframe(self):
        # described in 3.3. Keyframing DPVO paper
        # "after each update, compute flow_mag <t-5, t-3> and remove <t-4> if less than 64px"
        i = self.n - self.cfg.KEYFRAME_INDEX - 1 # t-5, KF_INDEX = 4 per default
        j = self.n - self.cfg.KEYFRAME_INDEX + 1 # t-3
        m = self.motionmag(i, j) + self.motionmag(j, i) 
 
        if m / 2 < self.cfg.KEYFRAME_THRESH:
            k = self.n - self.cfg.KEYFRAME_INDEX # scalar
            t0 = self.tstamps_[k-1].item()
            t1 = self.tstamps_[k].item()

            dP = SE3(self.poses_[k]) * SE3(self.poses_[k-1]).inv()
            self.delta[t1] = (t0, dP) # store relative pose between <t-5, t-4>

            to_remove = (self.ii == k) | (self.jj == k)
            self.remove_factors(to_remove)

            self.kk[self.ii > k] -= self.M
            self.ii[self.ii > k] -= 1
            self.jj[self.jj > k] -= 1

            for i in range(k, self.n-1):
                self.tstamps_[i] = self.tstamps_[i+1]
                self.colors_[i] = self.colors_[i+1]
                self.poses_[i] = self.poses_[i+1]
                self.patches_[i] = self.patches_[i+1]
                self.patches_gt_[i] = self.patches_gt_[i+1]
                self.intrinsics_[i] = self.intrinsics_[i+1]

                self.imap_[i%self.mem] = self.imap_[(i+1) % self.mem]
                self.gmap_[i%self.mem] = self.gmap_[(i+1) % self.mem]
                self.fmap1_[0,i%self.mem] = self.fmap1_[0,(i+1)%self.mem]
                self.fmap2_[0,i%self.mem] = self.fmap2_[0,(i+1)%self.mem]

            self.n -= 1 # remove frame
            self.m -= self.M

        to_remove = self.ix[self.kk] < self.n - self.cfg.REMOVAL_WINDOW
        self.remove_factors(to_remove)

    def update(self):
        coords = self.reproject()

        with autocast(enabled=True):
            
            corr = self.corr(coords)    #CORRELATION BLOCK 
            ctx = self.imap[:,self.kk % (self.M * self.mem)]
            with Timer("other", enabled=self.enable_timing):
                self.net, (delta, weight, _) = \
                    self.network.update(self.net, ctx, corr, None, self.ii, self.jj, self.kk)

        lmbda = torch.as_tensor([1e-4], device="cuda")
        weight = weight.float()
            
        target = coords[...,self.P//2,self.P//2] + delta.float()

        #with Timer("BA", enabled=self.enable_timing):
        #    t0 = self.n - self.cfg.OPTIMIZATION_WINDOW if self.is_initialized else 1
        #    t0 = max(t0, 1)
#
#
        #    try:
        #        fastba.BA(self.poses, self.patches, self.intrinsics, 
        #            target, weight, lmbda, self.ii, self.jj, self.kk, t0, self.n, 2)
        #    except:
        #        print("Warning BA failed...")
        #                #create random patches from index t0 to self.n, like emulating hte fastba.BA
#
        #    #find the length of the valid poses. skip the first one since it is the identity pose.
        #    #trasnform a torch tensor of poses into a list of poses
        #    self.poses_np = (self.poses.data.cpu().numpy().tolist())[0]
        #    self.last_pose_index = self.poses_np[1:].index([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
#
            
        with Timer("BA", enabled=self.enable_timing):
            t0 = self.n - self.cfg.OPTIMIZATION_WINDOW if self.is_initialized else 1
            t0 = max(t0, 1)

            pose_file_name = os.path.join(self.pose_folder, f"{self.ba_inner_counter}.txt")
            patch_file_name = os.path.join(self.pose_folder, f"{self.ba_inner_counter}_patches.npz")

            if self.load_precomputed_poses:
                # ------------ READ POSES ------------
                if not os.path.exists(pose_file_name):
                    raise RuntimeError(f"Pose file {pose_file_name} not found!")

                with open(pose_file_name, "r") as f:
                    lines = f.readlines()

                num_poses = len(lines)

                # Reset poses to identity
                identity_pose = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], device=self.poses.device, dtype=self.poses.dtype)
                self.poses[0, :, :] = identity_pose

                for idx, line in enumerate(lines):
                    vals = [float(v) for v in line.strip().split()]
                    self.poses[0, idx] = torch.tensor(vals, device=self.poses.device, dtype=self.poses.dtype)

                self.poses_np = self.poses[0].data.cpu().numpy().tolist()
                try:
                    self.last_pose_index = self.poses_np[1:].index([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]) + 1
                except ValueError:
                    self.last_pose_index = len(self.poses_np)

                # ------------ READ PATCHES ------------
                if not os.path.exists(patch_file_name):
                    raise RuntimeError(f"Patch file {patch_file_name} not found!")

                patch_data = np.load(patch_file_name)["patches"]
                patch_tensor = torch.from_numpy(patch_data).to(device=self.patches.device, dtype=self.patches.dtype)

                # Reset all patches before writing new
                self.patches.fill_(0.0)

                self.patches[0] = patch_tensor

                print(f"[BA] Read {num_poses} poses and patches from {pose_file_name}, {patch_file_name}")

                self.ba_inner_counter += 1

            else:
                # ------------ RUN BA ------------

                #print(f"[BA] Running BA with t0={t0}, n={self.n}, ii={self.ii.shape}, jj={self.jj.shape}, kk={self.kk.shape}")
                #dump_ba_inputs_npz(f"test/ba_poses_mvsec2_trial{self.trial}/{self.ba_inner_counter}.npz",
                #    self.poses, self.patches, self.intrinsics, 
                #    target, weight, lmbda, self.ii, self.jj, self.kk, t0, self.n)

                if self.cfg.BA_PRECISION == "double":
                    #uncomment for using double precision
                    self.poses_ = self.poses_.to(dtype=torch.float64, device=self.poses_.device)
                    self.patches_ = self.patches_.to(dtype=torch.float64, device=self.patches_.device)
                    self.intrinsics_ = self.intrinsics_.to(dtype=torch.float64, device=self.intrinsics_.device)

                    target = target.to(dtype=torch.float64, device=target.device)
                    weight = weight.to(dtype=torch.float64, device=weight.device)
                    lmbda = lmbda.to(dtype=torch.float64, device=lmbda.device)

                    fastba.BA_double(self.poses, self.patches, self.intrinsics, 
                            target, weight, lmbda, self.ii, self.jj, self.kk, t0, self.n, 2)
                    #now convert everything back to float32
                    self.poses_ = self.poses_.to(dtype=torch.float32, device=self.poses_.device)
                    self.patches_ = self.patches_.to(dtype=torch.float32, device=self.patches_.device)  
                    self.intrinsics_ = self.intrinsics_.to(dtype=torch.float32, device=self.intrinsics_.device)
                
                elif self.cfg.BA_PRECISION == "truncate":
                    decimal_places = 9
                    fastba.BA_trunc(self.poses, self.patches, self.intrinsics, 
                            target, weight, lmbda, self.ii, self.jj, self.kk, t0, self.n, 2, decimal_places=decimal_places)
                    
                elif self.cfg.BA_PRECISION == "single":
                    #single thread float32
                    fastba.BA_single_thread(self.poses, self.patches, self.intrinsics, 
                            target, weight, lmbda, self.ii, self.jj, self.kk, t0, self.n, 2)
                    
                elif self.cfg.BA_PRECISION == "cpu_fw":
                    #print(f"[BA] Running BA with cpu_fw precision")
                    #single thread float32
                    fastba.BA_red_cpu_fw(self.poses, self.patches, self.intrinsics, 
                            target, weight, lmbda, self.ii, self.jj, self.kk, t0, self.n, 2)
                elif self.cfg.BA_PRECISION == "cpu_fw_save":
                    #print(f"[BA] Running BA with cpu_fw_save precision")
                    #single thread float32
                    fastba.BA_red_cpu_fw_save(self.poses, self.patches, self.intrinsics, 
                            target, weight, lmbda, self.ii, self.jj, self.kk, t0, self.n, 2)
                    
                elif self.cfg.BA_PRECISION == "cpu_bw":
                    #print(f"[BA] Running BA with cpu_bw precision")
                    #single thread float32
                    fastba.BA_red_cpu_bw(self.poses, self.patches, self.intrinsics, 
                            target, weight, lmbda, self.ii, self.jj, self.kk, t0, self.n, 2)
                
                elif self.cfg.BA_PRECISION == "cpu_bw_save":
                    #print(f"[BA] Running BA with cpu_bw_save precision")
                    #single thread float32
                    fastba.BA_red_cpu_bw_save(self.poses, self.patches, self.intrinsics, 
                            target, weight, lmbda, self.ii, self.jj, self.kk, t0, self.n, 2)
                elif self.cfg.BA_PRECISION == "reduction":
                    #reduction method, deterministic float32

                    #print(f"[BA] Running BA with reduction method, t0={t0}, n={self.n}, ii={self.ii.shape}, jj={self.jj.shape}, kk={self.kk.shape}")
                    torch.cuda.synchronize()  # Wait for any prior GPU work
                    start_ba = time.time()

                    fastba.BA_red(self.poses, self.patches, self.intrinsics, 
                                target, weight, lmbda, self.ii, self.jj, self.kk, t0, self.n, 2)

                    torch.cuda.synchronize()  # Wait for this call to fully finish

                    elapsed_ba_ms = (time.time() - start_ba) * 1000  # Convert seconds → ms

                    #print(f"[BA] Total elapsed time: {elapsed_ba_ms:.2f} ms")
                
                elif self.cfg.BA_PRECISION == "red2":
                    #reduction method, deterministic float32
                    reduction_config = [1, 0, 1, 1, 1]
                    #reduction config = [B,C,E,v,u]
                    #print(f"[BA] Running BA with reduction method, t0={t0}, n={self.n}, ii={self.ii.shape}, jj={self.jj.shape}, kk={self.kk.shape}")
                    torch.cuda.synchronize()  # Wait for any prior GPU work
                    start_ba = time.time()

                    fastba.BA_red2(self.poses, self.patches, self.intrinsics, 
                                target, weight, lmbda, self.ii, self.jj, self.kk, t0, self.n, 2, reduction_config)

                    torch.cuda.synchronize()  # Wait for this call to fully finish

                    elapsed_ba_ms = (time.time() - start_ba) * 1000  # Convert seconds → ms

                    #print(f"[BA] Total elapsed time: {elapsed_ba_ms:.2f} ms")
                
                elif self.cfg.BA_PRECISION == "debug":
                    
                    torch.cuda.synchronize()  # Wait for any prior GPU work
                    start_ba = time.time()


                    #draw patch graph

                    if(self.iteration >= 35):
                        print(f"[BA] iteration {self.iteration} ")

                        #draw_patch_graph(self.ii, self.jj, self.kk, output_path="og.pdf")
                        #draw_patch_graph(self.ii-t0, self.jj-t0, self.kk, output_path="og_t0.pdf")
                        #patch ok
                        #poses ok
                        #intrinsics ok
                        #target ok
                        #weight ok
                        #ii are less than expected
                        trimmed_poses, trimmed_patches = save_ba_debug_inputs_as_c_pair(
                            self.poses, self.patches, self.intrinsics,
                            target, weight, lmbda, self.ii, self.jj, self.kk, t0, t1=self.n,
                            base_name="ba_in_fp32")
                        

                        #cut off the edges which would not be considered in the BA
                                # === Create edge mask: keep only edges NOT completely before t0 ===
                        mask = torch.logical_not(torch.logical_and(self.ii < t0, self.jj < t0))

                        # === Apply mask to edge-related arrays ===
                        ii_trimmed = self.ii[mask]
                        jj_trimmed = self.jj[mask]
                        kk_trimmed = self.kk[mask]
                        target_trimmed = target[:, mask, :]
                        weight_trimmed = weight[:, mask, :]
                        
                        #print(f"[BA] Running BA with debug precision, t0={t0}, n={self.n}, ii={self.ii.shape}, jj={self.jj.shape}, kk={self.kk.shape}")
                        #move every tensor to cpu
                        trimmed_poses = trimmed_poses.to(device="cpu")
                        trimmed_patches = trimmed_patches.to(device="cpu")
                        self.intrinsics_ = self.intrinsics_.to(device="cpu")

                        target_trimmed = target_trimmed.to(device="cpu")
                        weight_trimmed = weight_trimmed.to(device="cpu")
                        lmbda = lmbda.to(device="cpu")

                        ii_trimmed = ii_trimmed.to(device="cpu")
                        jj_trimmed = jj_trimmed.to(device="cpu")
                        kk_trimmed = kk_trimmed.to(device="cpu")

                        fastba.BA_cpu_debug(trimmed_poses,trimmed_patches, self.intrinsics,
                                target_trimmed, weight_trimmed, lmbda, ii_trimmed, jj_trimmed, kk_trimmed, t0, self.n, 1)
                        
                        #SAVE OUTPUT POSES AND PATCHES in a file
                        #extract only the center x y and depth from the patches
                        #self.patches shape = [1,4096*96,3,3,3]
                        #do self.patches.view(1, -1, 3, 3, 3)[:, :, 1, 1, 0:2] to get the center x y and depth
                        #trimmed_patches = trimmed_patches[0][:, :, 1, 1]
                        save_golden_outputs_as_c_pair(trimmed_poses, trimmed_patches, base_name="ba_out_fp32")

                        exit(0)  # exit after the first iteration, to avoid running the BA multiple times
                        
                    
                    #print(f"[BA] Running BA with debug precision, t0={t0}, n={self.n}, ii={self.ii.shape}, jj={self.jj.shape}, kk={self.kk.shape}")
                    fastba.BA_debug(self.poses, self.patches, self.intrinsics,
                            target, weight, lmbda, self.ii, self.jj, self.kk, t0, self.n, 2)

                    
                    torch.cuda.synchronize()  # Wait for this call to fully finish

                    elapsed_ba_ms = (time.time() - start_ba) * 1000  # Convert seconds → ms

                    #prepare to send the files:
                    #ba_golden_matrix_fp32_0.h
                    #ba_golden_matrix_fp32_1.h
                    #ba_in_fp32.h
                    #ba_out_fp32.h
                    #all via scp to 


                    #print(f"[BA] Total elapsed time: {elapsed_ba_ms:.2f} ms")
                elif self.cfg.BA_PRECISION == "kahan":
                    #kahan summation method, deterministic float32
                    torch.cuda.synchronize()  # Wait for any prior GPU work
                    start_ba = time.time()

                    fastba.BA_red_kahan(self.poses, self.patches, self.intrinsics, 
                            target, weight, lmbda, self.ii, self.jj, self.kk, t0, self.n, 2)

                    torch.cuda.synchronize()  # Wait for this call to fully finish

                    elapsed_ba_ms = (time.time() - start_ba) * 1000  # Convert seconds → ms
                
                elif self.cfg.BA_PRECISION == "kahan_db64":
                    #kahan summation method, deterministic float32
                    torch.cuda.synchronize()  # Wait for any prior GPU work
                    start_ba = time.time()

                    fastba.BA_kahan_db64(self.poses, self.patches, self.intrinsics, 
                            target, weight, lmbda, self.ii, self.jj, self.kk, t0, self.n, 2)


                    torch.cuda.synchronize()  # Wait for this call to fully finish

                    elapsed_ba_ms = (time.time() - start_ba) * 1000  # Convert seconds → ms


                elif self.cfg.BA_PRECISION == "kahan_bw":
                    #kahan summation method, deterministic float32
                    torch.cuda.synchronize()  # Wait for any prior GPU work
                    start_ba = time.time()

                    fastba.BA_red_kahan_bw(self.poses, self.patches, self.intrinsics, 
                            target, weight, lmbda, self.ii, self.jj, self.kk, t0, self.n, 2)

                    torch.cuda.synchronize()  # Wait for this call to fully finish

                    elapsed_ba_ms = (time.time() - start_ba) * 1000  # Convert seconds → ms
                    #print(f"[BA] Total elapsed time: {elapsed_ba_ms:.2f} ms")
                
                elif self.cfg.BA_PRECISION == "cpu":
                    #kahan summation method, deterministic float32
                    #print(f"[BA] Running BA with cpu precision")
                    torch.cuda.synchronize()  # Wait for any prior GPU work
                    start_ba = time.time()

                    self.poses_ = self.poses_.to(device="cpu")
                    self.patches_ = self.patches_.to(device="cpu")
                    self.intrinsics_ = self.intrinsics_.to(device="cpu")

                    target = target.to(device="cpu")
                    weight = weight.to(device="cpu")
                    lmbda = lmbda.to(device="cpu")

                    self.ii = self.ii.to(device="cpu")
                    self.jj = self.jj.to(device="cpu")
                    self.kk = self.kk.to(device="cpu")
                    fastba.BA_cpu(self.poses, self.patches, self.intrinsics, 
                            target, weight, lmbda, self.ii, self.jj, self.kk, t0, self.n, 2)

                    self.poses_ = self.poses_.to(device="cuda")
                    self.patches_ = self.patches_.to(device="cuda")
                    self.intrinsics_ = self.intrinsics_.to(device="cuda")

                    target = target.to(device="cuda")
                    weight = weight.to(device="cuda")
                    lmbda = lmbda.to(device="cuda")

                    self.ii = self.ii.to(device="cuda")
                    self.jj = self.jj.to(device="cuda")
                    self.kk = self.kk.to(device="cuda")
                    torch.cuda.synchronize()  # Wait for this call to fully finish

                    elapsed_ba_ms = (time.time() - start_ba) * 1000  # Convert seconds → ms
                    #print(f"[BA] Total elapsed time: {elapsed_ba_ms:.2f} ms")

                elif self.cfg.BA_PRECISION == "cpu_profile":
                    #kahan summation method, deterministic float32
                    #print(f"[BA] Running BA with cpu precision")
                    #print(f"[BA] Running BA with cpu profile precision")
                    torch.cuda.synchronize()  # Wait for any prior GPU work
                    start_ba = time.time()

                    self.poses_ = self.poses_.to(device="cpu")
                    self.patches_ = self.patches_.to(device="cpu")
                    self.intrinsics_ = self.intrinsics_.to(device="cpu")

                    target = target.to(device="cpu")
                    weight = weight.to(device="cpu")
                    lmbda = lmbda.to(device="cpu")

                    self.ii = self.ii.to(device="cpu")
                    self.jj = self.jj.to(device="cpu")
                    self.kk = self.kk.to(device="cpu")

                    #weight has shape (1,N,2), set to zero the ones which are not the top 70% 
                    #weight = weight * (weight > torch.quantile(weight, 0.25, dim=1, keepdim=True))
                    fastba.BA_cpu_profile(self.poses, self.patches, self.intrinsics, 
                            target, weight, lmbda, self.ii, self.jj, self.kk, t0, self.n, 2)

                    self.poses_ = self.poses_.to(device="cuda")
                    self.patches_ = self.patches_.to(device="cuda")
                    self.intrinsics_ = self.intrinsics_.to(device="cuda")

                    target = target.to(device="cuda")
                    weight = weight.to(device="cuda")
                    lmbda = lmbda.to(device="cuda")

                    self.ii = self.ii.to(device="cuda")
                    self.jj = self.jj.to(device="cuda")
                    self.kk = self.kk.to(device="cuda")
                    torch.cuda.synchronize()  # Wait for this call to fully finish

                    elapsed_ba_ms = (time.time() - start_ba) * 1000  # Convert seconds → ms
                    #print(f"[BA] Total elapsed time: {elapsed_ba_ms:.2f} ms")
                
                elif self.cfg.BA_PRECISION == "cpu_fp128":
                    #kahan summation method, deterministic quad precision
                    #print(f"[BA] Running BA with cpu precision")
                    torch.cuda.synchronize()  # Wait for any prior GPU work
                    start_ba = time.time()

                    self.poses_ = self.poses_.to(dtype=torch.float64, device='cpu')
                    self.patches_ = self.patches_.to(dtype=torch.float64, device='cpu')
                    self.intrinsics_ = self.intrinsics_.to(dtype=torch.float64, device='cpu')

                    target = target.to(dtype=torch.float64, device='cpu')
                    weight = weight.to(dtype=torch.float64, device='cpu')
                    lmbda = lmbda.to(dtype=torch.float64, device='cpu')

                    fastba.BA_cpu_fp128(self.poses, self.patches, self.intrinsics, 
                            target, weight, lmbda, self.ii, self.jj, self.kk, t0, self.n, 2)

                    self.poses_ = self.poses_.to(dtype=torch.float32, device='cuda')
                    self.patches_ = self.patches_.to(dtype=torch.float32, device='cuda')
                    self.intrinsics_ = self.intrinsics_.to(dtype=torch.float32, device='cuda')

                    target = target.to(device="cuda")
                    weight = weight.to(device="cuda")
                    lmbda = lmbda.to(device="cuda")

                    self.ii = self.ii.to(device="cuda")
                    self.jj = self.jj.to(device="cuda")
                    self.kk = self.kk.to(device="cuda")
                    torch.cuda.synchronize()  # Wait for this call to fully finish

                    elapsed_ba_ms = (time.time() - start_ba) * 1000  # Convert seconds → ms
                    #print(f"[BA] Total elapsed time: {elapsed_ba_ms:.2f} ms")
                

                elif self.cfg.BA_PRECISION == "truncate_double":
                    decimal_places = 13
                    #uncomment for using double precision
                    self.poses_ = self.poses_.to(dtype=torch.float64, device=self.poses_.device)
                    self.patches_ = self.patches_.to(dtype=torch.float64, device=self.patches_.device)
                    self.intrinsics_ = self.intrinsics_.to(dtype=torch.float64, device=self.intrinsics_.device)

                    target = target.to(dtype=torch.float64, device=target.device)
                    weight = weight.to(dtype=torch.float64, device=weight.device)
                    lmbda = lmbda.to(dtype=torch.float64, device=lmbda.device)

                    fastba.BA_trunc_double(self.poses, self.patches, self.intrinsics, 
                            target, weight, lmbda, self.ii, self.jj, self.kk, t0, self.n, 2, decimal_places=decimal_places)

                    #now convert everything back to float32
                    self.poses_ = self.poses_.to(dtype=torch.float32, device=self.poses_.device)
                    self.patches_ = self.patches_.to(dtype=torch.float32, device=self.patches_.device)  
                    self.intrinsics_ = self.intrinsics_.to(dtype=torch.float32, device=self.intrinsics_.device)
                else:
                    torch.cuda.synchronize()  # Wait for any prior GPU work
                    start_ba = time.time()
                    fastba.BA(self.poses, self.patches, self.intrinsics, 
                            target, weight, lmbda, self.ii, self.jj, self.kk, t0, self.n, 2)
                    torch.cuda.synchronize()  # Wait for this call to fully finish

                    elapsed_ba_ms = (time.time() - start_ba) * 1000  # Convert seconds → ms

                    #print(f"[BA] Total elapsed time: {elapsed_ba_ms:.2f} ms")
                

                self.poses_np = self.poses[0].data.cpu().numpy().tolist()
                #try:
                #    self.last_pose_index = self.poses_np[1:].index([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]) + 1
                #except ValueError:
                #    self.last_pose_index = len(self.poses_np)

                ## ------------ WRITE POSES ------------
                #with open(pose_file_name, "w") as f:
                #    for i in range(self.last_pose_index):
                #        pose_line = " ".join([str(v) for v in self.poses_np[i]])
                #        f.write(pose_line + "\n")
                #    f.flush()
                #    os.fsync(f.fileno())
#
                ## ------------ WRITE PATCHES ------------
                #patch_tensor = self.patches[0, :].detach().cpu().numpy()
                #np.savez_compressed(patch_file_name, patches=patch_tensor)
#
                #print(f"[BA] Wrote {self.last_pose_index} poses and patches to {pose_file_name}, {patch_file_name}, at ITER {self.iteration}, inner {self.ba_inner_counter}")
#
                self.ba_inner_counter += 1

            # Point cloud update
            points = pops.point_cloud(SE3(self.poses), self.patches[:, :self.m], self.intrinsics, self.ix[:self.m])
            points = (points[...,1,1,:3] / points[...,1,1,3:]).reshape(-1, 3)
            self.points_[:len(points)] = points[:]

            log_stats(self.poses, "ba_poses", self.logname, ignore_zeros=True)
            log_stats(points, "ba_points", self.logname, ignore_zeros=True)

            

    def __edges_forw(self):
        r=self.cfg.PATCH_LIFETIME  # default: 13
        t0 = self.M * max((self.n - r), 0)
        t1 = self.M * max((self.n - 1), 0)
        return flatmeshgrid(
            torch.arange(t0, t1, device="cuda"),
            torch.arange(self.n-1, self.n, device="cuda"), indexing='ij')

    def __edges_back(self):
        r=self.cfg.PATCH_LIFETIME  # default: 13
        t0 = self.M * max((self.n - 1), 0)
        t1 = self.M * max((self.n - 0), 0)
        return flatmeshgrid(
            torch.arange(t0, t1, device="cuda"),
            torch.arange(max(self.n-r, 0), self.n, device="cuda"), indexing='ij')

    def __call__(self, tstamp, image, intrinsics, scale=1.0):
        """ track new frame """

        if (self.n+1) >= self.N:
            raise Exception(f'The buffer size is too small. You can increase it using "--buffer {self.N*2}"')

        image = image[None,None]

        if self.n == 0:
            nonzero_ev = (image != 0.0)
            zero_ev = ~nonzero_ev
            num_nonzeros = nonzero_ev.sum().item()
            num_zeros = zero_ev.sum().item()
            # [DEBUG]
            # print("nonzero-zero-ratio", num_nonzeros, num_zeros, num_nonzeros / (num_zeros + num_nonzeros))
            if num_nonzeros / (num_zeros + num_nonzeros) < 2e-2: # TODO eval hyperparam (add to config.py)
                #print(f"skip voxel at {tstamp} due to lack of events!")
                return

        b, n, v, h, w = image.shape
        flatten_image = image.view(b,n,-1)
        
        #PRINT LOG flatten image
        log_stats(flatten_image, "flatten_image", self.logname, ignore_zeros=True)

        nonzero_ev = (flatten_image != 0.0)
        num_nonzeros = nonzero_ev.sum(dim=-1)
        if torch.all(num_nonzeros > 0):

            mean = torch.sum(flatten_image, dim=-1, dtype=torch.float32) / num_nonzeros  # force torch.float32 to prevent overflows when using 16-bit precision
            stddev = torch.sqrt(torch.sum(flatten_image ** 2, dim=-1, dtype=torch.float32) / num_nonzeros - mean ** 2)
            mask = nonzero_ev.type_as(flatten_image)
            flatten_image = mask * (flatten_image - mean[...,None]) / stddev[...,None]


        image = flatten_image.view(b,n,v,h,w)

        #PRINT LOG
        log_stats(image, "image", self.logname, ignore_zeros=True)
        

        if image.shape[-1] == 346:
            image = image[..., 1:-1] # hack for MVSEC, FPV,...
    

        # TODO patches with depth is available (val)
        with autocast(enabled=self.cfg.MIXED_PRECISION):
            fmap, gmap, imap, patches, coords= \
                self.network.patchify(image,
                    patches_per_image=self.cfg.PATCHES_PER_FRAME, 
                    return_color=True,
                    scorer_eval_mode=self.cfg.SCORER_EVAL_MODE,
                    scorer_eval_use_grid=self.cfg.SCORER_EVAL_USE_GRID)

        self.patches_gt_[self.n] = patches.clone()

        #PRINT LOG fmap, gmap, imap
        log_stats(fmap, "fmap", self.logname, ignore_zeros=True)
        log_stats(gmap, "gmap", self.logname, ignore_zeros=True)
        log_stats(imap, "imap", self.logname, ignore_zeros=True)
        log_stats(imap, "patches_gt", self.logname, ignore_zeros=True)

        if self.viz:
            show_voxel_coordinates(image, coords, index=self.iteration)

        ### update state attributes ###
        self.tlist.append(tstamp)
        self.tstamps_[self.n] = self.counter
        self.intrinsics_[self.n] = intrinsics / self.RES

        self.index_[self.n + 1] = self.n + 1
        self.index_map_[self.n + 1] = self.m + self.M

        if self.n > 1:
            if self.cfg.MOTION_MODEL == 'DAMPED_LINEAR':
                P1 = SE3(self.poses_[self.n-1])
                P2 = SE3(self.poses_[self.n-2])
                
                xi = self.cfg.MOTION_DAMPING * (P1 * P2.inv()).log()
                tvec_qvec = (SE3.exp(xi) * P1).data
                self.poses_[self.n] = tvec_qvec
            else:
                tvec_qvec = self.poses[self.n-1]
                self.poses_[self.n] = tvec_qvec

        #initialize patches with depth 1.0
        patches[:, :, 2] = 1.0
        

        if self.is_initialized:
            s = torch.median(self.patches_[self.n-3:self.n,:,2])
            patches[:,:,2] = s

        self.patches_[self.n] = patches


        #PRINT LOG self.poses, patches
        log_stats(self.poses, "poses", self.logname, ignore_zeros=True)
        log_stats(self.patches, "patches", self.logname, ignore_zeros=True)



        ### update network attributes ###
        self.imap_[self.n % self.mem] = imap.squeeze()
        self.gmap_[self.n % self.mem] = gmap.squeeze()
        
        log_stats(self.imap_, "imap_", self.logname, ignore_zeros=True)
        log_stats(self.gmap_, "gmap_", self.logname, ignore_zeros=True)

        
        #CREATING THE MATCHING FEATURES FOR THE PYRAMID
        self.fmap1_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 1, 1)
        self.fmap2_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 4, 4)

        #PRINT LOG self.fmap1_, self.fmap2_
        log_stats(self.fmap1_, "fmap1", self.logname, ignore_zeros=True)
        log_stats(self.fmap2_, "fmap2", self.logname, ignore_zeros=True)

        self.counter += 1

        if self.n > 0 and not self.is_initialized:
            thres = 2.0 if scale == 1.0 else scale ** 2 # TODO adapt thres for lite version
            if self.motion_probe() < thres: # TODO: replace by 8 pixels flow criterion (as described in 3.3 Initialization)
                self.delta[self.counter - 1] = (self.counter - 2, Id[0])
                return

        self.n += 1 # add one (key)frame
        self.m += self.M # add patches per (key)frames to patch number

        # relative pose
        self.append_factors(*self.__edges_forw())
        self.append_factors(*self.__edges_back())

        if self.n == 8 and not self.is_initialized:
            self.is_initialized = True            

            for itr in range(12):
                self.update()

        elif self.is_initialized:
            self.update()

            self.keyframe()


        # with open(f"test_randomness/full_runs/{self.logname}.txt", mode='a', newline='') as f:
        #     #write a ending line to the file like ====
        #     f.write(f"================\n")

        self.iteration += 1


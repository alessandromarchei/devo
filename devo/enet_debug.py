import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import torch_scatter
from torch_scatter import scatter_sum
from torchvision.ops import batched_nms

from . import fastba
from . import altcorr
from . import lietorch
from .lietorch import SE3

from .extractor import *
from .blocks import GradientClip, GatedResidual, SoftAgg
from .selector_debug import Scorer, SelectionMethod, PatchSelector, ev_density_selector

from .utils import *
from .ba import BA
from . import projective_ops as pops

autocast = torch.cuda.amp.autocast
import matplotlib.pyplot as plt

from utils.voxel_utils import std, rescale, voxel_augment
#from utils.viz_utils import visualize_voxel, visualize_N_voxels, visualize_scorer_map
from .utils import log_stats


DIM = 384 # default 384

class Update(nn.Module):
    def __init__(self, p, dim=DIM, use_pyramid=True, use_softagg=True, use_tempconv=True, **kwargs):
        
        super(Update, self).__init__()
        self.max_nedges = 0
        self.dim = dim
        self.use_pyramid = use_pyramid
        self.use_softagg = use_softagg
        self.use_tempconv = use_tempconv
        
        self.c1 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim))

        self.c2 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim))

        self.norm = nn.LayerNorm(dim, eps=1e-3)


        self.agg_kk = SoftAgg(dim)
        self.agg_ij = SoftAgg(dim)


        self.gru = nn.Sequential(
            nn.LayerNorm(dim, eps=1e-3),
            GatedResidual(dim),
            nn.LayerNorm(dim, eps=1e-3),
            GatedResidual(dim),
        )

        if(use_pyramid == False) :
            #SMALLER VERSION : ONLY THE HIGH RESOLUTION MATCHING FEATURES ARE USED (1/4 ON, 1/16 NOT ADDED) --> NOT PYRAMID.
            # only half the features are used
            self.corr = nn.Sequential(
                nn.Linear(49*p*p, dim),
                nn.ReLU(inplace=True),
                nn.Linear(dim, dim),
                nn.LayerNorm(dim, eps=1e-3),
                nn.ReLU(inplace=True),
                nn.Linear(dim, dim),
            )
        else:
            # DEFAULT CASE, with matching features downsampled at 1/4 and 1/16 and stacked toghether, with output flattened = 2*49*p*p
            self.corr = nn.Sequential(
                nn.Linear(2*49*p*p, dim),
                nn.ReLU(inplace=True),
                nn.Linear(dim, dim),
                nn.LayerNorm(dim, eps=1e-3),
                nn.ReLU(inplace=True),
                nn.Linear(dim, dim),
        )



        self.d = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(dim, 2),
            GradientClip())

        self.w = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(dim, 2),
            GradientClip(),
            nn.Sigmoid())
    
        self.logname = kwargs.get("logname", None)

    # def forward(self, net, inp, corr, flow, ii, jj, kk):
    #     """ update operator """

    #     if self.max_nedges < net.shape[1]:
    #         self.max_nedges = net.shape[1]
            
    #     with open('edges1.txt', 'a') as f:
    #         f.write(f"{net[0].mean().item()} \n")

    #     net = net + inp + self.corr(corr)
    #     net = self.norm(net) # (b,edges,384)

    #     ix, jx = fastba.neighbors(kk, jj)
    #     mask_ix = (ix >= 0).float().reshape(1, -1, 1)
    #     mask_jx = (jx >= 0).float().reshape(1, -1, 1)
        
    #     #apply the temporal convolution to the features
    #     net = net + self.c1(mask_ix * net[:,ix])
    #     net = net + self.c2(mask_jx * net[:,jx])

    #     #default configuration
    #     net = net + self.agg_kk(net, kk)
    #     net = net + self.agg_ij(net, ii*12345 + jj)


    #     net = self.gru(net)
    #     patch_flow=self.d(net)
    #     confidence_weigths = self.w(net)

    #     with open('edges1.txt', 'a') as f:
    #         f.write(f"{net[0].mean().item()} {net[0].std().item()} {net.shape[0]}\n")
    #         f.write(f"{patch_flow.mean().item()} {patch_flow.std().item()}\n")
    #         f.write(f"{confidence_weigths.mean().item()} {confidence_weigths.std().item()}\n")
    #         f.write(f"\n\n")
    #     return net, (patch_flow, confidence_weigths, None)
        
    def forward(self, net, inp, corr, flow, ii, jj, kk):
        """ update operator """

        if self.max_nedges < net.shape[1]:
            self.max_nedges = net.shape[1]

        #log_stats(self.poses, "ba_poses", self.logname)
        log_stats(net, "net_in", self.logname)
        log_stats(inp, "ctx_in", self.logname)
        log_stats(corr, "corr_in", self.logname)


        corr_feat = self.corr(corr)
        log_stats(corr_feat, "corr_out", self.logname)


        net = net + inp + corr_feat
        log_stats(net, "net+inp+corr", self.logname)

        net = self.norm(net)
        log_stats(net, "norm_out", self.logname)

        ix, jx = fastba.neighbors(kk, jj)
        mask_ix = (ix >= 0).float().reshape(1, -1, 1)
        mask_jx = (jx >= 0).float().reshape(1, -1, 1)

        t1 = self.c1(mask_ix * net[:,ix])
        net = net + t1
        log_stats(t1, "t1_out", self.logname)

        t2 = self.c2(mask_jx * net[:,jx])
        net = net + t2
        log_stats(t2, "t2_out", self.logname)

        a1 = self.agg_kk(net, kk)
        net = net + a1
        log_stats(a1, "agg_kk", self.logname)
        a2 = self.agg_ij(net, ii*12345 + jj)
        net = net + a2
        log_stats(a2, "agg_ij", self.logname)

        net = self.gru(net)
        log_stats(net, "gru_out", self.logname)
        patch_flow = self.d(net)
        log_stats(patch_flow, "patch_flow", self.logname)
        confidence_weights = self.w(net)
        log_stats(confidence_weights, "confidence_weights", self.logname)

        return net, (patch_flow, confidence_weights, None)




class Update_small(nn.Module):
    def __init__(self, p, dim=DIM, use_pyramid=True, use_softagg=True, use_tempconv=True):
        super(Update_small, self).__init__()
        self.dim = dim
        self.use_pyramid = use_pyramid
        self.use_softagg = use_softagg
        self.use_tempconv = use_tempconv
        
        # if self.use_tempconv:
        self.c1 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim))

        self.c2 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim))
        # else:
        #     #skip the temporal convolution in this case
        #     self.c1 = nn.Identity()
        #     self.c2 = nn.Identity()
            
        self.norm = nn.LayerNorm(dim, eps=1e-3)

        if self.use_softagg:
            self.agg_kk = SoftAgg(dim)
            self.agg_ij = SoftAgg(dim)
        else:
            self.agg_kk = nn.Identity()
            self.agg_ij = nn.Identity()

        if use_tempconv:
            self.gru = nn.Sequential(
                nn.LayerNorm(dim, eps=1e-3),
                GatedResidual(dim),
                nn.LayerNorm(dim, eps=1e-3),
                GatedResidual(dim),
            )
        else:
            self.gru = nn.Sequential(
                nn.LayerNorm(dim, eps=1e-3),
                nn.Linear(dim, dim),
                nn.ReLU(inplace=True),
                nn.Linear(dim, dim),
                nn.LayerNorm(dim, eps=1e-3),
                nn.Linear(dim, dim),
                nn.ReLU(inplace=True),
                nn.Linear(dim, dim),
            )
            

        if(use_pyramid == False) :
            #SMALLER VERSION : ONLY THE HIGH RESOLUTION MATCHING FEATURES ARE USED (1/4 ON, 1/16 NOT ADDED) --> NOT PYRAMID.
            # only half the features are used
            self.corr = nn.Sequential(
                nn.Linear(49*p*p, dim),
                nn.ReLU(inplace=True),
                nn.Linear(dim, dim),
                nn.LayerNorm(dim, eps=1e-3),
                nn.ReLU(inplace=True),
                nn.Linear(dim, dim),
            )
        else:
            # DEFAULT CASE, with matching features downsampled at 1/4 and 1/16 and stacked toghether, with output flattened = 2*49*p*p
            self.corr = nn.Sequential(
                nn.Linear(2*49*p*p, dim),
                nn.ReLU(inplace=True),
                nn.Linear(dim, dim),
                nn.LayerNorm(dim, eps=1e-3),
                nn.ReLU(inplace=True),
                nn.Linear(dim, dim),
        )



        self.d = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(dim, 2),
            GradientClip())

        self.w = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(dim, 2),
            GradientClip(),
            nn.Sigmoid())


    def forward(self, net, inp, corr, flow, ii, jj, kk):
        """ update operator """
        net = net + inp + self.corr(corr)
        net = self.norm(net) # (b,edges,384)


        #temporarily move the use_tempconv to use or not the GRU block, since we saw that removing the temporal convolution leads to really bad results
        # if self.use_tempconv:
        #     ix, jx = fastba.neighbors(kk, jj)
        #     mask_ix = (ix >= 0).float().reshape(1, -1, 1)
        #     mask_jx = (jx >= 0).float().reshape(1, -1, 1)
            
        #     #apply the temporal convolution to the features
        #     net = net + self.c1(mask_ix * net[:,ix])
        #     net = net + self.c2(mask_jx * net[:,jx])

        ix, jx = fastba.neighbors(kk, jj)
        mask_ix = (ix >= 0).float().reshape(1, -1, 1)
        mask_jx = (jx >= 0).float().reshape(1, -1, 1)
        
        #apply the temporal convolution to the features
        net = net + self.c1(mask_ix * net[:,ix])
        net = net + self.c2(mask_jx * net[:,jx])

        if self.use_softagg:
            #default configuration
            net = net + self.agg_kk(net, kk)
            net = net + self.agg_ij(net, ii*12345 + jj)



        net = self.gru(net)
        patch_flow=self.d(net)
        confidence_weigths = self.w(net)

        return net, (patch_flow, confidence_weigths, None)


class Patchifier(nn.Module):
    def __init__(self, event_bins = 5, patch_size=3, dim=32, match_feat_dim=128, ctx_feat_dim=384, patch_selector=SelectionMethod.SCORER, model="mksmall", **kwargs):
        """ Patchifier for extracting patches from event voxel grids """
        super(Patchifier, self).__init__()
        self.patch_size = patch_size
        self.match_feat_dim = match_feat_dim
        self.ctx_feat_dim = ctx_feat_dim
        self.patch_selector = patch_selector.lower()
        self.evs_bins = event_bins
        self.model = model
        self.iteration = 0

        self.kwargs = kwargs

        #CHOOSE ARCHITECTURE BASED ON THE MODEL
        if self.model == "mksmall":
            print("Using MKSmallEncoder")
            self.matching_feat_encoder = MKSmallEncoder(
                in_channels=self.evs_bins,
                output_dim=self.match_feat_dim,       #MATCHING FEATURE DIMENSION
                norm_fn="instance",
            )
            self.ctx_feat_encoder = MKSmallEncoder(
                in_channels=self.evs_bins,
                output_dim=self.ctx_feat_dim,       #CONTEXT FEATURE DIMENSION
                norm_fn="none",
            )

        elif self.model == "mkbig":
            print("Using MKBigEncoder")
            self.matching_feat_encoder = MKBigEncoder(
                in_channels=self.evs_bins,
                output_dim=self.match_feat_dim,       #MATCHING FEATURE DIMENSION
                norm_fn="instance",
            )
            self.ctx_feat_encoder = MKBigEncoder(
                in_channels=self.evs_bins,
                output_dim=self.ctx_feat_dim,       #CONTEXT FEATURE DIMENSION
                norm_fn="none",
            )

        elif self.model == "original":
            print("Using BasicEncoder4Evs")
            self.matching_feat_encoder = BasicEncoder4Evs(
                bins=self.evs_bins,
                output_dim=self.match_feat_dim,       #MATCHING FEATURE DIMENSION
                dim=dim,
                norm_fn="instance",
            )

            self.ctx_feat_encoder = BasicEncoder4Evs(
                bins=self.evs_bins,
                dim=dim,
                output_dim=self.ctx_feat_dim,       #CONTEXT FEATURE DIMENSION
                norm_fn="none",
            )

        elif self.model == "noskip":
            print("Using no skip")
            self.matching_feat_encoder = BasicEncoder4Evs_noskip(
                bins=self.evs_bins,
                output_dim=self.match_feat_dim,       #MATCHING FEATURE DIMENSION
                dim=dim,
                norm_fn="instance",
            )

            self.ctx_feat_encoder = BasicEncoder4Evs_noskip(
                bins=self.evs_bins,
                dim=dim,
                output_dim=self.ctx_feat_dim,       #CONTEXT FEATURE DIMENSION
                norm_fn="none",
            )
        
        elif self.model == "gradual":
            print("Using gradual encoder")
            self.matching_feat_encoder = GradualEncoder(
                in_channels=self.evs_bins,
                output_dim=self.match_feat_dim,       #MATCHING FEATURE DIMENSION
                dim=16,
                norm_fn="instance",
            )

            self.ctx_feat_encoder = GradualEncoder(
                in_channels=self.evs_bins,
                dim=16,
                output_dim=self.ctx_feat_dim,       #CONTEXT FEATURE DIMENSION
                norm_fn="none",
            )

        elif self.model == "gradual_halved":
            print("Using gradual_halved encoder")
            self.matching_feat_encoder = GradualEncoder_halved(
                in_channels=self.evs_bins,
                output_dim=self.match_feat_dim,       #MATCHING FEATURE DIMENSION
                dim=16,
                norm_fn="instance",
            )

            self.ctx_feat_encoder = GradualEncoder_halved(
                in_channels=self.evs_bins,
                dim=16,
                output_dim=self.ctx_feat_dim,       #CONTEXT FEATURE DIMENSION
                norm_fn="none",
            )
        
        elif self.model == "mobilenet":
            print("Using mobilenet encoder")
            self.matching_feat_encoder = MobileEncoder(
                in_channels=self.evs_bins,
                output_dim=self.match_feat_dim,       #MATCHING FEATURE DIMENSION
                dim=dim,
                norm_fn="instance",
            )

            self.ctx_feat_encoder = MobileEncoder(
                in_channels=self.evs_bins,
                dim=dim,
                output_dim=self.ctx_feat_dim,       #CONTEXT FEATURE DIMENSION
                norm_fn="none",
            )

        elif self.model == "original_bn":
            print("Using DEVO with batch normalization")
            self.matching_feat_encoder = BasicEncoder4Evs(
                bins=self.evs_bins,
                output_dim=self.match_feat_dim,       #MATCHING FEATURE DIMENSION
                dim=dim,
                norm_fn="batch",
            )

            self.ctx_feat_encoder = BasicEncoder4Evs(
                bins=self.evs_bins,
                dim=dim,
                output_dim=self.ctx_feat_dim,       #CONTEXT FEATURE DIMENSION
                norm_fn="none",
            )

        elif self.model == "DEVO":
            print("Using DEVO original, for evaluation only")
            self.fnet = BasicEncoder4Evs(output_dim=self.match_feat_dim, dim=dim, norm_fn='instance') # matching-feature extractor
            self.inet = BasicEncoder4Evs(output_dim=self.ctx_feat_dim, dim=dim, norm_fn='none') # context-feature extractor
        
        else:
            raise ValueError("Invalid model type")



        self.scorer = Scorer(5, **self.kwargs)

    def __event_gradient(self, events):
        events = events.sum(dim=2) # sum over bins
        dx = events[...,:-1,1:] - events[...,:-1,:-1]
        dy = events[...,1:,:-1] - events[...,:-1,:-1]
        g = torch.sqrt(dx**2 + dy**2)
        g = F.avg_pool2d(g, 4, 4)
        return g

    def forward(self, events, patches_per_image=96, disps=None, return_color=False, scorer_eval_mode="multi", scorer_eval_use_grid=True):
        """ extract patches from input events """
        if self.model == "DEVO":
            fmap = self.fnet(events) / 4.0 # (1, 15, 128, 120, 160)
            imap = self.inet(events) / 4.0
        else:
            fmap = self.matching_feat_encoder(events) / 4.0 # (1, 15, 128, 120, 160)
            imap = self.ctx_feat_encoder(events) / 4.0 # (1, 15, 384, 120, 160)

        b, n, c, h, w = fmap.shape # (1, 15, 128, 120, 160)
        P = self.patch_size

        scores = self.scorer(events) # (1, 15, 118, 158)
        scores = torch.sigmoid(scores)
        
        
        patch_selector_fn = PatchSelector(scorer_eval_mode, grid=scorer_eval_use_grid, **self.kwargs)
        
        if scorer_eval_mode != "density":
            x, y = patch_selector_fn(scores, patches_per_image)
        else:
            # here for the determienistic method based on density
            x, y = ev_density_selector(events=events, 
                           patches_per_image=patches_per_image, 
                           suppression_borders=11)


        x += 1
        y += 1
        coords = torch.stack([x, y], dim=-1).float() # in range (H//4, W//4)
        
        scores = altcorr.patchify(scores[0,:,None], coords, 0).view(n, patches_per_image) # extract weights of scorer map

        #save the coordinates in the range (H//4, W//4) with the dump function
        coords_path = f"test_randomness/mvsec1_coords/coords_{self.iteration}.npz"
        #save coordinates to a file txt
        #with open(f"mvsec4_coords/coords_{self.iteration}.txt", "w") as f:
        #    for i in range(coords.shape[0]):
        #        for j in range(coords.shape[1]):
        #            f.write(f"{coords[i,j,0].item()} {coords[i,j,1].item()}\n")
        
        #dump_extracted_coords_npz(path=coords_path,coords=coords)

        #load coordinates from the file
        coords = np.load(coords_path)["coords"]
        coords = torch.from_numpy(coords).to(device=fmap.device) # (b*n,patches_per_image, 2)

        #FMAP : MATCHING FEATURE MAP (1/4 RESOLUTION)
        #IMAP : CONTEXT FEATURE MAP (1/4 RESOLUTION)

        #extract the N PATCHES 1X1 FROM THE FMAP (INPUT : DIM,H/4,W/4) -> (OUTPUT : DIM,N_PATCHES,1,1)
        imap = altcorr.patchify(imap[0], coords, 0).view(b, -1, self.ctx_feat_dim, 1, 1) # [B, n_events*n_patches_per_image, ctx_feat_dim, 1, 1]
        
        #extract the N PATCHES 3X3 FROM THE FMAP (INPUT : DIM,H/4,W/4) -> (OUTPUT : DIM,N_PATCHES,3,3)
        gmap = altcorr.patchify(fmap[0], coords, P//2).view(b, -1, self.match_feat_dim, P, P) # [B, n_events*n_patches_per_image, match_feat_dim, 3, 3]

        if return_color:
            clr = altcorr.patchify(events[0].abs().sum(dim=1,keepdim=True), 4*(coords + 0.5), 0).clamp(min=0,max=255).view(b, -1, 1)

        if disps is None:
            disps = torch.ones(b, n, h, w, device="cuda")

        grid, _ = coords_grid_with_index(disps, device=fmap.device) # [B, n_events, 3, H//4, W//4]
        patches = altcorr.patchify(grid[0], coords, P//2).view(b, -1, 3, P, P) # [B, n_events*n_patches_per_image, 3, 3, 3]

        index = torch.arange(n, device="cuda").view(n, 1) # [n_events, 1]
        index = index.repeat(1, patches_per_image).reshape(-1) # [15, 80] => [15*80, 1] => [15*80]

        self.iteration += 1

        if self.training:
            if self.patch_selector == SelectionMethod.SCORER:
                return fmap, gmap, imap, patches, index, scores
        else:
            if return_color:
                return fmap, gmap, imap, patches, index, clr
            
        return fmap, gmap, imap, patches, index



class CorrBlock:
    """ Correlation block for computing correlation between matching features """
    def __init__(self, fmap, gmap, radius=3, dropout=0.2, levels=[1,4]):
        self.dropout = dropout
        self.radius = radius
        self.levels = levels

        self.gmap = gmap
        self.pyramid = pyramidify(fmap, lvls=levels)

    def __call__(self, ii, jj, coords):
        corrs = []
        for i in range(len(self.levels)):
            corrs += [ altcorr.corr(self.gmap, self.pyramid[i], coords / self.levels[i], ii, jj, self.radius, self.dropout) ]
        return torch.stack(corrs, -1).view(1, len(ii), -1)


class eVONet(nn.Module):
    def __init__(self, P=3, use_viewer=False, ctx_feat_dim=DIM, match_feat_dim=128, dim=32, patch_selector=SelectionMethod.SCORER, norm="std2", randaug=False, args=None, model="DEVO", use_pyramid=True, use_tempconv=True, use_softagg=True, **kwargs):
        """ eVONet for estimating SE3 between pairs of voxel grids """
        super(eVONet, self).__init__()
        print(f"Using {ctx_feat_dim} context feature dimension")
        print(f"Using {match_feat_dim} matching feature dimension")
        self.P = P
        self.ctx_feat_dim = ctx_feat_dim # dim of context extractor and hidden state (update operator)
        self.match_feat_dim = match_feat_dim # dim of matching extractor
        self.patch_selector = patch_selector
        self.model = model
        self.use_pyramid = use_pyramid
        self.use_tempconv = use_tempconv
        self.use_softagg = use_softagg

        if args is not None:
            self.use_pyramid = args.use_pyramid
            self.use_tempconv = args.use_tempconv
            self.use_softagg = args.use_softagg
            self.model = args.patchifier_model

        print(f"Using pyramid : {self.use_pyramid}")
        print(f"Using temporal conv : {self.use_tempconv}")
        print(f"Using softagg : {self.use_softagg}")
        print(f"Using patchifier model : {self.model}")

        self.kwargs = kwargs
        self.logname = kwargs.get("logname", None)

        self.patchify = Patchifier(patch_size=self.P, ctx_feat_dim=self.ctx_feat_dim, match_feat_dim=self.match_feat_dim, dim=dim, patch_selector=patch_selector, model=self.model, **self.kwargs)
        if self.use_softagg == False or self.use_tempconv == False:
            self.update = Update_small(self.P, self.ctx_feat_dim, use_pyramid=self.use_pyramid, use_softagg=self.use_softagg, use_tempconv=self.use_tempconv)
        else:
            self.update = Update(self.P, self.ctx_feat_dim, use_pyramid=self.use_pyramid, **self.kwargs)


        self.dim = dim # dim of the first layer in extractor
        self.RES = 4.0
        self.norm = norm
        self.randaug = randaug



    @autocast(enabled=False)
    def forward(self, images, poses, disps, intrinsics, M=1024, STEPS=12, P=1, structure_only=False, plot_patches=False, patches_per_image=80):
        """ Estimates SE3 between pair of voxel grids """
        
        # images (b,n_frames,c,h,w)
        # poses (b,n_frames)
        # disps (b,n_frames,h,w)
        
        b, n, v, h, w = images.shape

        # Normalize event voxel grids (rescale, std)
        if self.norm == 'none':
            pass
        elif self.norm == 'rescale' or self.norm == 'norm':
            # Normalize (rescaling) neg events into [-1,0) and pos events into (0,1] sequence-wise (by default)
            images = rescale(images)
        elif self.norm == 'standard' or self.norm == 'std':
            # Data standardization of events (voxel-wise)
            images = std(images, sequence=False)
        elif self.norm == 'standard2' or self.norm == 'std2':
            # Data standardization of events (sequence-wise by default)
            images = std(images)
        else:
            print(f"{self.norm} not implemented")
            raise NotImplementedError

        if self.training and self.randaug:
            if np.random.rand() < 0.33:
                if self.norm == 'rescale' or self.norm == 'norm':
                    images = voxel_augment(images, rescaled=True)
                elif 'std' in self.norm:
                    images = voxel_augment(images, rescaled=False)
                else:
                    print(f"{self.norm} not implemented")
                    raise NotImplementedError

        if plot_patches:
            plot_data = []

        intrinsics = intrinsics / self.RES
        if disps is not None:
            disps = disps[:, :, 1::4, 1::4].float()
            
        if self.patch_selector == SelectionMethod.SCORER:
            fmap, gmap, imap, patches, ix, scores = self.patchify(images, patches_per_image=patches_per_image, disps=disps)
        else:
            fmap, gmap, imap, patches, ix = self.patchify(images, patches_per_image=patches_per_image, disps=disps)
        # 1200 patches / 15 imgs = 80 patches per image
        # ix are image indices, i.e. simply (n_images, 80).flatten() = 15*80 = 1200 = n_patches
        # patches is (B, n_patches, 3, 3, 3), where (:, n_patches, 0, :, :) are x-coords, (:, n_patches, 1, :, :) are y-coords, (:, n_patches, 2, :, :) are depths 
        
        if self.use_pyramid == False:
            #create only 1 level (in case pyramid is not used)
            levels = [1]
            corr_fn = CorrBlock(fmap, gmap, levels=levels)
        else:
            #DEFAULT : use 2 levels (1 and 4)
            levels = [1, 4]
            corr_fn = CorrBlock(fmap, gmap, levels=levels)

        b, N, c, h, w = fmap.shape
        p = self.P

        patches_gt = patches.clone()
        Ps = poses

        d = patches[..., 2, p//2, p//2]
        patches = set_depth(patches, torch.rand_like(d))

        # first 8 images for initialization
        # kk are indixes for (first 8) patches/ixs of shape (1200*8/15)*8 = (640*8) = (5120)
        # jj are indices for (first 8) images/poses/intr in range (0, 7) of shape (640)*8 = (5120)
        kk, jj = flatmeshgrid(torch.where(ix < 8)[0], torch.arange(0,8, device="cuda"), indexing="ij")
        ii = ix[kk] # here, ii are image indices for initialization (5120)

        imap = imap.view(b, -1, self.ctx_feat_dim) # (b,n_patches,ctx_feat_dim) = (b,1200,384)
        net = torch.zeros(b, len(kk), self.ctx_feat_dim, device="cuda", dtype=torch.float) # init hidden state
        
        Gs = SE3.IdentityLike(poses)

        if structure_only:
            Gs.data[:] = poses.data[:]

        traj = []
        bounds = [-64, -64, w + 64, h + 64]
        
        while len(traj) < STEPS:
            Gs = Gs.detach()
            patches = patches.detach()

            n = ii.max() + 1
            if len(traj) >= 8 and n < images.shape[1]:
                if not structure_only: Gs.data[:,n] = Gs.data[:,n-1]
                kk1, jj1 = flatmeshgrid(torch.where(ix  < n)[0], torch.arange(n, n+1, device="cuda"))
                kk2, jj2 = flatmeshgrid(torch.where(ix == n)[0], torch.arange(0, n+1, device="cuda"))

                ii = torch.cat([ix[kk1], ix[kk2], ii])
                jj = torch.cat([jj1, jj2, jj])
                kk = torch.cat([kk1, kk2, kk])

                net1 = torch.zeros(b, len(kk1) + len(kk2), self.ctx_feat_dim, device="cuda")
                net = torch.cat([net1, net], dim=1)

                if np.random.rand() < 0.1:
                    k = (ii != (n - 4)) & (jj != (n - 4))
                    ii = ii[k]
                    jj = jj[k]
                    kk = kk[k]
                    net = net[:,k]

                patches[:,ix==n,2] = torch.median(patches[:,(ix == n-1) | (ix == n-2),2])
                n = ii.max() + 1

            coords = pops.transform(Gs, patches, intrinsics, ii, jj, kk)
            coords1 = coords.permute(0, 1, 4, 2, 3).contiguous() # (B,edges,P,P,2) -> (B,edges,2,P,P)


            #call the correlation function METHOD, specify if using the pyramid or not (baseline uses pyramid)
            corr = corr_fn(kk, jj, coords1)
            # corr (b,edges,p*p*7*7*2)
            # delta, weights (b,edges,2)
            # net (b,edges,384)
            net, (delta, weight, _) = self.update(net, imap[:,kk], corr, None, ii, jj, kk)

            lmbda = 1e-4
            target = coords[...,p//2,p//2,:] + delta # (B, edges, 2)

            ep = 10
            for itr in range(2):
                Gs, patches = BA(Gs, patches, intrinsics, target, weight, lmbda, ii, jj, kk, 
                    bounds, ep=ep, fixedp=1, structure_only=structure_only)

            kl = torch.as_tensor(0)
            dij = (ii - jj).abs()
            k = (dij > 0) & (dij <= 2) # k.sum() = (close_edges), i.e. > 0 and <= 2

            if self.patch_selector == SelectionMethod.SCORER:
                coords_full = pops.transform(Gs, patches, intrinsics, ii, jj, kk) # p_ij (B,close_edges,P,P,2)
                coords_gt_full, valid_full = pops.transform(Ps, patches_gt, intrinsics, ii, jj, kk, valid=True)
                coords = pops.transform(Gs, patches, intrinsics, ii[k], jj[k], kk[k]) # p_ij (B,close_edges,P,P,2)
                coords_gt, valid = pops.transform(Ps, patches_gt, intrinsics, ii[k], jj[k], kk[k], valid=True)
                
                k = (dij > 0) & (dij <= 16) # default 16
                traj.append((valid, coords, coords_gt, Gs[:,:n], Ps[:,:n], kl, scores, valid_full[0,k], coords_full.detach()[0,k], coords_gt_full.detach()[0,k], weight.detach()[0,k], kk[k], dij[k]))
            else:
                coords = pops.transform(Gs, patches, intrinsics, ii[k], jj[k], kk[k]) # p_ij (B,close_edges,P,P,2)
                coords_gt, valid = pops.transform(Ps, patches_gt, intrinsics, ii[k], jj[k], kk[k], valid=True)
                
                traj.append((valid, coords, coords_gt, Gs[:,:n], Ps[:,:n], kl))
            
            if plot_patches:
                coords_gt = pops.transform(Ps, patches_gt, intrinsics, ii, jj, kk)
                coords1_gt = coords_gt.permute(0, 1, 4, 2, 3).contiguous()
                coordsAll = pops.transform(Gs, patches, intrinsics, ii, jj, kk)
                coordsAll = coordsAll.permute(0, 1, 4, 2, 3).contiguous() 
                plot_data.append((ii, jj, patches, coordsAll, coords1_gt))

        if plot_patches:
            traj.append(plot_data)        
        return traj
    


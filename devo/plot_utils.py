from copy import deepcopy

import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import numpy as np
from evo.core import sync
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import plot
from evo.core.geometry import GeometryException
from pathlib import Path


def make_traj(args) -> PoseTrajectory3D:
    if isinstance(args, tuple):
        traj, tstamps = args
        return PoseTrajectory3D(positions_xyz=traj[:,:3], orientations_quat_wxyz=traj[:,3:], timestamps=tstamps)
    assert isinstance(args, PoseTrajectory3D), type(args)
    return deepcopy(args)

def best_plotmode(traj):
    _, i1, i2 = np.argsort(np.var(traj.positions_xyz, axis=0))
    plot_axes = "xyz"[i2] + "xyz"[i1]
    return getattr(plot.PlotMode, plot_axes)

def plot_trajectory(pred_traj, gt_traj=None, title="", filename="", align=True, correct_scale=True, max_diff_sec=0.01):
    pred_traj = make_traj(pred_traj)        #create a POSETRAJECTORY3D object

    if gt_traj is not None:
        gt_traj = make_traj(gt_traj)    #create a POSETRAJECTORY3D object
        gt_traj, pred_traj = sync.associate_trajectories(gt_traj, pred_traj, max_diff=max_diff_sec)

        if align:
            try:
                #ALIGN WITH THE UMEYAMA ALIGNMENT
                pred_traj.align(gt_traj, correct_scale=correct_scale)
            except GeometryException as e:
                print("Plotting error:", e)

    plot_collection = plot.PlotCollection("PlotCol")
    fig = plt.figure(figsize=(8, 8))
    plot_mode = best_plotmode(gt_traj if (gt_traj is not None) else pred_traj)
    ax = plot.prepare_axis(fig, plot_mode)
    ax.set_title(title)

    if gt_traj is not None:
        plot.traj(ax, plot_mode, gt_traj, '--', 'gray', "Ground Truth")

        # Highlight the starting point of the ground truth
        start_gt = gt_traj.positions_xyz[0]
        ax.plot(start_gt[2], start_gt[0], marker='o', color='red', markersize=5, label='GT Start')

    plot.traj(ax, plot_mode, pred_traj, '-', 'blue', "Predicted")
    plot_collection.add_figure("traj (error)", fig)
    plot_collection.export(filename, confirm_overwrite=False)
    plt.close(fig=fig)
    print(f"Saved {filename}")

# TODO refactor: merge in previous function plot_trajectory() with figure=False and save=False
def fig_trajectory(pred_traj, gt_traj=None, title="", filename="", align=True, correct_scale=True, save=False, return_figure=False, max_diff_sec=0.01):
    plt.switch_backend('Agg') # TODO instead install evo from source to use qt5agg backend
    
    pred_traj = make_traj(pred_traj)

    if gt_traj is not None:
        gt_traj = make_traj(gt_traj)
        gt_traj, pred_traj = sync.associate_trajectories(gt_traj, pred_traj, max_diff=max_diff_sec)

        if align:
            try:
                pred_traj.align(gt_traj, correct_scale=correct_scale)
            except GeometryException as e:
                print("Plotting error:", e)

    plot_collection = plot.PlotCollection("PlotCol")

    fig = plt.figure(figsize=(8, 8))
    plot_mode = best_plotmode(gt_traj if (gt_traj is not None) else pred_traj)
    ax = plot.prepare_axis(fig, plot_mode)
    ax.set_title(title)
    if gt_traj is not None:
        plot.traj(ax, plot_mode, gt_traj, '--', 'gray', "Ground Truth")
    plot.traj(ax, plot_mode, pred_traj, '-', 'blue', "Predicted")
    
    plot_collection.add_figure("traj (error)", fig)
    
    if save:
        plot_collection.export(filename, confirm_overwrite=False)
    if return_figure:
        return fig
    plt.close(fig=fig)
    return None

def save_trajectory_tum_format(traj, filename):
    traj = make_traj(traj)
    tostr = lambda a: ' '.join(map(str, a))
    with Path(filename).open('w') as f:
        for i in range(traj.num_poses):
            f.write(f"{traj.timestamps[i]} {tostr(traj.positions_xyz[i])} {tostr(traj.orientations_quat_wxyz[i][[1,2,3,0]])}\n")
    print(f"Saved {filename}")






def plot_trajectory_zx(traj, filename="trajectory.png", title="Trajectory"):
    """
    Nicely styled plot of trajectory (Z vs X) saved as an image.

    Parameters:
        traj (np.ndarray): Nx7 array of poses [x, y, z, qx, qy, qz, qw]
        filename (str): Path to save the plot image.
        title (str): Title for the plot.
    """
    x = traj[:, 0]  # X (right)
    z = traj[:, 2]  # Z (forward)

    # Use a clean modern style
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)

    ax.plot(z, x, marker='o', markersize=2, linewidth=1.5, color='#007acc', label="Trajectory")

    # Aesthetics
    ax.set_xlabel('Z (forward)', fontsize=12)
    ax.set_ylabel('X (right)', fontsize=12)
    ax.set_title(title, fontsize=13, weight='bold', pad=10)

    ax.legend(frameon=False, fontsize=10)
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(labelsize=10)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

    # Remove top/right spines for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

"""Shared trajectory-accuracy helpers for the SuperSLAM benchmark scripts.

Uses evo (https://github.com/MichaelGrupp/evo) for I/O, SE3 alignment, ATE and RPE,
plus the official KITTI per-segment translation/rotation error.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
from evo.core import metrics, sync
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.tools import file_interface


def load_kitti(path: str) -> PosePath3D:
    return file_interface.read_kitti_poses_file(path)


def load_tum(path: str) -> PoseTrajectory3D:
    return file_interface.read_tum_trajectory_file(path)


def load_euroc_gt(path: str) -> PoseTrajectory3D:
    return file_interface.read_euroc_csv_trajectory(path)


def load_tartan_gt(path: str) -> PosePath3D:
    """TartanAir/TartanGround pose file: per-frame 'tx ty tz qx qy qz qw' (no timestamp)."""
    a = np.loadtxt(path)
    xyz = a[:, 0:3]
    quat_wxyz = a[:, [6, 3, 4, 5]]  # evo expects w,x,y,z
    return PosePath3D(positions_xyz=xyz, orientations_quat_wxyz=quat_wxyz)


def ate(
    gt: PosePath3D, est: PosePath3D, align: bool = True, correct_scale: bool = False
) -> dict:
    """Absolute Trajectory Error (translation) after SE3/Sim3 Umeyama alignment."""
    est_a = copy.deepcopy(est)
    if align:
        est_a.align(gt, correct_scale=correct_scale)
    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((gt, est_a))
    return ape.get_all_statistics()


def rpe(
    gt: PosePath3D,
    est: PosePath3D,
    relation=metrics.PoseRelation.translation_part,
    delta: float = 1.0,
    unit=metrics.Unit.meters,
) -> dict:
    """Relative Pose Error over fixed deltas (drift)."""
    r = metrics.RPE(relation, delta, unit, all_pairs=False)
    r.process_data((gt, est))
    return r.get_all_statistics()


def sync_timestamped(
    gt: PoseTrajectory3D, est: PoseTrajectory3D, max_diff: float = 0.02
):
    """Associate two timestamped trajectories by nearest timestamp."""
    return sync.associate_trajectories(gt, est, max_diff)


# --- Official KITTI odometry metric: errors over 100..800 m subsequences ---
_KITTI_LENGTHS = [100, 200, 300, 400, 500, 600, 700, 800]
_KITTI_STEP = 10


def _cumulative_lengths(xyz: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def _last_frame_for_length(dist: np.ndarray, first: int, length: float):
    target = dist[first] + length
    for i in range(first, len(dist)):
        if dist[i] >= target:
            return i
    return None


def kitti_segments(gt_poses: np.ndarray, est_poses: np.ndarray) -> dict:
    """KITTI t_rel (fraction) and r_rel (deg/m) averaged over all segment lengths.

    gt_poses / est_poses are [N,4,4] homogeneous camera-to-world poses.
    """
    dist = _cumulative_lengths(gt_poses[:, :3, 3])
    t_errs, r_errs = [], []
    for first in range(0, len(gt_poses), _KITTI_STEP):
        for length in _KITTI_LENGTHS:
            last = _last_frame_for_length(dist, first, length)
            if last is None:
                continue
            gt_rel = np.linalg.inv(gt_poses[first]) @ gt_poses[last]
            est_rel = np.linalg.inv(est_poses[first]) @ est_poses[last]
            err = np.linalg.inv(est_rel) @ gt_rel
            t_errs.append(np.linalg.norm(err[:3, 3]) / length)
            cos = (np.trace(err[:3, :3]) - 1.0) * 0.5
            r_errs.append(np.arccos(max(-1.0, min(1.0, cos))) / length)
    if not t_errs:
        return {"t_rel_percent": float("nan"), "r_rel_deg_per_m": float("nan")}
    return {
        "t_rel_percent": float(np.mean(t_errs) * 100.0),
        "r_rel_deg_per_m": float(np.degrees(np.mean(r_errs))),
    }


def plot_xy(
    gt: PosePath3D,
    est: PosePath3D,
    out_png: str,
    title: str,
    align: bool = True,
    axes: tuple = (0, 1),
) -> None:
    # Plot the trajectory top-down on two world axes. Default to x and y; pass axes=(0, 2)
    # for the x-z ground plane (KITTI, where y is the gravity axis).
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    est_a = copy.deepcopy(est)
    if align:
        est_a.align(gt, correct_scale=False)
    g = gt.positions_xyz
    e = est_a.positions_xyz
    a, b = axes
    label = {0: "x", 1: "y", 2: "z"}
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(g[:, a], g[:, b], "k-", linewidth=1.5, label="ground truth")
    ax.plot(e[:, a], e[:, b], "r-", linewidth=1.0, label="SuperSLAM")
    ax.scatter(g[0, a], g[0, b], c="g", marker="o", label="start")
    ax.set_aspect("equal", "datalim")
    ax.set_xlabel(f"{label[a]} [m]")
    ax.set_ylabel(f"{label[b]} [m]")
    ax.legend()
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


def poses_to_matrices(traj: PosePath3D) -> np.ndarray:
    return np.array(traj.poses_se3)

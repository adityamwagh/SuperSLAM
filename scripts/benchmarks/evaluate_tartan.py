#!/usr/bin/env python3
"""Evaluate SuperSLAM trajectories against TartanAir / TartanGround ground truth.

  python scripts/benchmarks/evaluate_tartan.py --traj-dir results/tartanair \
      --data-dir ~/datasets/tartanair/ArchVizTinyHouseDay/Data_easy

Estimated trajectories are KITTI-format files named <traj>.txt (e.g. P000.txt); ground
truth is <data-dir>/<traj>/pose_lcam_front.txt. Associated by frame index, SE3-aligned
(Umeyama absorbs the NED<->camera frame difference). Reports ATE, RPE and the KITTI
segment metric, with an aligned XY plot per trajectory.
"""

import argparse
from pathlib import Path

import _eval_common as ev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", required=True)
    ap.add_argument(
        "--data-dir",
        required=True,
        help="env Data dir (has <traj>/pose_lcam_front.txt)",
    )
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--name", default="TartanAir", help="dataset label for plot titles")
    args = ap.parse_args()

    traj_dir = Path(args.traj_dir)
    out_dir = Path(args.out_dir or args.traj_dir)
    rows = []
    for est_file in sorted(traj_dir.glob("*.txt")):
        traj = est_file.stem
        gt_file = Path(args.data_dir) / traj / "pose_lcam_front.txt"
        if not gt_file.exists():
            print(f"[skip] {traj}: no ground truth ({gt_file})")
            continue
        gt = ev.load_tartan_gt(str(gt_file))
        est = ev.load_kitti(str(est_file))
        a = ev.ate(gt, est, align=True)
        r = ev.rpe(gt, est)
        seg = ev.kitti_segments(ev.poses_to_matrices(gt), ev.poses_to_matrices(est))
        ev.plot_xy(gt, est, str(out_dir / f"{traj}_traj.png"), f"{args.name} {traj}")
        rows.append((traj, a["rmse"], a["mean"], r["rmse"], seg["t_rel_percent"]))
        print(
            f"{traj}: ATE_rmse={a['rmse']:.3f}m  RPE_rmse={r['rmse']:.3f}m  "
            f"t_rel={seg['t_rel_percent']:.2f}%"
        )

    out = out_dir / "metrics_tartan.md"
    with open(out, "w") as f:
        f.write("| traj | ATE RMSE (m) | ATE mean (m) | RPE RMSE (m) | t_rel (%) |\n")
        f.write("|---|---|---|---|---|\n")
        for t, ar, am, rr, tp in rows:
            f.write(f"| {t} | {ar:.3f} | {am:.3f} | {rr:.3f} | {tp:.2f} |\n")
    print(f"\nWrote {out} and {len(rows)} plots to {out_dir}")


if __name__ == "__main__":
    main()

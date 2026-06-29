#!/usr/bin/env python3
"""Evaluate SuperSLAM trajectories against KITTI odometry ground truth.

  python scripts/benchmarks/evaluate_kitti.py --traj-dir results/kitti \
      --data-dir ~/datasets/kitti/dataset --out-dir results/kitti

Estimated trajectories are KITTI-format files named <seq>.txt (00..10); ground truth is
<data-dir>/poses/<seq>.txt. Reports ATE, RPE, and the official KITTI segment metric, and
writes an aligned x-z ground-plane plot per sequence.
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
        help="KITTI dataset dir (has poses/ and sequences/)",
    )
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    traj_dir = Path(args.traj_dir)
    out_dir = Path(args.out_dir or args.traj_dir)
    rows = []
    for est_file in sorted(traj_dir.glob("*.txt")):
        seq = est_file.stem
        gt_file = Path(args.data_dir) / "poses" / f"{seq}.txt"
        if not gt_file.exists():
            print(f"[skip] {seq}: no ground truth ({gt_file})")
            continue
        gt = ev.load_kitti(str(gt_file))
        est = ev.load_kitti(str(est_file))
        a = ev.ate(gt, est, align=True)
        r = ev.rpe(gt, est)
        seg = ev.kitti_segments(ev.poses_to_matrices(gt), ev.poses_to_matrices(est))
        ev.plot_xy(
            gt, est, str(out_dir / f"{seq}_traj.png"), f"KITTI {seq}", axes=(0, 2)
        )
        rows.append(
            (
                seq,
                a["rmse"],
                a["mean"],
                r["rmse"],
                seg["t_rel_percent"],
                seg["r_rel_deg_per_m"],
            )
        )
        print(
            f"{seq}: ATE_rmse={a['rmse']:.3f}m  t_rel={seg['t_rel_percent']:.2f}%  "
            f"r_rel={seg['r_rel_deg_per_m']:.4f}deg/m"
        )

    out = out_dir / "metrics_kitti.md"
    with open(out, "w") as f:
        f.write(
            "| seq | ATE RMSE (m) | ATE mean (m) | RPE RMSE (m) | t_rel (%) | r_rel (deg/m) |\n"
        )
        f.write("|---|---|---|---|---|---|\n")
        for s, ar, am, rr, tp, rd in rows:
            f.write(f"| {s} | {ar:.3f} | {am:.3f} | {rr:.3f} | {tp:.2f} | {rd:.4f} |\n")
    print(f"\nWrote {out} and {len(rows)} plots to {out_dir}")


if __name__ == "__main__":
    main()

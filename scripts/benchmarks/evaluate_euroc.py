#!/usr/bin/env python3
"""Evaluate SuperSLAM trajectories against EuRoC MAV ground truth.

  python scripts/benchmarks/evaluate_euroc.py --traj-dir results/euroc \
      --data-dir ~/datasets/euroc --out-dir results/euroc

Estimated trajectories are TUM-format files named <sequence>.txt; ground truth is
<data-dir>/<sequence>/mav0/state_groundtruth_estimate0/data.csv. Timestamped, so the two
trajectories are associated by timestamp before SE3 alignment. Reports ATE + RPE and an
aligned XY plot per sequence.
"""

import argparse
from pathlib import Path

import _eval_common as ev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", required=True)
    ap.add_argument(
        "--data-dir", required=True, help="EuRoC dataset dir (has <seq>/mav0/...)"
    )
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    traj_dir = Path(args.traj_dir)
    out_dir = Path(args.out_dir or args.traj_dir)
    rows = []
    for est_file in sorted(traj_dir.glob("*.txt")):
        seq = est_file.stem
        gt_file = (
            Path(args.data_dir)
            / seq
            / "mav0"
            / "state_groundtruth_estimate0"
            / "data.csv"
        )
        if not gt_file.exists():
            print(f"[skip] {seq}: no ground truth ({gt_file})")
            continue
        gt = ev.load_euroc_gt(str(gt_file))
        est = ev.load_tum(str(est_file))
        gt_s, est_s = ev.sync_timestamped(gt, est)
        a = ev.ate(gt_s, est_s, align=True)
        r = ev.rpe(gt_s, est_s)
        ev.plot_xy(gt_s, est_s, str(out_dir / f"{seq}_traj.png"), f"EuRoC {seq}")
        rows.append((seq, a["rmse"], a["mean"], a["max"], r["rmse"]))
        print(
            f"{seq}: ATE_rmse={a['rmse']:.3f}m  ATE_max={a['max']:.3f}m  RPE_rmse={r['rmse']:.3f}m"
        )

    out = out_dir / "metrics_euroc.md"
    with open(out, "w") as f:
        f.write("| seq | ATE RMSE (m) | ATE mean (m) | ATE max (m) | RPE RMSE (m) |\n")
        f.write("|---|---|---|---|---|\n")
        for s, ar, am, amx, rr in rows:
            f.write(f"| {s} | {ar:.3f} | {am:.3f} | {amx:.3f} | {rr:.3f} |\n")
    print(f"\nWrote {out} and {len(rows)} plots to {out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Download TartanGround stereo sequences into ~/datasets/tartanground.

Wraps the official `tartanair` toolbox (pip install tartanair). Fetches only the front
RGB stereo pair and ground-truth poses for the SLAM example.

  uv run --with tartanair python scripts/datasets/download_tartanground.py
  uv run --with tartanair python scripts/datasets/download_tartanground.py --env OldTownSummer --traj P0000
  uv run --with tartanair python scripts/datasets/download_tartanground.py --list
"""

import argparse
import os


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download TartanGround stereo into ~/datasets/tartanground"
    )
    ap.add_argument("--out", default="~/datasets/tartanground")
    ap.add_argument("--env", nargs="+", default=["OldTownSummer"])
    ap.add_argument(
        "--traj", nargs="+", default=[], help="trajectory ids (default: all)"
    )
    ap.add_argument(
        "--list", action="store_true", help="list available environments and exit"
    )
    args = ap.parse_args()

    import tartanair as ta

    root = os.path.expanduser(args.out)
    os.makedirs(root, exist_ok=True)
    ta.init(root)
    if args.list:
        print(ta.list_envs())
        return

    ta.download_ground(
        env=args.env,
        traj=args.traj,
        modality=["image", "pose"],
        camera_name=["lcam_front", "rcam_front"],
        unzip=True,
        num_workers=4,
    )
    print(f"Done. Sequences under {root}")


if __name__ == "__main__":
    main()

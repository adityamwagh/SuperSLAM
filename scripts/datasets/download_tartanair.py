#!/usr/bin/env python3
"""Download TartanAir-V2 stereo sequences into ~/datasets/tartanair.

Wraps the official `tartanair` toolbox (pip install tartanair). Fetches only the front
RGB stereo pair (lcam_front/rcam_front) and the ground-truth poses for the SLAM example.

  uv run --with tartanair python scripts/datasets/download_tartanair.py            # ArchVizTinyHouseDay easy
  uv run --with tartanair python scripts/datasets/download_tartanair.py --env Office Downtown --difficulty hard
  uv run --with tartanair python scripts/datasets/download_tartanair.py --list
"""

import argparse
import os


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download TartanAir-V2 stereo into ~/datasets/tartanair"
    )
    ap.add_argument("--out", default="~/datasets/tartanair")
    ap.add_argument("--env", nargs="+", default=["ArchVizTinyHouseDay"])
    ap.add_argument("--difficulty", default="easy", choices=["easy", "hard"])
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

    ta.download(
        env=args.env,
        difficulty=[args.difficulty],
        modality=["image", "pose"],
        camera_name=["lcam_front", "rcam_front"],
        unzip=True,
        num_workers=4,
    )
    print(f"Done. Sequences under {root}")


if __name__ == "__main__":
    main()

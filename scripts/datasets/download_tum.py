#!/usr/bin/env python3
"""Download TUM RGB-D sequences into ~/datasets/tum.

Each sequence is a .tgz (rgb + depth + groundtruth.txt). Useful for monocular/RGB-D
evaluation and as an additional loop-closure testbed (fr1/fr2 desk revisits).

  python scripts/datasets/download_tum.py                       # fr1_xyz (default)
  python scripts/datasets/download_tum.py --seq fr1_desk fr2_desk
  python scripts/datasets/download_tum.py --list
"""

import argparse

from _common import dataset_dir, fetch

BASE = "https://cvg.cit.tum.de/rgbd/dataset"
# short name -> (freiburgN, archive stem)
SEQUENCES = {
    "fr1_xyz": ("freiburg1", "rgbd_dataset_freiburg1_xyz"),
    "fr1_desk": ("freiburg1", "rgbd_dataset_freiburg1_desk"),
    "fr1_room": ("freiburg1", "rgbd_dataset_freiburg1_room"),
    "fr2_xyz": ("freiburg2", "rgbd_dataset_freiburg2_xyz"),
    "fr2_desk": ("freiburg2", "rgbd_dataset_freiburg2_desk"),
    "fr3_long_office": ("freiburg3", "rgbd_dataset_freiburg3_long_office_household"),
    "fr3_walking_xyz": ("freiburg3", "rgbd_dataset_freiburg3_walking_xyz"),
    "fr3_walking_halfsphere": ("freiburg3", "rgbd_dataset_freiburg3_walking_halfsphere"),
    "fr3_walking_rpy": ("freiburg3", "rgbd_dataset_freiburg3_walking_rpy"),
    "fr3_walking_static": ("freiburg3", "rgbd_dataset_freiburg3_walking_static"),
    "fr3_sitting_xyz": ("freiburg3", "rgbd_dataset_freiburg3_sitting_xyz"),
    "fr3_sitting_halfsphere": ("freiburg3", "rgbd_dataset_freiburg3_sitting_halfsphere"),
    "fr3_sitting_rpy": ("freiburg3", "rgbd_dataset_freiburg3_sitting_rpy"),
    "fr3_sitting_static": ("freiburg3", "rgbd_dataset_freiburg3_sitting_static"),
}


def url_for(key: str) -> str:
    grp, stem = SEQUENCES[key]
    return f"{BASE}/{grp}/{stem}.tgz"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download TUM RGB-D sequences into ~/datasets/tum"
    )
    ap.add_argument("--out", help="target dir (default ~/datasets/tum)")
    ap.add_argument(
        "--seq", nargs="+", default=["fr1_xyz"], help="sequence keys (see --list)"
    )
    ap.add_argument("--all", action="store_true", help="download all listed sequences")
    ap.add_argument(
        "--list", action="store_true", help="list available sequences and exit"
    )
    args = ap.parse_args()

    if args.list:
        for k, (grp, stem) in SEQUENCES.items():
            print(f"  {k:16s} {grp}/{stem}")
        return

    keys = list(SEQUENCES) if args.all else args.seq
    unknown = [k for k in keys if k not in SEQUENCES]
    if unknown:
        raise SystemExit(f"unknown sequence(s): {unknown} (try --list)")

    dest = dataset_dir("tum", args.out)
    print(f"TUM RGB-D -> {dest}")
    for k in keys:
        print(f"[{k}]")
        fetch(url_for(k), dest, unpack=True)

    print(f"Done. Sequences under {dest}")


if __name__ == "__main__":
    main()

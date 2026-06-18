#!/usr/bin/env python3
"""Download the KITTI odometry dataset into ~/datasets/kitti.

The odometry benchmark ships as three archives (grayscale images, calibration, ground
-truth poses); the color images are a separate, larger archive. SuperSLAM's stereo
example uses the grayscale stereo pair (image_0 / image_1) + calib + poses.

  python scripts/datasets/download_kitti.py                 # gray + calib + poses
  python scripts/datasets/download_kitti.py --color         # also fetch color images
  python scripts/datasets/download_kitti.py --out /data/kitti
"""

import argparse

from _common import dataset_dir, fetch

# Public KITTI odometry mirror (avg-kitti S3). Sizes are approximate.
BASE = "https://s3.eu-central-1.amazonaws.com/avg-kitti"
ARCHIVES = {
    "calib": f"{BASE}/data_odometry_calib.zip",  # ~1 MB
    "poses": f"{BASE}/data_odometry_poses.zip",  # ~4 MB
    "gray": f"{BASE}/data_odometry_gray.zip",  # ~22 GB
    "color": f"{BASE}/data_odometry_color.zip",  # ~65 GB
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download KITTI odometry into ~/datasets/kitti"
    )
    ap.add_argument("--out", help="target dir (default ~/datasets/kitti)")
    ap.add_argument(
        "--color", action="store_true", help="also download color images (~65 GB)"
    )
    ap.add_argument(
        "--no-images",
        action="store_true",
        help="only calib + poses (no image archives)",
    )
    args = ap.parse_args()

    dest = dataset_dir("kitti", args.out)
    print(f"KITTI odometry -> {dest}")

    keys = ["calib", "poses"]
    if not args.no_images:
        keys.append("gray")
        if args.color:
            keys.append("color")

    for k in keys:
        fetch(ARCHIVES[k], dest, unpack=True)

    print(f"Done. KITTI sequences under {dest}/dataset/sequences")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Download EuRoC MAV stereo sequences into ~/datasets/euroc.

The dataset is hosted on the ETH Research Collection as per-area bundle zips, each a
zip-of-zips holding the nested ASL .zip (and a rosbag) per sequence. This downloads a
bundle and extracts only the nested ASL sequences into ~/datasets/euroc/<sequence>/mav0.

  python scripts/datasets/download_euroc.py                      # machine_hall
  python scripts/datasets/download_euroc.py --area vicon_room1
  python scripts/datasets/download_euroc.py --area all
  python scripts/datasets/download_euroc.py --list
"""

import argparse
import io
import zipfile

from _common import dataset_dir, download

CONTENT = (
    "https://www.research-collection.ethz.ch/server/api/core/bitstreams/{id}/content"
)
BUNDLES = {
    "machine_hall": "7b2419c1-62b5-4714-b7f8-485e5fe3e5fe",
    "vicon_room1": "02ecda9a-298f-498b-970c-b7c44334d880",
    "vicon_room2": "ea12bc01-3677-4b4c-853d-87c7870b8c44",
    "calibration": "5732e864-10f1-49e7-befb-669ee29ff770",
}


def extract_sequences(bundle_path, dest) -> None:
    with zipfile.ZipFile(bundle_path) as z:
        nested = [n for n in z.namelist() if n.endswith(".zip") and "/" in n]
        for name in nested:
            seq = name.rsplit("/", 1)[-1][:-4]
            target = dest / seq
            if (target / "mav0").exists():
                print(f"  [skip] {seq} already extracted")
                continue
            print(f"  [extract] {seq} -> {target}")
            with z.open(name) as f:
                data = io.BytesIO(f.read())
            with zipfile.ZipFile(data) as nz:
                nz.extractall(target)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download EuRoC MAV sequences into ~/datasets/euroc"
    )
    ap.add_argument("--out", help="target dir (default ~/datasets/euroc)")
    ap.add_argument(
        "--area", nargs="+", default=["machine_hall"], help="bundle(s) to fetch"
    )
    ap.add_argument("--all", action="store_true", help="download every area bundle")
    ap.add_argument(
        "--list", action="store_true", help="list available area bundles and exit"
    )
    args = ap.parse_args()

    if args.list:
        for k in BUNDLES:
            print(f"  {k}")
        return

    areas = list(BUNDLES) if args.all else args.area
    unknown = [a for a in areas if a not in BUNDLES]
    if unknown:
        raise SystemExit(f"unknown area(s): {unknown} (try --list)")

    dest = dataset_dir("euroc", args.out)
    print(f"EuRoC -> {dest}")
    for area in areas:
        print(f"[{area}]")
        bundle = download(CONTENT.format(id=BUNDLES[area]), dest / f"{area}.zip")
        if area != "calibration":
            extract_sequences(bundle, dest)

    print(f"Done. Sequences under {dest}")


if __name__ == "__main__":
    main()

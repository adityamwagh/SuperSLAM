"""Shared helpers for the per-model artifact downloaders.

Resolve the GitHub repo from the git remote and download prebuilt ONNX from a GitHub
Release into weights/. Idempotent: skip a file already present and non-empty. Use only the
standard library (urllib); `uv run python ...` needs no extra packages.
"""
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights"
DEFAULT_TAG = "weights-v1"  # the GitHub Release that holds the prebuilt ONNX


def repo_slug() -> str:
    """`owner/repo` from `git remote get-url origin` (ssh or https form)."""
    url = subprocess.check_output(["git", "remote", "get-url", "origin"], text=True).strip()
    slug = url.split("github.com")[-1].lstrip(":/")
    return slug[:-4] if slug.endswith(".git") else slug


def download(names, tag: str | None = None, dest: Path | None = None) -> None:
    tag = tag or os.environ.get("SUPERSLAM_WEIGHTS_TAG", DEFAULT_TAG)
    dest = Path(dest) if dest else WEIGHTS_DIR
    dest.mkdir(parents=True, exist_ok=True)
    slug = repo_slug()
    for name in names:
        out = dest / name
        if out.exists() and out.stat().st_size > 0:
            print(f"[skip] {name} already present ({out.stat().st_size // 1024} KiB)")
            continue
        url = f"https://github.com/{slug}/releases/download/{tag}/{name}"
        print(f"[get ] {url}")
        try:
            urllib.request.urlretrieve(url, out)  # noqa: S310
        except urllib.error.HTTPError as exc:
            print(f"[FAIL] {name}: HTTP {exc.code} ({url})", file=sys.stderr)
            print(
                f"       Is the '{tag}' release published? See the README section "
                "'Maintainer: publishing ONNX', or regenerate with the download_weights_* "
                "scripts + the converters.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"[ok  ] {name} ({out.stat().st_size // 1024} KiB)")

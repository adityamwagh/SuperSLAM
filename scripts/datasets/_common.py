"""Shared helpers for the SuperSLAM dataset downloaders.

Resumable HTTP download (Range), optional SHA-256 verification, archive extraction,
and a consistent default layout: ~/datasets/<dataset> (override with --out or the
SUPERSLAM_DATASETS env var). Standard-library only -- no pip dependencies.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path


def datasets_root() -> Path:
    """Base directory for all datasets (~/datasets by default)."""
    env = os.environ.get("SUPERSLAM_DATASETS")
    return Path(env).expanduser() if env else Path.home() / "datasets"


def dataset_dir(name: str, out: str | None = None) -> Path:
    """Target directory for one dataset, lowercased (e.g. ~/datasets/kitti)."""
    base = Path(out).expanduser() if out else datasets_root() / name.lower()
    base.mkdir(parents=True, exist_ok=True)
    return base


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def download(url: str, dest: Path, resume: bool = True) -> Path:
    """Download url to dest with resume support; skips if already complete."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    # Probe remote size for skip/resume decisions.
    head = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(head) as r:
            total = int(r.headers.get("Content-Length", 0))
    except Exception:
        total = 0

    if dest.exists() and total and dest.stat().st_size == total:
        print(f"  [skip] {dest.name} already complete ({_human(total)})")
        return dest

    existing = tmp.stat().st_size if (resume and tmp.exists()) else 0
    req = urllib.request.Request(url)
    if existing:
        req.add_header("Range", f"bytes={existing}-")
        print(f"  [resume] {dest.name} from {_human(existing)}")
    else:
        print(f"  [get] {dest.name} ({_human(total) if total else 'unknown size'})")

    mode = "ab" if existing else "wb"
    with urllib.request.urlopen(req) as resp, open(tmp, mode) as f:
        done = existing
        grand = total or (existing + int(resp.headers.get("Content-Length", 0)))
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if grand:
                pct = 100.0 * done / grand
                sys.stdout.write(
                    f"\r  {dest.name}: {_human(done)}/{_human(grand)} ({pct:4.1f}%)"
                )
                sys.stdout.flush()
    sys.stdout.write("\n")
    tmp.rename(dest)
    return dest


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path, expected: str | None) -> None:
    if not expected:
        return
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(
            f"checksum mismatch for {path.name}:\n  expected {expected}\n  actual   {actual}"
        )
    print(f"  [ok] sha256 verified {path.name}")


def extract(archive: Path, dest: Path) -> None:
    """Extract a .zip/.tar.*/.tgz archive into dest (idempotent-ish)."""
    print(f"  [extract] {archive.name} -> {dest}")
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            t.extractall(dest)
    else:
        raise SystemExit(f"unknown archive format: {archive}")


def fetch(
    url: str, dest_dir: Path, *, sha: str | None = None, unpack: bool = False
) -> Path:
    """Download (and optionally verify + extract) a single file into dest_dir."""
    name = url.split("/")[-1].split("?")[0]
    archive = download(url, dest_dir / name)
    verify(archive, sha)
    if unpack:
        extract(archive, dest_dir)
    return archive

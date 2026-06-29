#!/usr/bin/env python3
"""Confirm the committed SuperPoint source weights are present (weights/superpoint_v1.pth and
weights/superpoint.pt).

    uv run python scripts/models/download_weights_superpoint.py

These regenerate the ONNX from source. For the normal path, use
download_onnx_engine_superpoint.py.
"""

from pathlib import Path

WEIGHTS = Path(__file__).resolve().parents[2] / "weights"

if __name__ == "__main__":
    present = [
        n for n in ("superpoint_v1.pth", "superpoint.pt") if (WEIGHTS / n).exists()
    ]
    if present:
        print(f"[ok  ] SuperPoint weights present in weights/: {', '.join(present)}")
    else:
        print(
            "[warn] SuperPoint weights not found in weights/ - they are committed to the "
            "repo; ensure the clone is complete (and git-lfs pulled if used)."
        )

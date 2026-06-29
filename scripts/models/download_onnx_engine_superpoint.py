#!/usr/bin/env python3
"""Download the prebuilt SuperPoint ONNX (dense grid, dynamic batch) from the SuperSLAM
GitHub Release into weights/. Build the TensorRT engine afterwards with
`make build-engines-tensorrt10`.

    uv run python scripts/models/download_onnx_engine_superpoint.py

Idempotent: skip the file if already present.
"""

from _release import download

if __name__ == "__main__":
    download(["superpoint_dense_dynamic_batch.onnx"])

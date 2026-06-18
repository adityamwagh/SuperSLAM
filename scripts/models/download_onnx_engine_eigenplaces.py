#!/usr/bin/env python3
"""Download the prebuilt EigenPlaces ONNX (loop-closure place recognition) from the SuperSLAM
GitHub Release into weights/. Build the TensorRT engine afterwards with `make build-engines`.

    uv run python scripts/models/download_onnx_engine_eigenplaces.py

EigenPlaces' ONNX uses EXTERNAL DATA: graph in `eigenplaces_resnet18_512.onnx`, weights in the
sidecar `eigenplaces_resnet18_512.onnx.data` -- both downloaded, both needed together. Only
required if you enable loop closure. Idempotent: skips files already present.
"""
from _release import download

if __name__ == "__main__":
    download(["eigenplaces_resnet18_512.onnx", "eigenplaces_resnet18_512.onnx.data"])

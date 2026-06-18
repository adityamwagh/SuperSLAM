#!/usr/bin/env python3
"""Download the prebuilt LightGlue ONNX from the SuperSLAM GitHub Release into weights/.
Build the TensorRT engine afterwards with `make build-engines`.

    uv run python scripts/models/download_onnx_engine_lightglue.py

LightGlue's ONNX uses EXTERNAL DATA: the graph is in `lightglue_superpoint.onnx` and the
large weight tensors are in the sidecar `lightglue_superpoint.onnx.data` next to it -- both
are downloaded, and both must sit in weights/ together for the engine build to load it.
Idempotent: skips files already present.
"""
from _release import download

if __name__ == "__main__":
    download(["lightglue_superpoint.onnx", "lightglue_superpoint.onnx.data"])

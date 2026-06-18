#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "torch",
#   "onnx",
#   "onnxscript",
#   "lightglue @ git+https://github.com/cvg/LightGlue.git",
# ]
# ///
"""Export a matcher-only LightGlue (SuperPoint features) to ONNX for SuperSLAM.

Usage (uv builds the venv from the inline dependency metadata above):

    uv run utils/convert_lightglue_to_onnx.py --out weights/lightglue_superpoint.onnx
    ./utils/convert_lightglue_to_onnx.py --out weights/lightglue_superpoint.onnx

For CPU-only torch, prepend UV_TORCH_BACKEND=cpu.

Produces an ONNX whose I/O matches LightGlueTRT:

  inputs  : kpts0 [1,N,2] f32, kpts1 [1,M,2] f32,
            desc0 [1,N,256] f32, desc1 [1,M,256] f32   (descriptors are [N,256])
  outputs : matches0 [1,N0] int32 (index into set1 per query, -1 if unmatched),
            mscores0 [1,N0] f32

IMPORTANT contract notes (must stay in sync with src/LightGlueTRT.cc):
  - Keypoints are normalized in the C++ wrapper as (kpt - size/2)/(max(w,h)/2).
    LightGlue's own in-graph normalization is patched to a no-op (see below). The
    wrapper feeds ALREADY-NORMALIZED keypoints and the graph must not re-normalize.
    (The patch also removes the rank-0 size.max() that TensorRT 8.5's ONNX parser
    rejects.)
  - Indices are cast to int32 (TensorRT 8.5 has no int64).
  - Early-exit and point-pruning are disabled.

Then build the engine (dynamic keypoint counts), e.g.:

  trtexec --onnx=weights/lightglue_superpoint.onnx --fp16 \
    --minShapes=kpts0:1x1x2,desc0:1x1x256,kpts1:1x1x2,desc1:1x1x256 \
    --optShapes=kpts0:1x512x2,desc0:1x512x256,kpts1:1x512x2,desc1:1x512x256 \
    --maxShapes=kpts0:1x1024x2,desc0:1x1024x256,kpts1:1x1024x2,desc1:1x1024x256 \
    --saveEngine=weights/lightglue_superpoint_fp16.engine

Wire it via the settings yaml lightglue section:
    lightglue:
      engine_file: "lightglue_superpoint_fp16.engine"
"""

import argparse

import torch
import torch.nn as nn
import lightglue.lightglue as _lg
from lightglue import LightGlue

# Make LightGlue's in-graph keypoint normalization a no-op. The C++ wrapper
# (LightGlueTRT::normalizeKeypoints) normalizes the keypoints; this graph must not.
# The patch also removes the rank-0 `size.max()` that TensorRT 8.5's ONNX parser
# rejects ("at least 1 dimensions are required"). The wrapper feeds
# ALREADY-NORMALIZED keypoints into this graph.
_lg.normalize_keypoints = lambda kpts, size=None: kpts


class LightGlueMatcherONNX(nn.Module):
    """Thin wrapper exposing tensor I/O (no dict, no extractor) for ONNX export."""

    def __init__(self):
        super().__init__()
        lg = LightGlue(features="superpoint")
        # Disable export-hostile dynamic control flow.
        lg.conf.flash = False
        if hasattr(lg, "conf"):
            lg.conf.depth_confidence = -1  # no early exit
            lg.conf.width_confidence = -1  # no point pruning
        self.matcher = lg.eval()

    def forward(self, kpts0, desc0, kpts1, desc1):
        # Keypoints arrive already normalized (the C++ wrapper does it, and
        # LightGlue's in-graph normalization is patched to a no-op above).
        data = {
            "image0": {"keypoints": kpts0, "descriptors": desc0},
            "image1": {"keypoints": kpts1, "descriptors": desc1},
        }
        out = self.matcher(data)
        # Emit LightGlue's RAW fixed-shape outputs (shape [1, N0]) instead of a
        # compacted [K,2]. The C++ wrapper filters the -1 entries on CPU (see
        # LightGlueTRT::postprocessOutputs).
        matches0 = out["matches0"].to(torch.int32)  # [1, N0]: index into set1, or -1
        mscores0 = out["matching_scores0"]  # [1, N0]
        return matches0, mscores0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="weights/lightglue_superpoint.onnx")
    ap.add_argument(
        "--n", type=int, default=512, help="dummy keypoint count for tracing"
    )
    args = ap.parse_args()

    model = LightGlueMatcherONNX().eval()
    n = args.n
    kpts0 = torch.randn(1, n, 2)
    kpts1 = torch.randn(1, n, 2)
    desc0 = torch.randn(1, n, 256)
    desc1 = torch.randn(1, n, 256)

    torch.onnx.export(
        model,
        (kpts0, desc0, kpts1, desc1),
        args.out,
        input_names=["kpts0", "desc0", "kpts1", "desc1"],
        output_names=["matches0", "mscores0"],
        dynamic_axes={
            "kpts0": {1: "n0"},
            "desc0": {1: "n0"},
            "kpts1": {1: "n1"},
            "desc1": {1: "n1"},
            "matches0": {1: "n0"},
            "mscores0": {1: "n0"},
        },
        opset_version=18,
        do_constant_folding=True,
        dynamo=True,  # dynamo-based ONNX exporter
    )
    print(f"Wrote {args.out}. Build the engine with trtexec (see header).")


if __name__ == "__main__":
    main()

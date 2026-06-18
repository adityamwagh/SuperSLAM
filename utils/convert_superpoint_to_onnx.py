#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["torch", "onnx", "onnxscript"]
# ///
"""Export the open SuperPoint (rpautrat, MIT) to the dense ONNX SuperSLAM consumes.

Emit two dense tensors matching the on-device gather contract:
  scores      [batch, H, W]           dense keypoint heatmap (softmax, depth-to-space, NMS)
  descriptors [batch, 256, H/8, W/8]  L2-normalized descriptor grid
The batch axis is symbolic (1 for mono, 2 for batched stereo), so no separate batch-dim surgery.

Usage:
    uv run --with torch --with onnx --with onnxscript python utils/convert_superpoint_to_onnx.py \
        --weights weights/superpoint_v6_from_tf.pth --out weights/superpoint_dense_dynamic_batch.onnx
"""
import argparse
from collections import OrderedDict
from types import SimpleNamespace

import torch
import torch.nn as nn

# SuperPoint architecture from the MIT PyTorch re-implementation by Remi Pautrat and
# Paul-Edouard Sarlin (github.com/rpautrat/SuperPoint). Only the layers are kept; the dense
# forward used for export lives in DenseSuperPoint.


class VGGBlock(nn.Sequential):
    def __init__(self, c_in, c_out, kernel_size, relu=True):
        padding = (kernel_size - 1) // 2
        conv = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=1, padding=padding)
        activation = nn.ReLU(inplace=True) if relu else nn.Identity()
        bn = nn.BatchNorm2d(c_out, eps=0.001)
        super().__init__(
            OrderedDict([("conv", conv), ("activation", activation), ("bn", bn)])
        )


class SuperPoint(nn.Module):
    default_conf = {"descriptor_dim": 256, "channels": [64, 64, 128, 128, 256]}

    def __init__(self, **conf):
        super().__init__()
        self.conf = SimpleNamespace(**{**self.default_conf, **conf})
        self.stride = 2 ** (len(self.conf.channels) - 2)
        channels = [1, *self.conf.channels[:-1]]

        backbone = []
        for i, c in enumerate(channels[1:], 1):
            layers = [VGGBlock(channels[i - 1], c, 3), VGGBlock(c, c, 3)]
            if i < len(channels) - 1:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            backbone.append(nn.Sequential(*layers))
        self.backbone = nn.Sequential(*backbone)

        c = self.conf.channels[-1]
        self.detector = nn.Sequential(
            VGGBlock(channels[-1], c, 3),
            VGGBlock(c, self.stride**2 + 1, 1, relu=False),
        )
        self.descriptor = nn.Sequential(
            VGGBlock(channels[-1], c, 3),
            VGGBlock(c, self.conf.descriptor_dim, 1, relu=False),
        )


class DenseSuperPoint(nn.Module):
    """Return the dense score heatmap and descriptor grid for ONNX export."""

    def __init__(self, sp: SuperPoint, nms_radius: int):
        super().__init__()
        self.backbone = sp.backbone
        self.detector = sp.detector
        self.descriptor = sp.descriptor
        self.stride = sp.stride
        self.nms_radius = nms_radius

    def forward(self, image):
        features = self.backbone(image)
        descriptors = torch.nn.functional.normalize(self.descriptor(features), p=2, dim=1)
        scores = torch.nn.functional.softmax(self.detector(features), 1)[:, :-1]
        b, _, h, w = scores.shape
        s = self.stride
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, s, s)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * s, w * s)
        if self.nms_radius > 0:
            # Keep MaxPool 4D (NCHW) for the TensorRT ONNX parser. Single-pass max
            # suppression: zero any score that is not the local maximum in its window.
            r = self.nms_radius
            s4 = scores.unsqueeze(1)
            pooled = torch.nn.functional.max_pool2d(s4, 2 * r + 1, stride=1, padding=r)
            scores = torch.where(s4 == pooled, s4, torch.zeros_like(s4)).squeeze(1)
        return scores, descriptors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/superpoint_v6_from_tf.pth")
    ap.add_argument("--out", default="weights/superpoint_dense_dynamic_batch.onnx")
    ap.add_argument("--nms-radius", type=int, default=4)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    sp = SuperPoint()
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        state = state.get("model", state.get("state_dict", state))
    sp.load_state_dict(state)
    sp.eval()

    model = DenseSuperPoint(sp, args.nms_radius).eval()
    dummy = torch.randn(1, 1, 480, 640)
    torch.onnx.export(
        model,
        dummy,
        args.out,
        input_names=["input"],
        output_names=["scores", "descriptors"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "scores": {0: "batch", 1: "height", 2: "width"},
            "descriptors": {0: "batch", 2: "grid_h", 3: "grid_w"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"[convert_superpoint_to_onnx] wrote {args.out}")


if __name__ == "__main__":
    main()

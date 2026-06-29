#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["torch", "onnx"]
# ///
"""Export Magic Leap SuperPoint to the dense ONNX SuperSLAM consumes.

Emit two dense tensors matching the on-device gather contract:
  scores      [batch, H, W]           dense keypoint heatmap (softmax, depth-to-space, NMS)
  descriptors [batch, 256, H/8, W/8]  L2-normalized descriptor grid
The batch axis is symbolic (1 for mono, 2 for batched stereo).

Load the committed Magic Leap weights (weights/superpoint_v1.pth, conv1a layout).

Usage:
    uv run --with torch --with onnx python utils/convert_superpoint_to_onnx.py \
        --weights weights/superpoint_v1.pth --out weights/superpoint_dense_dynamic_batch.onnx
"""

import argparse

import torch
import torch.nn as nn


class SuperPoint(nn.Module):
    """SuperPoint detector and descriptor (DeTone, Malisiewicz, Rabinovich, CVPRW 2019).

    The conv1a..conv4b shared encoder, convPa/convPb detector head, and convDa/convDb
    descriptor head match the weights/superpoint_v1.pth state dict.
    """

    def __init__(self, descriptor_dim: int = 256):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256
        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(c5, descriptor_dim, kernel_size=1, stride=1, padding=0)

    def encode(self, image):
        """Run the shared encoder and return the stride-8 feature map."""
        x = self.relu(self.conv1a(image))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        return x


class DenseSuperPoint(nn.Module):
    """Return the dense score heatmap and descriptor grid for ONNX export."""

    def __init__(self, sp: SuperPoint, nms_radius: int):
        super().__init__()
        self.sp = sp
        self.nms_radius = nms_radius

    def forward(self, image):
        x = self.sp.encode(image)
        scores = self.sp.convPb(self.sp.relu(self.sp.convPa(x)))
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
        if self.nms_radius > 0:
            # MaxPool stays 4D (NCHW) for the TensorRT ONNX parser.
            r = self.nms_radius
            s4 = scores.unsqueeze(1)
            pooled = torch.nn.functional.max_pool2d(s4, 2 * r + 1, stride=1, padding=r)
            scores = torch.where(s4 == pooled, s4, torch.zeros_like(s4)).squeeze(1)
        descriptors = self.sp.convDb(self.sp.relu(self.sp.convDa(x)))
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)
        return scores, descriptors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/superpoint_v1.pth")
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

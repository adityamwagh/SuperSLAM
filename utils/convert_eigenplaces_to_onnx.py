#!/usr/bin/env python
"""Convert EigenPlaces (ResNet18, 512-d) to ONNX and safetensors for SuperSLAM loop closure.

Fetch pretrained weights via torch.hub (gmberton/eigenplaces).

  python utils/convert_eigenplaces_to_onnx.py --output_dir output
  python utils/convert_eigenplaces_to_onnx.py --backbone ResNet18 --dim 512
"""

import argparse
import os

import numpy as np
import onnx
import onnxruntime
import torch

from export_safetensors import save_state_dict_safetensors


def to_numpy(t):
    return t.detach().cpu().numpy() if t.requires_grad else t.cpu().numpy()


def main():
    ap = argparse.ArgumentParser(
        description="Convert EigenPlaces to ONNX + safetensors"
    )
    ap.add_argument("--output_dir", default="output", help="output directory")
    ap.add_argument(
        "--backbone",
        default="ResNet18",
        help="EigenPlaces backbone",
    )
    ap.add_argument("--dim", type=int, default=512, help="descriptor dimensionality")
    ap.add_argument(
        "--height",
        type=int,
        default=512,
        help="input height (fixed; must match the loop YAML image_height)",
    )
    ap.add_argument(
        "--width",
        type=int,
        default=512,
        help="input width (fixed; must match the loop YAML image_width)",
    )
    ap.add_argument("--opset", type=int, default=17, help="ONNX opset")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Pretrained EigenPlaces from the authors' hub entry.
    model = torch.hub.load(
        "gmberton/eigenplaces",
        "get_trained_model",
        backbone=args.backbone,
        fc_output_dim=args.dim,
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"EigenPlaces {args.backbone} dim={args.dim}: {n_params / 1e6:.1f}M params")

    stem = f"eigenplaces_{args.backbone.lower()}_{args.dim}"
    onnx_path = os.path.join(args.output_dir, stem + ".onnx")
    st_path = os.path.join(args.output_dir, stem + ".safetensors")

    dummy = torch.randn(1, 3, args.height, args.width)
    with torch.no_grad():
        torch_out = model(dummy)

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["descriptor"],
        dynamic_axes={
            "image": {0: "batch"},
            "descriptor": {0: "batch"},
        },
    )

    onnx.checker.check_model(onnx.load(onnx_path))
    sess = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {sess.get_inputs()[0].name: to_numpy(dummy)})
    np.testing.assert_allclose(to_numpy(torch_out), ort_out[0], rtol=1e-3, atol=1e-5)
    print(f"ONNX verified against PyTorch -> {onnx_path}")

    save_state_dict_safetensors(model.state_dict(), st_path)
    print("Done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Relax the SuperPoint ONNX input batch dimension to symbolic by graph surgery.

Set input.dim[0] of the source ONNX to 'batch'. The outputs are already batch-symbolic.

Usage:
    uv run --with onnx python utils/make_dynamic_batch_onnx.py \
        --in weights/superpoint_dense_int32.onnx --out weights/superpoint_dense_dynamic_batch.onnx

Build the engine with scripts/rebuild_engines.sh.
"""

import argparse

import onnx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default="weights/superpoint_dense_int32.onnx")
    ap.add_argument(
        "--out", dest="dst", default="weights/superpoint_dense_dynamic_batch.onnx"
    )
    args = ap.parse_args()

    model = onnx.load(args.src)
    batch_dim = model.graph.input[0].type.tensor_type.shape.dim[0]
    batch_dim.ClearField("dim_value")  # drop the literal 1
    batch_dim.dim_param = "batch"  # make it symbolic

    onnx.checker.check_model(model)
    onnx.save(model, args.dst)

    shape = [
        d.dim_param or d.dim_value
        for d in model.graph.input[0].type.tensor_type.shape.dim
    ]
    print(f"[make_dynamic_batch_onnx] {args.src} to {args.dst}  input={shape}")


if __name__ == "__main__":
    main()

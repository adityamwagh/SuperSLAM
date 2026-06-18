#!/usr/bin/env python3
"""No checkpoint to download. The convert_lightglue_to_onnx.py export fetches LightGlue
source weights via the `lightglue` package. Print the export command for regenerating the
ONNX from source.

    uv run python scripts/models/download_weights_lightglue.py

For the normal path use download_onnx_engine_lightglue.py (prebuilt ONNX from the release).
"""
if __name__ == "__main__":
    print(
        "LightGlue weights are downloaded by the `lightglue` package at export time.\n"
        "To regenerate the ONNX from source:\n"
        "  uv run --with torch --with onnx --with lightglue \\\n"
        "      python utils/convert_lightglue_to_onnx.py \\\n"
        "      --out weights/lightglue_superpoint.onnx\n"
        "Otherwise just run: uv run python scripts/models/download_onnx_engine_lightglue.py"
    )

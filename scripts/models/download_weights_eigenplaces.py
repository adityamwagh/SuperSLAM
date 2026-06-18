#!/usr/bin/env python3
"""No checkpoint to download. The convert_eigenplaces_to_onnx.py export fetches EigenPlaces
source weights from torch.hub (gmberton/eigenplaces). Print the export command for
regenerating the ONNX from source.

    uv run python scripts/models/download_weights_eigenplaces.py

For the normal path use download_onnx_engine_eigenplaces.py (prebuilt ONNX from the release).
"""
if __name__ == "__main__":
    print(
        "EigenPlaces weights are downloaded from torch.hub (gmberton/eigenplaces) at export "
        "time.\nTo regenerate the ONNX from source:\n"
        "  uv run --with torch --with onnx --with onnxruntime \\\n"
        "      python utils/convert_eigenplaces_to_onnx.py --output_dir weights\n"
        "Otherwise just run: uv run python scripts/models/download_onnx_engine_eigenplaces.py"
    )

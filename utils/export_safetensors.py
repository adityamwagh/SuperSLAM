#!/usr/bin/env python
"""Save a PyTorch state_dict to the .safetensors format."""

from __future__ import annotations

import os

import torch


def save_state_dict_safetensors(state_dict: dict, path: str) -> str:
    """Write a contiguous CPU state_dict to `path` as safetensors and return the path."""
    try:
        from safetensors.torch import save_file
    except ImportError as e:  # pragma: no cover - dependency hint
        raise SystemExit("safetensors not installed: pip install safetensors") from e

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # safetensors requires contiguous CPU tensors and no shared storage.
    clean = {
        k: v.detach().cpu().contiguous()
        for k, v in state_dict.items()
        if isinstance(v, torch.Tensor)
    }
    save_file(clean, path)
    print(f"Saved safetensors -> {path} ({len(clean)} tensors)")
    return path

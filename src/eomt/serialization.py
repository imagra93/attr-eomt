"""Checkpoint metadata wrapping and model (re)loading.

Checkpoints are plain ``torch.save`` dicts carrying the state dict plus enough
metadata (size / task / nc / names / imgsz) to rebuild the model with
:func:`load_model` — no external registry needed.
"""

from __future__ import annotations

import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch

from .config import HIDDEN_TO_SIZE
from .model import EoMTModel, build_model

SCHEMA_VERSION = "1.0"


def get_version() -> str:
    """Return the installed package version, with an editable-install fallback."""
    try:
        return version("libre-eomt")
    except PackageNotFoundError:
        return "0.0.0.dev0"


def normalize_names(names: Any, nc: int) -> dict[int, str]:
    """Normalize class names to a canonical ``{int: str}`` covering 0..nc-1."""
    if names is None:
        return {i: f"class_{i}" for i in range(nc)}
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not isinstance(names, dict):
        raise ValueError("names must be a dict[int, str] or list[str].")
    normalized = {int(k): str(v) for k, v in names.items()}
    missing = [i for i in range(nc) if i not in normalized]
    if missing:
        warnings.warn(
            f"names missing class indices {missing}; padding with class_i labels.",
            RuntimeWarning,
            stacklevel=2,
        )
    return {i: normalized.get(i, f"class_{i}") for i in range(nc)}


def wrap_checkpoint(
    state_dict: dict[str, torch.Tensor],
    *,
    size: str,
    nc: int,
    imgsz: int,
    names: dict[int, str] | list[str] | None = None,
    task: str = "instance",
    family: str = "eomt",
    **extra: Any,
) -> dict[str, Any]:
    """Build a metadata-wrapped checkpoint that :func:`load_model` can restore."""
    checkpoint: dict[str, Any] = {
        "model": state_dict,
        "schema_version": SCHEMA_VERSION,
        "eomt_version": get_version(),
        "model_family": family,
        "size": size,
        "task": task,
        "nc": int(nc),
        "names": normalize_names(names, nc),
        "imgsz": int(imgsz),
    }
    checkpoint.update({k: v for k, v in extra.items() if v is not None})
    return checkpoint


def save_checkpoint(checkpoint: dict[str, Any], path: str | Path) -> None:
    """Atomically write a checkpoint to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, tmp)
    tmp.rename(path)


def load_raw(path: str | Path, *, map_location: Any = "cpu") -> dict[str, Any]:
    """Load a checkpoint dict from disk."""
    return torch.load(path, map_location=map_location, weights_only=False)


def _resolve_device(device: str) -> torch.device:
    if device in ("", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_model(path: str | Path, *, device: str = "auto") -> EoMTModel:
    """Rebuild an :class:`EoMTModel` from a wrapped checkpoint and load its weights."""
    ckpt = load_raw(path)
    if "model" not in ckpt:
        raise ValueError(f"{path} is not a wrapped EoMT checkpoint (no 'model' key).")
    state = ckpt["model"]

    size = ckpt.get("size") or _infer_size(state)
    nc = int(ckpt.get("nc") or _infer_nc(state))
    imgsz = int(ckpt.get("imgsz") or _infer_imgsz(state))
    names = ckpt.get("names")

    model = build_model(size, nc=nc, imgsz=imgsz, names=names)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        warnings.warn(
            f"load_state_dict: {len(missing)} missing / {len(unexpected)} unexpected keys.",
            RuntimeWarning,
            stacklevel=2,
        )
    return model.to(_resolve_device(device)).eval()


# --- best-effort inference of metadata from a bare state dict ---------------


def _infer_size(state: dict) -> str:
    for key in ("eomt.layernorm.weight", "layernorm.weight"):
        if key in state:
            size = HIDDEN_TO_SIZE.get(int(state[key].shape[0]))
            if size:
                return size
    raise ValueError("could not infer model size from state dict.")


def _infer_nc(state: dict) -> int:
    for key in ("eomt.class_predictor.weight", "class_predictor.weight"):
        if key in state:
            return int(state[key].shape[0]) - 1  # drop the +1 null class
    raise ValueError("could not infer nc from state dict.")


def _infer_imgsz(state: dict, patch_size: int = 14) -> int:
    import math

    for key in (
        "eomt.embeddings.position_embeddings.weight",
        "embeddings.position_embeddings.weight",
    ):
        if key in state:
            grid = int(math.isqrt(int(state[key].shape[0])))
            return grid * patch_size
    raise ValueError("could not infer imgsz from state dict.")

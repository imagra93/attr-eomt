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

from .config import HIDDEN_TO_SIZE, aux_specs_from_meta, aux_specs_to_meta
from .model import EoMTModel, build_model

SCHEMA_VERSION = "1.0"


def get_version() -> str:
    """Return the installed package version, with an editable-install fallback."""
    try:
        return version("attr-eomt")
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
    aux_heads: list | None = None,
    aux_head_arch: dict | None = None,
    letterbox: bool = False,
    loss_weights: dict | None = None,
    num_upscale_blocks: int | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build a metadata-wrapped checkpoint that :func:`load_model` can restore.

    ``aux_head_arch`` records the secondary-head network shape (``layers`` /
    ``hidden`` / ``dropout``) so an MLP head is rebuilt identically on reload —
    without it a non-linear head would be silently dropped by ``strict=False``.
    ``loss_weights`` records the segmentation-loss/criterion weights so a tuned
    objective is rebuilt on reload (the criterion is built once from the config, so
    a bare reload would otherwise reset them to defaults). ``num_upscale_blocks``
    records the mask-head depth, which is load-bearing for the ``upscale_block``
    weight shapes.
    """
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
        "letterbox": bool(letterbox),
        "aux_heads": aux_specs_to_meta(aux_heads),
    }
    if aux_heads and aux_head_arch is not None:
        checkpoint["aux_head_arch"] = dict(aux_head_arch)
    if loss_weights is not None:
        checkpoint["loss_weights"] = dict(loss_weights)
    if num_upscale_blocks is not None:
        checkpoint["num_upscale_blocks"] = int(num_upscale_blocks)
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


def resolve_checkpoint(path: str | Path, *, prefer: str = "best") -> Path:
    """Resolve a checkpoint path that may be a file **or** a run/weights folder.

    A file is returned as-is. A directory is searched — itself and its
    ``weights/`` subdir — for ``best.pt`` / ``last.pt``. ``prefer`` sets the
    order: ``"best"`` for inference, ``"last"`` for resuming training. So
    ``runs/train/eomt`` (or ``runs/train/eomt/weights``) resolves to the right
    checkpoint without naming the file.
    """
    p = Path(path)
    if p.is_file():
        return p
    if p.is_dir():
        order = ("best.pt", "last.pt") if prefer == "best" else ("last.pt", "best.pt")
        for d in (p, p / "weights"):
            for fname in order:
                cand = d / fname
                if cand.is_file():
                    return cand
        raise FileNotFoundError(
            f"no {' or '.join(order)} found under {p} or {p / 'weights'}"
        )
    raise FileNotFoundError(f"checkpoint path does not exist: {p}")


def _resolve_device(device: str) -> torch.device:
    if device in ("", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_model(path: str | Path, *, device: str = "auto") -> EoMTModel:
    """Rebuild an :class:`EoMTModel` from a wrapped checkpoint and load its weights.

    ``path`` may be a checkpoint file or a run/weights folder (``best.pt`` is
    preferred); the model size is recovered from the checkpoint metadata.
    """
    path = resolve_checkpoint(path, prefer="best")
    ckpt = load_raw(path)
    if "model" not in ckpt:
        raise ValueError(f"{path} is not a wrapped EoMT checkpoint (no 'model' key).")
    state = ckpt["model"]

    size = ckpt.get("size") or _infer_size(state)
    nc = int(ckpt.get("nc") or _infer_nc(state))
    imgsz = int(ckpt.get("imgsz") or _infer_imgsz(state))
    names = ckpt.get("names")
    # Head family ("instance" | "detect") is recorded in ``task``; fall back to the
    # state dict (a box head has ``eomt.box_head.*`` keys, masks have ``eomt.mask_head.*``).
    family = ckpt.get("task") or _infer_family(state)
    aux_heads = aux_specs_from_meta(ckpt.get("aux_heads")) or _infer_aux_heads(state)
    # Rebuild the exact secondary-head shape. Checkpoints written before aux_head_arch
    # existed only ever had single Linear heads, so default to that for them.
    aux_head_arch = ckpt.get("aux_head_arch") or ({"layers": 1} if aux_heads else None)
    # Restore the tuned criterion + mask-head depth. ``num_upscale_blocks`` changes
    # the ``upscale_block`` weight shapes, so it must match the saved tensors for a
    # clean load — recover it from the state dict when the metadata predates it.
    loss_weights = ckpt.get("loss_weights")
    num_upscale_blocks = ckpt.get("num_upscale_blocks") or _infer_num_upscale_blocks(state)

    model = build_model(
        size,
        nc=nc,
        imgsz=imgsz,
        names=names,
        family=family,
        aux_heads=aux_heads,
        aux_head_arch=aux_head_arch,
        loss_weights=loss_weights,
        num_upscale_blocks=num_upscale_blocks,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        warnings.warn(
            f"load_state_dict: {len(missing)} missing / {len(unexpected)} unexpected keys.",
            RuntimeWarning,
            stacklevel=2,
        )
    # Record the preprocessing mode the model was trained with so val/predict
    # match it (older checkpoints predate this flag → legacy stretch resize).
    model.preprocess_letterbox = bool(ckpt.get("letterbox", False))
    return model.to(_resolve_device(device)).eval()


# --- best-effort inference of metadata from a bare state dict ---------------


def _infer_size(state: dict) -> str:
    for key in ("eomt.layernorm.weight", "layernorm.weight"):
        if key in state:
            size = HIDDEN_TO_SIZE.get(int(state[key].shape[0]))
            if size:
                return size
    raise ValueError("could not infer model size from state dict.")


def _infer_family(state: dict) -> str:
    """Recover the head family from the state dict: box head -> detect, else instance."""
    if any(k.startswith(("eomt.box_head.", "box_head.")) for k in state):
        return "detect"
    return "instance"


def _infer_nc(state: dict) -> int:
    for key in ("eomt.class_predictor.weight", "class_predictor.weight"):
        if key in state:
            return int(state[key].shape[0]) - 1  # drop the +1 null class
    raise ValueError("could not infer nc from state dict.")


def _infer_aux_heads(state: dict) -> list:
    """Recover aux-head specs from ``aux_heads.<name>.weight`` keys (names lost)."""
    from .config import AuxHeadSpec

    specs = []
    for key, tensor in state.items():
        for prefix in ("aux_heads.", "eomt_aux_heads."):
            if key.startswith(prefix) and key.endswith(".weight"):
                name = key[len(prefix) : -len(".weight")]
                # Only bare Linear heads are inferable from a metadata-less state dict;
                # an MLP head (``aux_heads.<name>.<idx>.weight``) needs the saved
                # ``aux_head_arch`` to rebuild and is skipped here.
                if "." in name:
                    continue
                specs.append(AuxHeadSpec(name=name, num_classes=int(tensor.shape[0])))
    return specs


def _infer_num_upscale_blocks(state: dict) -> int | None:
    """Recover the mask-head upscale depth by counting ``upscale_block.block.<i>`` indices.

    Returns ``None`` when no upscale-block keys are present (let the size preset
    default apply). Tolerates the ``eomt.`` prefix.
    """
    idxs = set()
    for key in state:
        for prefix in ("eomt.upscale_block.block.", "upscale_block.block."):
            if key.startswith(prefix):
                rest = key[len(prefix) :]
                head = rest.split(".", 1)[0]
                if head.isdigit():
                    idxs.add(int(head))
                break
    return (max(idxs) + 1) if idxs else None


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

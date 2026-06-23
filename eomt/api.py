"""High-level :class:`EoMT` interface — init from a checkpoint or a size, then
``train`` / ``val`` / ``predict``.

    from eomt import EoMT

    model = EoMT("l")                       # fresh large model (DINOv2 backbone)
    model.train(data="coco", epochs=50)     # COCO auto-downloads if missing

    model = EoMT("runs/train/eomt-l")       # reload a run (everything auto-detected)
    model.val(data="coco")
    results = model.predict("images/", plot=True)

The class is a thin orchestration layer over the existing engine functions; the
trainable network itself is :class:`~eomt.model.EoMTModel`, reachable via
``model.model``.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .config import SIZES
from .data import CocoValImages, load_data_config
from .engine import evaluate as _evaluate
from .engine import predict as _predict
from .engine import train as _train
from .model import build_model, load_dinov2_backbone
from .serialization import load_model, save_checkpoint, wrap_checkpoint

#: Named dataset aliases that resolve to a bundled YAML config.
_DATASET_ALIASES = {"coco": "configs/coco.yaml"}


def _looks_like_checkpoint(spec: str | Path) -> bool:
    """True if ``spec`` points at a checkpoint file or a run/weights folder."""
    p = Path(spec)
    if p.suffix == ".pt":
        return True
    if p.is_file():
        return True
    if p.is_dir():
        # A run folder if it (or its weights/ subdir) holds a best/last checkpoint.
        return any((p / sub).is_file() for sub in ("best.pt", "last.pt", "weights/best.pt", "weights/last.pt"))
    return False


def _resolve_data(data: str | Path) -> dict:
    """Resolve a dataset spec (alias like ``"coco"`` or a YAML path) to absolute paths."""
    yaml_path = _DATASET_ALIASES.get(str(data), str(data))
    if not Path(yaml_path).is_file():
        raise FileNotFoundError(
            f"dataset config not found: {yaml_path!r}. Pass a dataset YAML path or one "
            f"of {sorted(_DATASET_ALIASES)} (run from the repo root for the bundled configs)."
        )
    return load_data_config(yaml_path)


class EoMT:
    """An EoMT instance-segmentation model with train/val/predict.

    Construct from either a model size (``"s"`` / ``"b"`` / ``"l"`` — a fresh model
    with a pretrained DINOv2 backbone and a randomly initialized head) or a
    checkpoint (a ``.pt`` file or a run/weights folder — size, classes, image size
    and any secondary heads are all auto-detected from the checkpoint).
    """

    def __init__(self, model: str | Path = "l", *, device: str = "auto", pretrained: bool = True, **build_kwargs):
        self.device = device
        self._ckpt: str | None = None
        self._pretrained = pretrained
        self._build_kwargs = build_kwargs

        if isinstance(model, (str, Path)) and _looks_like_checkpoint(model):
            self._model = load_model(model, device=device)
            self._ckpt = str(model)
            self.size = self._model.size
        elif str(model) in SIZES:
            self.size = str(model)
            self._model = None  # built lazily (avoids a wasted DINOv2 load before train())
        else:
            raise ValueError(
                f"{model!r} is neither a known size {tuple(SIZES)} nor an existing checkpoint."
            )

    # ------------------------------------------------------------------ model
    @property
    def model(self):
        """The underlying trainable :class:`~eomt.model.EoMTModel` (built on demand)."""
        if self._model is None:
            self._model = build_model(self.size, **self._build_kwargs)
            if self._pretrained:
                load_dinov2_backbone(self._model)
            dev = "cuda" if (self.device in ("", "auto") and torch.cuda.is_available()) else \
                ("cpu" if self.device in ("", "auto") else self.device)
            self._model = self._model.to(dev).eval()
        return self._model

    # ------------------------------------------------------------------ train
    def train(self, data: str | Path = "coco", *, resume: bool = False, **hp) -> dict:
        """Train on a COCO-format dataset.

        ``data`` is a dataset YAML path or a known alias (``"coco"`` auto-downloads).
        Extra keyword args (``epochs``, ``batch``, ``lr0``, ``aux_w``, …) are passed
        straight through to the training engine. For a checkpoint-initialized model,
        ``resume=True`` continues the original run; otherwise the checkpoint warm-starts
        a fresh run (fine-tune). Returns the engine's result dict and reloads the best
        weights into this object.
        """
        cfg = _resolve_data(data)
        if not (cfg["train_images"] and cfg["train_json"]):
            raise ValueError(f"dataset {data!r} has no train split (train_images/train_json).")

        if self._ckpt is not None:
            hp["resume" if resume else "init_weights"] = self._ckpt

        result = _train(
            train_images=cfg["train_images"],
            train_json=cfg["train_json"],
            val_images=cfg["val_images"],
            val_json=cfg["val_json"],
            size=self.size,
            device=self.device,
            **hp,
        )
        best = result.get("best") or result.get("last")
        if best:
            self._model = load_model(best, device=self.device)
            self._ckpt = best
        return result

    # -------------------------------------------------------------------- val
    def val(self, data: str | Path = "coco", *, batch: int = 4, workers: int = 4,
            conf_thres: float = 0.0, max_det: int = 100, letterbox: bool | None = None, **kw) -> dict:
        """Evaluate on a dataset's val split, returning COCO segm/bbox mAP metrics."""
        cfg = _resolve_data(data)
        if not (cfg["val_images"] and cfg["val_json"]):
            raise ValueError(f"dataset {data!r} has no val split (val_images/val_json).")

        model = self.model
        dev = next(model.parameters()).device
        lb = letterbox if letterbox is not None else bool(getattr(model, "preprocess_letterbox", False))
        val_ds = CocoValImages(cfg["val_images"], cfg["val_json"], imgsz=int(model.image_size), letterbox=lb)
        return _evaluate(
            model, val_ds, device=dev, batch_size=batch, num_workers=workers,
            conf_thres=conf_thres, max_det=max_det, **kw,
        )

    # ---------------------------------------------------------------- predict
    def predict(self, source: str | Path, *, plot: bool = False, save: str | None = "runs/predict",
                conf_thres: float = 0.3, max_det: int = 100, mask_thresh: float = 0.5, **kw) -> list[dict]:
        """Run inference on an image or a directory.

        Returns one result dict per image (``boxes`` / ``scores`` / ``classes`` /
        ``masks`` and, for models with secondary heads, ``aux``). With ``plot=True``
        each image is rendered with masks/boxes/labels and saved under ``save``.
        """
        return _predict(
            self.model, str(source), plot=plot, save=save,
            conf_thres=conf_thres, max_det=max_det, mask_thresh=mask_thresh, **kw,
        )

    # ------------------------------------------------------------------- save
    def save(self, path: str | Path) -> None:
        """Write a self-describing checkpoint (reloadable with ``EoMT(path)``)."""
        m = self.model
        ckpt = wrap_checkpoint(
            m.state_dict(),
            size=m.size, nc=m.nc, imgsz=m.image_size, names=m.names,
            aux_heads=m.aux_specs, aux_head_arch=m.aux_head_arch,
            letterbox=bool(getattr(m, "preprocess_letterbox", False)),
            loss_weights=m.loss_weights, num_upscale_blocks=m.num_upscale_blocks,
        )
        save_checkpoint(ckpt, path)

    def __repr__(self) -> str:
        src = f"ckpt={self._ckpt!r}" if self._ckpt else f"size={self.size!r}"
        return f"EoMT({src})"

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

from .config import SIZES
from .data import CocoValImages, load_data_config
from .device import resolve_device
from .engine import evaluate as _evaluate
from .engine import evaluate_detection as _evaluate_detection
from .engine import predict as _predict
from .engine import train as _train
from .model import build_model, load_dinov2_backbone
from .serialization import (
    format_summary,
    is_hf_ref,
    load_model,
    save_checkpoint,
    summarize_checkpoint,
    wrap_checkpoint,
)

#: Named dataset aliases that resolve to a bundled YAML config.
_DATASET_ALIASES = {"coco": "configs/coco.yaml"}


def _looks_like_checkpoint(spec: str | Path) -> bool:
    """True if ``spec`` points at a checkpoint file, a run/weights folder, or a Hub ref."""
    if is_hf_ref(spec):
        return True
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

    # ----------------------------------------------------------- huggingface
    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        *,
        filename: str = "model.pt",
        revision: str | None = None,
        device: str = "auto",
    ) -> "EoMT":
        """Load a model from a Hugging Face Hub repo (downloaded once, then cached).

            model = EoMT.from_pretrained("imagra93/eomt-l-coco")

        Equivalent to ``EoMT("hf://<repo_id>/<filename>")``. ``revision`` pins a
        branch, tag or commit; ``filename`` selects the checkpoint inside the repo.
        """
        ref = f"hf://{repo_id}/{filename}"
        self = cls.__new__(cls)
        self.device = device
        self._pretrained = True
        self._build_kwargs = {}
        # Resolve via the Hub (cached) and load, recording the ref for repr.
        from .serialization import download_from_hub
        local = download_from_hub(ref, revision=revision)
        self._model = load_model(local, device=device)
        self._ckpt = ref
        self.size = self._model.size
        return self

    def push_to_hub(
        self,
        repo_id: str,
        *,
        filename: str = "model.pt",
        private: bool = True,
        commit_message: str = "Upload EoMT checkpoint",
    ) -> str:
        """Upload this model's weights to a Hugging Face Hub repo (created if absent).

        Writes a self-describing checkpoint (same format as :meth:`save`) and
        uploads it as ``filename``. Returns the repo URL. Requires a Hub token
        (``huggingface-cli login`` or ``HF_TOKEN``). Defaults to a **private** repo.
        """
        try:
            from huggingface_hub import HfApi
        except ImportError as e:  # pragma: no cover
            raise ImportError("push_to_hub needs 'huggingface_hub' (pip install huggingface_hub).") from e

        import tempfile

        api = HfApi()
        api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / filename
            self.save(local)
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message,
            )
        return f"https://huggingface.co/{repo_id}"

    # ------------------------------------------------------------------ model
    @property
    def model(self):
        """The underlying trainable :class:`~eomt.model.EoMTModel` (built on demand)."""
        if self._model is None:
            self._model = build_model(self.size, **self._build_kwargs)
            if self._pretrained:
                load_dinov2_backbone(self._model)
            self._model = self._model.to(resolve_device(self.device)).eval()
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
        """Evaluate on a dataset's val split, returning COCO mAP metrics.

        Segmentation models report ``segm/*`` (+ ``bbox/*``); detection
        (``family="detect"``) models report ``bbox/*`` only.
        """
        cfg = _resolve_data(data)
        if not (cfg["val_images"] and cfg["val_json"]):
            raise ValueError(f"dataset {data!r} has no val split (val_images/val_json).")

        model = self.model
        dev = next(model.parameters()).device
        lb = letterbox if letterbox is not None else bool(getattr(model, "preprocess_letterbox", False))
        val_ds = CocoValImages(
            cfg["val_images"], cfg["val_json"], imgsz=int(model.image_size), letterbox=lb,
            mean=getattr(model, "pixel_mean", None), std=getattr(model, "pixel_std", None),
        )
        eval_fn = _evaluate_detection if getattr(model, "family", "instance") == "detect" else _evaluate
        return eval_fn(
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
            task=getattr(m, "family", "instance"),
            aux_heads=m.aux_specs, aux_head_arch=m.aux_head_arch,
            letterbox=bool(getattr(m, "preprocess_letterbox", False)),
            loss_weights=m.loss_weights, num_upscale_blocks=m.num_upscale_blocks,
            norm_mean=getattr(m, "pixel_mean", None), norm_std=getattr(m, "pixel_std", None),
            patch_size=getattr(m, "patch_size", None),
        )
        save_checkpoint(ckpt, path)

    # ------------------------------------------------------------------- info
    def info(self, *, verbose: bool = True) -> dict:
        """Summarize this model's checkpoint metadata (and optionally print it).

        Requires the model to have been loaded from a checkpoint (``EoMT(path)``);
        for a size-initialized model there is no checkpoint yet — call :meth:`save`
        first. Returns the :func:`~eomt.serialization.summarize_checkpoint` dict.
        """
        if self._ckpt is None:
            raise ValueError("info() needs a checkpoint-backed model; save() one first.")
        summary = summarize_checkpoint(self._ckpt)
        if verbose:
            print(format_summary(summary))
        return summary

    def __repr__(self) -> str:
        src = f"ckpt={self._ckpt!r}" if self._ckpt else f"size={self.size!r}"
        return f"EoMT({src})"

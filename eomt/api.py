"""High-level :class:`EoMT` interface — init from a checkpoint or a size, then
``train`` / ``val`` / ``predict``.

    from eomt import EoMT

    model = EoMT("l")                       # fresh large model (DINOv2 backbone)
    model.train(data="coco")                # COCO auto-downloads if missing; 12 epochs, the official recipe

    model = EoMT("runs/train/eomt-l")       # reload a run (everything auto-detected)
    model.val(data="coco")
    results = model.predict("images/", plot=True)

The class is a thin orchestration layer over the existing engine functions; the
trainable network itself is :class:`~eomt.model.EoMTModel`, reachable via
``model.model``.
"""

from __future__ import annotations

from pathlib import Path

from .config import INFERENCE_DEFAULTS, SIZES
from .data import CocoValImages, load_data_config
from .device import resolve_device
from .engine import evaluate as _evaluate
from .engine import evaluate_detection as _evaluate_detection
from .engine import match as _match
from .engine import predict as _predict
from .engine import track as _track
from .engine import train as _train
from .model import build_model, load_dinov2_backbone
from .serialization import (
    format_summary,
    is_hf_ref,
    load_model,
    load_raw,
    resolve_checkpoint,
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


def _inference_kw(model, **given) -> dict:
    """Each setting as the call gives it, else as the checkpoint saved it, else :data:`INFERENCE_DEFAULTS`."""
    saved = getattr(model, "inference", None) or {}
    return {k: given[k] if given.get(k) is not None else saved.get(k, d) for k, d in INFERENCE_DEFAULTS.items()}


def _resolve_imgsz(imgsz: int | None, model) -> int:
    """Resolve an inference image size, defaulting to the model's trained size.

    The ViT needs a size that is a multiple of the patch size (14); a request that
    isn't is rounded *down* to the nearest valid size (with a warning) rather than
    crashing, so ``imgsz=900`` quietly becomes 896.
    """
    if imgsz is None:
        return int(model.image_size)
    ps = int(getattr(model, "patch_size", 14))
    snapped = max(ps, (int(imgsz) // ps) * ps)
    if snapped != int(imgsz):
        import warnings

        warnings.warn(
            f"imgsz={imgsz} is not a multiple of patch size {ps}; using {snapped} instead.",
            stacklevel=3,
        )
    return snapped


def _state_dict_mb(model) -> float:
    """Serialized size (MB) of a model's state dict — captures packed int8 tensors."""
    import io

    import torch

    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return round(buf.getbuffer().nbytes / 1e6, 1)


def _measure_latency(model, device, *, warmup: int = 3, iters: int = 10) -> float | None:
    """Mean forward latency (ms) on a synthetic input at the model's image size.

    Best-effort: returns ``None`` if a forward pass fails (e.g. an int8 kernel that
    needs a GPU not present here).
    """
    import time

    import torch

    imgsz = int(model.image_size)
    dtype = model.eomt.layernorm.weight.dtype
    x = torch.randn(1, 3, imgsz, imgsz, device=device, dtype=dtype)
    dev = device if isinstance(device, torch.device) else torch.device(device)
    is_cuda = dev.type == "cuda"
    try:
        model.eval()
        with torch.no_grad():
            for _ in range(warmup):
                model(x)
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                model(x)
            if is_cuda:
                torch.cuda.synchronize()
        return round((time.perf_counter() - t0) / iters * 1e3, 2)
    except Exception:  # noqa: BLE001 - latency is informational
        return None


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
        #: torchao recipe metadata once :meth:`compress` has been applied (else None).
        self._compression: dict | None = None

        if isinstance(model, (str, Path)) and _looks_like_checkpoint(model):
            self._model = load_model(model, device=device)
            self._ckpt = str(model)
            self.size = self._model.size
            # Carry forward a compressed checkpoint's recipe so a re-save round-trips it.
            self._compression = getattr(self._model, "_compression_meta", None)
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

        The YAML is also the run config (:mod:`eomt.data.config`): its ``train`` section supplies the keyword args not
        passed here (``init_weights`` included, when this object was built from a size), ``attributes`` which heads
        train and how, ``inference`` the settings saved in the checkpoints.

        Augmentation: the dataset YAML may carry an ``aug`` (or ``train_aug``) mapping (any
        :class:`~eomt.data.transforms.AugConfig` field, or ``preset: legacy``); an ``aug=`` passed here is merged
        over it key by key, and the explicit ``flip_prob`` / ``min_scale`` / ``max_scale`` keywords beat both.
        ``train_transform=`` replaces the built-in pipeline with your own callable.
        """
        cfg = _resolve_data(data)
        if not (cfg["train_images"] and cfg["train_json"]):
            raise ValueError(f"dataset {data!r} has no train split (train_images/train_json).")

        # The run config's `train` section under the explicit keywords; its `init_weights` applies to a model built from
        # a size (a checkpoint given to EoMT() wins), its `attributes` and `inference` as the YAML sets them.
        yaml_hp = dict(cfg.get("train") or {})
        yaml_init = yaml_hp.pop("init_weights", None)
        if resume:  # re-seeding a resumed run would replay its first epoch's shuffle
            yaml_hp.pop("seed", None)
        hp = {**yaml_hp, **hp}
        for key in ("attributes", "inference"):
            if cfg.get(key) is not None:
                hp.setdefault(key, cfg[key])
        if self._ckpt is not None:
            hp["resume" if resume else "init_weights"] = self._ckpt
        elif yaml_init and not resume:
            hp.setdefault("init_weights", yaml_init)

        yaml_aug = cfg.get("train_aug")
        if yaml_aug:
            code_aug = hp.get("aug")
            code_aug = code_aug.to_dict() if hasattr(code_aug, "to_dict") else dict(code_aug or {})
            hp["aug"] = {**yaml_aug, **code_aug}

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
    def val(self, data: str | Path | None = None, *, batch: int = 4, workers: int = 4,
            conf_thres: float = 0.0, max_det: int = 100, letterbox: bool | None = None,
            imgsz: int | None = None, augment: bool | dict = False, tiles: bool | int | dict = False,
            attributes: bool = True, multiview: bool | dict | str | Path | None = None,
            run: str | Path | None = None, sets_file: str | Path | None = None, parts: tuple[str, ...] | None = None,
            link_group_by="auto", group_by: str | list[str] | None = None, **kw) -> dict:
        """Evaluate on a dataset's val split (default ``"coco"``), returning COCO mAP metrics.

        Segmentation models report ``segm/*`` (+ ``bbox/*``); detection
        (``family="detect"``) models report ``bbox/*`` only. ``imgsz`` overrides the
        evaluation image size (defaults to the trained size); the model interpolates
        its positional embeddings to the new resolution, so any multiple of the patch
        size works.

        ``augment`` / ``tiles`` evaluate with test-time augmentation / tiled inference (instance family; see
        :mod:`eomt.tta`; a head marked ``flip_aware`` is swapped back in mirrored views). For a segmentation model with
        secondary heads, ``attributes`` (default on) adds ``aux_acc/<head>`` (accuracy) and ``aux_mae/<head>`` (regression
        heads, real units), scored on the same predictions as the mAP; see :func:`eomt.engine.validate.evaluate`.
        ``group_by`` (an image field) adds the mAP per value of the field.

        ``multiview`` runs the validation of a multi-photo post-training, as the trainer does it, with the same keys as
        its ``metrics.csv`` (without ``val/``): the photos one by one, the real sets and the synthetic sets
        (``parts``: a subset of ``("photos", "synthetic", "real")``). ``True`` rebuilds them from the checkpoint's own run
        (its ``run_config``: data, multiview settings, augmentation, validation size); ``run`` (a run folder, its
        ``args.yaml`` or one of its checkpoints) takes another run's instead, to score this model exactly as that run
        validated its own; a dict, or a YAML file, of ``multiview`` settings replaces the run's; ``data`` (a run config
        YAML) replaces its data, e.g. to score one model on another dataset's validation. ``sets_file`` (a run's
        ``real_val_sets.json``) fixes the real sets, so two models or two dataset versions are scored on the same sets
        of the photos they share. The links are scored at the checkpoint's stored thresholds too, gated by
        ``link_group_by`` (``"auto"``: as they were validated).
        """
        if multiview:
            return self._val_sets(data, multiview, run=run, batch=batch, workers=workers, conf_thres=conf_thres,
                                  max_det=max_det, imgsz=imgsz, sets_file=sets_file, parts=parts,
                                  link_group_by=link_group_by, group_by=group_by, **kw)
        cfg = _resolve_data("coco" if data is None else data)
        if not (cfg["val_images"] and cfg["val_json"]):
            raise ValueError(f"dataset {data!r} has no val split (val_images/val_json).")

        model = self.model
        dev = next(model.parameters()).device
        lb = letterbox if letterbox is not None else bool(getattr(model, "preprocess_letterbox", False))
        val_ds = CocoValImages(
            cfg["val_images"], cfg["val_json"], imgsz=_resolve_imgsz(imgsz, model), letterbox=lb,
            mean=getattr(model, "pixel_mean", None), std=getattr(model, "pixel_std", None), attributes=attributes,
        )
        detect = getattr(model, "family", "instance") == "detect"
        eval_fn = _evaluate_detection if detect else _evaluate
        return eval_fn(
            model, val_ds, device=dev, batch_size=batch, num_workers=workers,
            conf_thres=conf_thres, max_det=max_det, augment=augment, tiles=tiles, group_by=group_by,
            **({} if detect else {"attributes": attributes}), **kw,
        )

    def _val_sets(self, data, multiview, *, run, batch, workers, conf_thres, max_det, imgsz, sets_file, parts,
                  link_group_by, group_by, amp: bool = True) -> dict:
        """:meth:`val` with ``multiview``: the trainer's validation of a multi-photo post-training, standalone."""
        import yaml

        from .data import AugConfig, CocoSubjectSets
        from .engine.train import DEFAULT_MULTIVIEW, _real_val_paths, build_val_parts, validate_parts

        model = self.model
        if run is not None:  # another run's record: its args.yaml, or the run_config of one of its checkpoints
            p = Path(run)
            if p.is_dir() and (p / "args.yaml").is_file():
                p = p / "args.yaml"
            run = (yaml.safe_load(p.read_text()) if p.suffix in (".yaml", ".yml")
                   else load_raw(resolve_checkpoint(p)).get("run_config")) or {}
        else:
            run = (load_raw(resolve_checkpoint(self._ckpt)).get("run_config") or {}) if self._ckpt else {}
        if multiview is True:
            if not run.get("multiview"):
                raise ValueError("the checkpoint records no multi-photo run: pass multiview=<settings dict or YAML>")
            mv = dict(run["multiview"])
        elif isinstance(multiview, (str, Path)):
            mv = yaml.safe_load(Path(multiview).read_text()) or {}
        else:
            mv = dict(multiview)
        mv = {**DEFAULT_MULTIVIEW, **mv}
        if data is not None:
            cfg = _resolve_data(data)
            val_images, val_json = cfg["val_images"], cfg["val_json"]
            aug = cfg.get("train_aug") or (run.get("aug") if isinstance(run.get("aug"), dict) else None)
        else:
            val_images, val_json = run.get("val_images"), run.get("val_json")
            aug = run.get("aug") if isinstance(run.get("aug"), dict) else None
        if not (val_json or (mv.get("real") or {}).get("val_json")):
            raise ValueError("no validation data: pass data=<run config YAML> (the checkpoint records none)")
        # the attribute labels: the model's heads over the val JSON's ids, checked against the model's categories
        maps_json = val_json or _real_val_paths(mv["real"], val_images, val_json)[1]
        maps_images = val_images or _real_val_paths(mv["real"], val_images, val_json)[0]
        probe = CocoValImages(maps_images, maps_json, imgsz=int(model.image_size),
                              attributes=[s.name for s in model.aux_specs])
        mine = {s.name: s for s in model.aux_specs}
        for s in probe.aux_specs:
            if not s.regression and s.names != mine[s.name].names:
                raise ValueError(f"attribute {s.name!r}: the data's categories {s.names} differ from the model's "
                                 f"{mine[s.name].names}")
        size = imgsz or run.get("val_imgsz") or int(model.image_size)
        fixed = sets_file or (mv.get("real") or {}).get("val_sets")
        built = build_val_parts(val_images=val_images, val_json=val_json, imgsz=size,
                                letterbox=bool(getattr(model, "preprocess_letterbox", True)), aug_cfg=AugConfig.resolve(aug),
                                aux=(model.aux_specs, probe._attr_id_maps),
                                is_detect=getattr(model, "family", "instance") == "detect", mv=mv,
                                view_types=(getattr(model, "xview_cfg", None) or {}).get("view_types"),
                                fixed=CocoSubjectSets.read_sets(fixed) if fixed else None)
        if parts is not None:
            built = {k: v for k, v in built.items() if k in parts or (k == "photos_aux" and "photos" in parts)}
        gate = getattr(model, "link_group_by", None) if link_group_by == "auto" else link_group_by
        metrics, _ = validate_parts(
            model, built, device=next(model.parameters()).device, batch=batch, workers=workers, conf_thres=conf_thres,
            max_det=max_det, amp=amp, is_detect=getattr(model, "family", "instance") == "detect",
            aux_iou_gate=float(run.get("aux_iou_gate", 0.5)), aux_class_gate=bool(run.get("aux_class_gate", True)),
            link_group_by=gate, thres={"match": model.match_thres, "reid_set": getattr(model, "reid_thres", None)},
            group_by=group_by if group_by is not None else run.get("val_group_by"),
        )
        return metrics

    # -------------------------------------------------------------- compress
    def compress(self, recipe: str = "int8", *, data: str | Path | None = None,
                 batch: int = 4, workers: int = 4, validate: bool = True,
                 save: str | Path | None = None) -> dict:
        """Quantize the ViT transformer blocks to int8 with ``torchao`` and report the impact.

        Int8 weight-only quantization is **data-free** (it just re-rounds the existing
        weights) and keeps the prediction heads in full precision; the state dict shrinks
        ≈3.4× on ViT-L. With ``data`` set and ``validate=True``, mAP is measured before
        and after. ``save``
        writes a self-describing compressed checkpoint reloadable with ``EoMT(path)``,
        plus a ``compression_metrics.json`` sidecar in the run folder recording these
        size / mAP / latency deltas. Returns the same metrics as a dict.
        """
        from . import compress as _compress

        model = self.model  # realize the fp model first (size-init builds it here)
        dev = next(model.parameters()).device
        source = self._ckpt  # original (fp) checkpoint, before save() reassigns it
        result: dict = {"recipe": recipe}

        def _primary(metrics: dict) -> float | None:
            for key in ("segm/mAP", "bbox/mAP"):
                if key in metrics:
                    return float(metrics[key])
            return next((float(v) for v in metrics.values()), None)

        result["size_mb_before"] = _state_dict_mb(model)
        if validate and data is not None:
            result["mAP_before"] = _primary(self.val(data, batch=batch, workers=workers))

        _compress.compress_model(model, recipe)
        self._compression = _compress.compression_meta(recipe)

        result["size_mb_after"] = _state_dict_mb(model)
        if result["size_mb_before"]:
            result["size_ratio"] = round(result["size_mb_after"] / result["size_mb_before"], 3)
        result["latency_ms"] = _measure_latency(model, dev)
        if validate and data is not None:
            result["mAP_after"] = _primary(self.val(data, batch=batch, workers=workers))
            if result.get("mAP_before") is not None and result["mAP_after"] is not None:
                result["mAP_delta"] = round(result["mAP_after"] - result["mAP_before"], 4)

        if save is not None:
            self.save(save)
            self._ckpt = str(save)
            result["saved"] = str(save)
            self._write_compression_metrics(save, result, source=source, data=data)
        return result

    @staticmethod
    def _write_compression_metrics(save, result: dict, *, source, data) -> str:
        """Drop a ``compression_metrics.json`` sidecar in the compressed run folder.

        Written next to the checkpoint's run dir (stripping a trailing ``weights/``)
        so a compressed run self-documents its size / mAP / latency impact.
        """
        import json

        run_dir = Path(save).parent
        if run_dir.name == "weights":
            run_dir = run_dir.parent
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics = {"source": source, "data": str(data) if data is not None else None,
                   **{k: v for k, v in result.items() if k != "saved"}}
        out = run_dir / "compression_metrics.json"
        out.write_text(json.dumps(metrics, indent=2) + "\n")
        return str(out)

    # ---------------------------------------------------------------- predict
    def predict(self, source: str | Path, *, plot: bool = False, save: str | None = "runs/predict",
                conf_thres: float | None = None, max_det: int | None = None, mask_thresh: float | None = None,
                imgsz: int | None = None, **kw) -> list[dict]:
        """Run inference on an image or a directory.

        Returns one result dict per image (``boxes`` / ``scores`` / ``classes`` /
        ``masks`` and, for models with secondary heads, ``aux``). With ``plot=True``
        each image is rendered with masks/boxes/labels and saved under ``save``.
        ``imgsz`` overrides the inference image size (defaults to the trained size). ``conf_thres`` / ``max_det`` /
        ``mask_thresh`` left out are the checkpoint's (its run config's ``inference``), else
        :data:`INFERENCE_DEFAULTS`.
        ``subject=True`` predicts the photos of each subject together (a model post-trained
        with ``train(multiview=...)``): ``source`` holds one subject's photos, or a subfolder
        per subject; see :func:`eomt.engine.predict.predict`.
        """
        return _predict(
            self.model, str(source), plot=plot, save=save,
            **_inference_kw(self.model, conf_thres=conf_thres, max_det=max_det, mask_thresh=mask_thresh),
            imgsz=_resolve_imgsz(imgsz, self.model), **kw,
        )

    # ------------------------------------------------------------------ track
    def track(self, source: str | Path, *, plot: bool = True, save: str | None = "runs/track",
              conf_thres: float | None = None, max_det: int | None = None, mask_thresh: float | None = None,
              imgsz: int | None = None, min_hits: int = 3, with_attr: bool = False,
              details: bool = False, trace: bool = False, hud: bool = True,
              tracker_kwargs: dict | None = None, **kw) -> list[dict] | dict:
        """Track instances across a video, returning one result dict per frame.

        Each dict carries the usual ``predict`` keys (``boxes`` / ``scores`` /
        ``classes`` / optional ``masks``) plus a persistent ``track_ids`` tensor and a
        ``frame`` index. For models with secondary heads, association runs on the
        main class + box only, and ``aux_track`` holds the temporally-smoothed
        attribute per track (``aux`` keeps the raw per-frame values). With ``plot`` an
        annotated ``.mp4`` colored by track id is written under ``save`` (``hud`` adds a
        ``Frame i/N | Tracks: n`` banner, ``trace=True`` draws motion trails), plus a
        ``<stem>_tracks.json`` per-track summary. ``min_hits`` requires a track to
        persist that many consecutive frames before it is emitted (speck suppression);
        ``tracker_kwargs`` are forwarded to ``ByteTrack`` (e.g.
        ``{"lost_track_buffer": 90}``). ``imgsz`` overrides the inference size.

        ``with_attr`` treats each attribute as part of the class — the tracked identity
        becomes the compound class ``cls-attr1-attr2…`` (one ByteTrack per compound
        class). ``details`` records the per-frame attribute evolution per track and
        changes the return to ``{"frames": [...], "tracks": [...]}``.
        """
        return _track(
            self.model, str(source), plot=plot, save=save,
            **_inference_kw(self.model, conf_thres=conf_thres, max_det=max_det, mask_thresh=mask_thresh),
            imgsz=_resolve_imgsz(imgsz, self.model), min_hits=min_hits,
            with_attr=with_attr, details=details, trace=trace,
            hud=hud, tracker_kwargs=tracker_kwargs, **kw,
        )

    # ------------------------------------------------------------------ match
    def infer_match(self, source: str | Path, *, plot: bool = True,
                    save: str | None = "runs/match", conf_thres: float = 0.3,
                    max_det: int = 30, mask_thresh: float = 0.5,
                    imgsz: int | None = None,
                    group_by: tuple[str, ...] | str | None = "auto",
                    sim_thres: float | None = None, **kw) -> dict:
        """Re-identify instances across a set of photos of the *same subject*.

        Given several photos of one subject from different viewpoints, decides which
        detections are the same physical instance — one instance seen from three
        angles is one identity, not three. Inference only: the fingerprint is the
        per-query embedding the detector already computes, so there is no second model
        and nothing to retrain.

        ``source`` is one folder of photos (non-recursive) = one subject; loop for
        several subjects. Each detection is stamped with an ``identity_ids`` entry, and
        ``identities`` summarizes each one (dominant class, smoothed attributes, how
        many photos it appears in). With ``plot`` a grid image is written under
        ``save`` — every photo, instances colored by identity, connectors between
        matched pairs — alongside a ``<name>_identities.json`` summary.

        ``group_by`` gates which pairs may match at all: ``("class",)`` by default, and
        any aux head name may be added — useful when the primary class is coarser than
        the distinction you care about, so ``("class", "position")`` keeps instances at
        different places on the subject from matching each other. ``None`` disables
        gating, which needs a higher ``sim_thres``. ``sim_thres`` is the cosine floor
        for "same instance" — Hungarian always returns a full assignment, so the
        threshold is what turns "best available partner" into "new instance".

        A cross-view model (post-trained with ``train(multiview=...)``) predicts the photos
        **jointly** — one forward pass in which each photo's queries take what the others
        show, at most 4 photos — and matches on its matching head with its calibrated
        threshold and no class gate (``group_by="auto"``, ``sim_thres=None``); pass
        ``joint=False`` for one pass per photo.
        """
        return _match(
            self.model, str(source), plot=plot, save=save, conf_thres=conf_thres,
            max_det=max_det, mask_thresh=mask_thresh,
            imgsz=_resolve_imgsz(imgsz, self.model), group_by=group_by,
            sim_thres=sim_thres, **kw,
        )

    #: Alias so the method, the engine function and ``scripts/match.py`` agree.
    match = infer_match

    # ------------------------------------------------------------------- save
    def save(self, path: str | Path, *, dtype: str | None = None) -> None:
        """Write a self-describing checkpoint (reloadable with ``EoMT(path)``), with everything the loaded checkpoint
        recorded (inference settings, link calibration, the run, the epoch and metrics it was saved at).

        ``dtype="bf16"`` (or ``"fp16"``) stores the weights and biases of the Linear and Conv layers in that precision,
        about half the file; norms, embeddings and the other parameters stay fp32. ``EoMT(path)`` loads them back into
        an fp32 model (the rounded values), and a service running in bf16 casts them anyway."""
        import torch
        from torch import nn

        m = self.model
        state = m.state_dict()
        if dtype is not None:
            casts = {"bf16": torch.bfloat16, "fp16": torch.float16}
            if dtype not in casts:
                raise ValueError(f"dtype {dtype!r}: 'bf16' or 'fp16'")
            if self._compression:
                raise ValueError("a compressed model keeps its int8 weights: save it without dtype")
            layers = {f"{name}.{p}" for name, mod in m.named_modules()
                      if isinstance(mod, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d))
                      for p in ("weight", "bias") if getattr(mod, p, None) is not None}
            state = {k: v.to(casts[dtype]) if k in layers and v.is_floating_point() else v for k, v in state.items()}
        ckpt = wrap_checkpoint(
            state,
            size=m.size, nc=m.nc, imgsz=m.image_size, names=m.names,
            task=getattr(m, "family", "instance"),
            aux_heads=m.aux_specs, aux_head_arch=m.aux_head_arch,
            letterbox=bool(getattr(m, "preprocess_letterbox", False)),
            loss_weights=m.loss_weights, num_upscale_blocks=m.num_upscale_blocks,
            fpn_scales=getattr(m, "fpn_scales", None),
            aux_box_head=bool(getattr(m, "aux_box_head", False)),
            norm_mean=getattr(m, "pixel_mean", None), norm_std=getattr(m, "pixel_std", None),
            patch_size=getattr(m, "patch_size", None),
            compression=self._compression,
            xview=getattr(m, "xview_cfg", None), match_dim=getattr(m, "match_dim", 0) or None,
            match_residual=getattr(m, "match_residual", False) or None,
            match_thres=getattr(m, "match_thres", None), reid_thres=getattr(m, "reid_thres", None),
            link_embed=getattr(m, "link_embed", None), link_f1=getattr(m, "link_f1", None),
            link_group_by=list(m.link_group_by) if getattr(m, "link_group_by", None) else None,
            inference=getattr(m, "inference", None) or None, dtype=dtype,
            **(getattr(m, "meta", None) or {}),
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

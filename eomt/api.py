"""The high-level :class:`EoMT` interface: build or load a model, then ``train`` / ``val`` / ``predict`` / ``match``.

    from eomt import EoMT

    model = EoMT("l")                               # a new large model (DINOv2 backbone)
    model.train(data="run.yaml")                    # the run config: data, attribute heads, settings
    model = EoMT("runs/train/eomt-l")               # a trained run (everything read from its checkpoint)
    model.val()                                     # on its own validation data
    results = model.predict("images/", plot=True)
    linked = model.match("photos/subject_42/")      # one identity per physical instance across the photos

The class is a thin layer over the engine functions (:mod:`eomt.engine`); the network itself is
:class:`~eomt.model.EoMTModel`, ``model.model``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

from ._log import LOG
from .config import INFERENCE_DEFAULTS, SIZES
from .data import CocoValImages, load_data_config
from .device import resolve_device
from .model import build_model, load_dinov2_backbone
from .serialization import (
    download_from_hub,
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
_DATASET_ALIASES = {"coco": str(Path(__file__).resolve().parent.parent / "configs" / "coco.yaml")}
#: The settings sections of a run (``train()`` keywords and run config sections), merged key by key.
_SECTIONS = ("attributes", "aug", "model", "optim", "aux", "val", "multiview", "inference")


def _looks_like_checkpoint(spec: str | Path) -> bool:
    """True if ``spec`` points at a checkpoint file, a run/weights folder, or a Hub ref."""
    if is_hf_ref(spec):
        return True
    p = Path(spec)
    if p.suffix == ".pt" or p.is_file():
        return True
    return p.is_dir() and any((p / sub).is_file() for sub in ("best.pt", "last.pt", "weights/best.pt", "weights/last.pt"))


def _inference_kw(model, **given) -> dict:
    """Each setting as the call gives it, else as the checkpoint saved it, else :data:`INFERENCE_DEFAULTS`."""
    saved = getattr(model, "inference", None) or {}
    return {k: given[k] if given.get(k) is not None else saved.get(k, d) for k, d in INFERENCE_DEFAULTS.items()}


def _resolve_imgsz(imgsz: int | None, model) -> int:
    """An inference input size: the model's trained one by default; a size that is no multiple of the patch (14) is
    rounded down to one, with a warning (``imgsz=900`` becomes 896)."""
    if imgsz is None:
        return int(model.image_size)
    ps = int(getattr(model, "patch_size", 14))
    snapped = max(ps, (int(imgsz) // ps) * ps)
    if snapped != int(imgsz):
        warnings.warn(f"imgsz={imgsz} is not a multiple of patch size {ps}; using {snapped} instead.", stacklevel=3)
    return snapped


def _resolve_data(data: str | Path) -> dict:
    """A dataset spec (an alias like ``"coco"`` or a YAML path) as absolute paths and run config sections."""
    yaml_path = _DATASET_ALIASES.get(str(data), str(data))
    if not Path(yaml_path).is_file():
        raise FileNotFoundError(f"dataset config not found: {yaml_path!r}. Pass a dataset YAML path or one of "
                                f"{sorted(_DATASET_ALIASES)}.")
    return load_data_config(yaml_path)


class EoMT:
    """An EoMT model with ``train`` / ``val`` / ``predict`` / ``match``.

    Construct from a model size (``"s"`` / ``"b"`` / ``"l"``: a new model with a pretrained DINOv2 backbone and
    randomly initialised heads) or a checkpoint (a ``.pt`` file, a run / weights folder or a ``hf://`` reference: the
    size, classes, input size, attribute heads and settings are read from it).
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
            self._model = None  # built lazily (no wasted DINOv2 load before train())
        else:
            raise ValueError(f"{model!r} is neither a known size {tuple(SIZES)} nor an existing checkpoint.")

    # ----------------------------------------------------------- huggingface
    @classmethod
    def from_pretrained(cls, repo_id: str, *, filename: str = "model.pt", revision: str | None = None,
                        device: str = "auto") -> EoMT:
        """Load a model from a Hugging Face Hub repo (downloaded once, then cached): ``EoMT.from_pretrained(
        "owner/eomt-l-coco")``. ``revision`` pins a branch, tag or commit; ``filename`` selects the checkpoint."""
        self = cls.__new__(cls)
        self.device, self._pretrained, self._build_kwargs = device, True, {}
        self._model = load_model(download_from_hub(f"hf://{repo_id}/{filename}", revision=revision), device=device)
        self._ckpt = f"hf://{repo_id}/{filename}"
        self.size = self._model.size
        return self

    def push_to_hub(self, repo_id: str, *, filename: str = "model.pt", private: bool = True,
                    commit_message: str = "Upload EoMT checkpoint") -> str:
        """Upload this model (a :meth:`save` checkpoint) to a Hugging Face Hub repo, created if absent and **private** by
        default. Needs a Hub token (``huggingface-cli login`` or ``HF_TOKEN``). Returns the repo URL."""
        import tempfile

        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / filename
            self.save(local)
            api.upload_file(path_or_fileobj=str(local), path_in_repo=filename, repo_id=repo_id, repo_type="model",
                            commit_message=commit_message)
        return f"https://huggingface.co/{repo_id}"

    # ------------------------------------------------------------------ model
    @property
    def model(self):
        """The underlying :class:`~eomt.model.EoMTModel` (built on demand)."""
        if self._model is None:
            self._model = build_model(self.size, **self._build_kwargs)
            if self._pretrained:
                load_dinov2_backbone(self._model)
            self._model = self._model.to(resolve_device(self.device)).eval()
        return self._model

    # ------------------------------------------------------------------ train
    def train(self, data: str | Path = "coco", *, resume: bool = False, **kw) -> dict:
        """Train on a COCO-format dataset (:func:`eomt.engine.train.train`).

        ``data`` is a run config YAML (or ``"coco"``): the data paths and the settings sections. Keywords beat it: a
        top-level keyword (``epochs``, ``batch``, ...) replaces the YAML's ``train`` value, and a section given as a mapping
        (``optim={"lr0": 5e-5}``) is merged over the YAML's section key by key. A model loaded from a checkpoint is
        warm-started from it (weights only, a new run), or with ``resume=True`` continues that run (its ``last.pt``).
        Returns the engine's result dict and loads the best weights into this object.
        """
        from .engine.train import train as _train

        if resume and self._ckpt is None:
            raise ValueError("resume=True continues the run this model was loaded from: EoMT('<run>/weights/last.pt')")
        cfg = _resolve_data(data)
        if not (cfg["train_images"] and cfg["train_json"]):
            raise ValueError(f"dataset {data!r} has no train split (train_images / train_json).")
        hp = dict(cfg["train"])
        yaml_init = hp.pop("init_weights", None)
        if resume:  # re-seeding a resumed run would replay its first epoch's shuffle
            hp.pop("seed", None)
        hp.update({k: v for k, v in kw.items() if k not in _SECTIONS})
        for key in _SECTIONS:
            yaml_v, code_v = cfg.get(key), kw.get(key)
            if isinstance(yaml_v, dict) and isinstance(code_v, dict):
                hp[key] = {**yaml_v, **code_v}
            elif code_v is not None or yaml_v is not None:
                hp[key] = code_v if code_v is not None else yaml_v
        if self._ckpt is not None:
            hp["resume" if resume else "init_weights"] = self._ckpt
        elif yaml_init and "init_weights" not in kw:
            hp["init_weights"] = yaml_init
        hp.setdefault("pretrained", self._pretrained)
        result = _train(train_images=cfg["train_images"], train_json=cfg["train_json"], val_images=cfg["val_images"],
                        val_json=cfg["val_json"], size=self.size, device=self.device, **hp)
        best = result.get("best") or result.get("last")
        if best:
            self._model = load_model(best, device=self.device)
            self._ckpt = best
        return result

    # -------------------------------------------------------------------- val
    def _run_config(self, run) -> dict:
        """The recorded run: ``run`` (a run folder, its ``args.yaml`` or one of its checkpoints), else this model's."""
        if run is None:
            return dict((getattr(self.model, "meta", None) or {}).get("run_config") or {})
        import yaml

        p = Path(run)
        if p.is_dir() and (p / "args.yaml").is_file():
            p = p / "args.yaml"
        rc = yaml.safe_load(p.read_text()) if p.suffix in (".yaml", ".yml") else load_raw(resolve_checkpoint(p)).get("run_config")
        return dict(rc or {})

    def val(self, data: str | Path | None = None, *, batch: int = 4, workers: int = 4, conf_thres: float = 0.0,
            max_det: int = 100, imgsz: int | None = None, augment: bool | dict = False, tiles: bool | int | dict = False,
            attributes: bool = True, multiview: bool | dict = False, run: str | Path | None = None,
            sets_file: str | Path | None = None, group_by: str | list[str] | None = None) -> dict:
        """Evaluate on a validation split, returning COCO metrics (``segm/*`` and ``bbox/*``; ``bbox/*`` only for the
        detect family) and, for a model with attribute heads, ``aux_acc/<head>`` (accuracy) and ``aux_mae/<head>``
        (regression heads, real units), scored on the same predictions as the mAP.

        ``data``: a run config YAML (or ``"coco"``); by default the validation data of the run that trained this model
        (or of ``run``). ``imgsz`` overrides the input size (any multiple of 14). ``augment`` / ``tiles`` evaluate
        with test-time augmentation / tiled inference (instance family, see :mod:`eomt.tta`). ``group_by`` (an image
        field) adds the mAP per value of the field.

        ``multiview=True`` runs the validation of a multi-view post-training as the trainer runs it, with the same keys
        as its ``metrics.csv`` (without ``val/``): the photos one by one and the sets of photos of each subject, from
        the run's ``multiview`` settings (a mapping replaces them). ``run`` (a run folder, its ``args.yaml`` or one of
        its checkpoints) scores this model as that run validated its own; ``sets_file`` (a run's ``val_sets.json``)
        fixes the sets, so that two models, or two versions of a dataset, are scored on the same sets.
        """
        from .engine.validate import evaluate, evaluate_detection

        model = self.model
        rc = self._run_config(run)
        if data is not None:
            cfg = _resolve_data(data)
            val_images, val_json = cfg["val_images"], cfg["val_json"]
        else:
            val_images, val_json = rc.get("val_images"), rc.get("val_json")
        if not (val_images and val_json):
            raise ValueError("no validation data: pass data=<run config YAML> (the checkpoint records none)")
        if multiview:
            return self._val_sets(rc, multiview, val_images, val_json, batch=batch, workers=workers,
                                  conf_thres=conf_thres, max_det=max_det, imgsz=imgsz, sets_file=sets_file,
                                  group_by=group_by)
        lb = bool(getattr(model, "preprocess_letterbox", True))
        val_ds = CocoValImages(val_images, val_json, imgsz=_resolve_imgsz(imgsz, model), letterbox=lb,
                               mean=model.pixel_mean, std=model.pixel_std, attributes=attributes)
        dev = next(model.parameters()).device
        if model.family == "detect":
            return evaluate_detection(model, val_ds, device=dev, batch_size=batch, num_workers=workers,
                                      conf_thres=conf_thres, max_det=max_det, amp=True, augment=augment, tiles=tiles,
                                      group_by=group_by)
        return evaluate(model, val_ds, device=dev, batch_size=batch, num_workers=workers, conf_thres=conf_thres,
                        max_det=max_det, amp=True, augment=augment, tiles=tiles, group_by=group_by,
                        attributes=attributes)

    def _val_sets(self, rc: dict, multiview, val_images, val_json, *, batch, workers, conf_thres, max_det, imgsz,
                  sets_file, group_by) -> dict:
        """:meth:`val` with ``multiview``: the trainer's validation of a multi-view post-training, standalone."""
        from .data import AugConfig, CocoSubjectSets
        from .engine.train import MultiviewConfig, _section, build_val_parts, validate_parts
        from .reid import validate_group_by

        model = self.model
        rec = rc.get("multiview") if multiview is True else dict(multiview)
        if not rec or "sets" not in rec:
            raise ValueError("the run records no multi-view post-training of this version: pass run=<such a run> or "
                             "multiview=<MultiviewConfig settings>")
        mv = _section(MultiviewConfig, rec, "multiview")
        # the attribute labels: the model's heads over the val JSON's ids, checked against the model's categories
        probe = CocoValImages(val_images, val_json, imgsz=int(model.image_size), attributes=[s.name for s in model.aux_specs])
        mine = {s.name: s for s in model.aux_specs}
        for s in probe.aux_specs:
            if not s.regression and s.names != mine[s.name].names:
                raise ValueError(f"attribute {s.name!r}: the data's categories {s.names} differ from the model's "
                                 f"{mine[s.name].names}")
        fixed = sets_file or mv.val_sets
        parts = build_val_parts(val_images=val_images, val_json=val_json,
                                imgsz=imgsz or (rc.get("val") or {}).get("imgsz") or int(model.image_size),
                                aug_cfg=AugConfig.resolve(rc.get("aug") if isinstance(rc.get("aug"), dict) else None),
                                aux=(model.aux_specs, probe._attr_id_maps), sets=mv.sets,
                                fixed=CocoSubjectSets.read_sets(fixed) if fixed else None)
        aux = rc.get("aux") or {}
        link_conf = (rc.get("inference") or {}).get("conf_thres", _inference_kw(model)["conf_thres"])
        metrics, _ = validate_parts(
            model, parts, device=next(model.parameters()).device, batch=batch, workers=workers, conf_thres=conf_thres,
            max_det=max_det, amp=True, aux_iou_gate=float(aux.get("iou_gate", 0.5)),
            aux_class_gate=bool(aux.get("class_gate", True)),
            link_group_by=list(validate_group_by(mv.gate, model.aux_specs)) or None, link_conf=float(link_conf),
            thres=model.match_thres, group_by=group_by if group_by is not None else (rc.get("val") or {}).get("group_by"),
        )
        return metrics

    # ---------------------------------------------------------------- predict
    def predict(self, source: str | Path, *, plot: bool = False, save: str | None = "runs/predict",
                conf_thres: float | None = None, max_det: int | None = None, mask_thresh: float | None = None,
                imgsz: int | None = None, batch: int = 8, stream: bool = False, amp: bool | str = "auto",
                augment: bool | dict = False, tiles: bool | int | dict = False, subject: bool = False, **draw):
        """Run inference on an image or a folder (:func:`eomt.engine.predict.predict`): one result dict per image
        (``boxes`` / ``scores`` / ``classes`` / ``masks`` and, with attribute heads, ``aux`` / ``aux_values``), or a
        generator of them with ``stream=True``. ``batch`` photos per forward pass, in bf16 where the GPU has it.
        ``conf_thres`` / ``max_det`` / ``mask_thresh`` left out are the checkpoint's (its run config's ``inference``),
        else :data:`~eomt.config.INFERENCE_DEFAULTS`; ``imgsz`` the trained size. ``plot=True`` renders every image
        under ``save``; ``augment`` / ``tiles`` add test-time augmentation / tiled inference; ``subject=True`` predicts
        the photos of each subject together (a model post-trained with ``multiview``): ``source`` holds one subject's
        photos, or a subfolder per subject. ``draw`` options: ``alpha``, ``draw_boxes``, ``show_scores``, ``color_by``.
        """
        from .engine.predict import predict as _predict

        return _predict(self.model, str(source), plot=plot, save=save,
                        **_inference_kw(self.model, conf_thres=conf_thres, max_det=max_det, mask_thresh=mask_thresh),
                        imgsz=_resolve_imgsz(imgsz, self.model), batch=batch, stream=stream, amp=amp, augment=augment,
                        tiles=tiles, subject=subject, **draw)

    # ------------------------------------------------------------------ match
    def match(self, source: str | Path, *, plot: bool = True, save: str | None = "runs/match",
              conf_thres: float | None = None, max_det: int | None = None, mask_thresh: float | None = None,
              imgsz: int | None = None, amp: bool | str = "auto", groups: list[list[int]] | None = None,
              group_by: tuple[str, ...] | str | None = "auto", sim_thres: float | None = None, **draw) -> dict:
        """Which detections across the photos of one subject (a folder) are the same physical instance
        (:func:`eomt.engine.match.match`): one identity per instance, with its members, fused attributes and values.

        The photos are linked by the model's matching head when it has one (``multiview`` with ``matching``), else by
        the per-query embedding the detector computes anyway (re-identification), at the threshold and within the gate
        (``group_by``: ``"class"`` and categorical attribute heads) its validation calibrated; ``sim_thres`` and
        ``group_by`` override them. A cross-view model predicts the photos together, its ``max_views`` at a time
        (``groups`` chooses them). With ``plot``, a grid of the photos colored by identity and a JSON summary are
        written under ``save``.
        """
        from .engine.match import match as _match

        return _match(self.model, str(source), plot=plot, save=save,
                      **_inference_kw(self.model, conf_thres=conf_thres, max_det=max_det, mask_thresh=mask_thresh),
                      imgsz=_resolve_imgsz(imgsz, self.model), amp=amp, groups=groups, group_by=group_by,
                      sim_thres=sim_thres, **draw)

    # ------------------------------------------------------------------- save
    def save(self, path: str | Path, *, dtype: str | None = None) -> None:
        """Write a self-describing checkpoint (reloadable with ``EoMT(path)``), with everything the loaded checkpoint
        recorded (inference settings, link calibration, the run, the epoch and metrics it was saved at).

        ``dtype="bf16"`` (or ``"fp16"``) stores the weights and biases of the Linear and Conv layers in that precision,
        about half the file; ``EoMT(path)`` loads them back into an fp32 model (the rounded values)."""
        import torch
        from torch import nn

        m = self.model
        state = m.state_dict()
        if dtype is not None:
            casts = {"bf16": torch.bfloat16, "fp16": torch.float16}
            if dtype not in casts:
                raise ValueError(f"dtype {dtype!r}: 'bf16' or 'fp16'")
            layers = {f"{name}.{p}" for name, mod in m.named_modules()
                      if isinstance(mod, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d))
                      for p in ("weight", "bias") if getattr(mod, p, None) is not None}
            state = {k: v.to(casts[dtype]) if k in layers and v.is_floating_point() else v for k, v in state.items()}
        ckpt = wrap_checkpoint(
            state, size=m.size, nc=m.nc, imgsz=m.image_size, names=m.names, task=m.family, aux_heads=m.aux_specs,
            aux_head_arch=m.aux_head_arch, letterbox=bool(getattr(m, "preprocess_letterbox", True)),
            loss_weights=m.loss_weights, num_upscale_blocks=m.num_upscale_blocks, fpn_scales=m.fpn_scales,
            norm_mean=m.pixel_mean, norm_std=m.pixel_std, patch_size=m.patch_size, xview=m.xview_cfg,
            matcher=m.matcher_cfg, match_thres=m.match_thres,
            link_group_by=list(m.link_group_by) if m.link_group_by else None,
            inference=getattr(m, "inference", None) or None, dtype=dtype, **(getattr(m, "meta", None) or {}),
        )
        save_checkpoint(ckpt, path)

    # ------------------------------------------------------------------- info
    def info(self, *, verbose: bool = True) -> dict:
        """Summarize this model's checkpoint metadata (a model loaded from a checkpoint; :meth:`save` one first)."""
        if self._ckpt is None:
            raise ValueError("info() needs a checkpoint-backed model; save() one first.")
        summary = summarize_checkpoint(self._ckpt)
        if verbose:
            LOG.info(format_summary(summary))
        return summary

    def __repr__(self) -> str:
        return f"EoMT({f'ckpt={self._ckpt!r}' if self._ckpt else f'size={self.size!r}'})"

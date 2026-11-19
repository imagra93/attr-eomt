"""Training loop for EoMT (instance segmentation or detection) with per-instance attribute heads.

:func:`train` follows the reference EoMT recipe (DINOv2 initialisation, AdamW with layer-wise LR decay, a two-stage
warmup and polynomial decay, masked attention annealed block by block, EMA of the weights) and validates with COCO mAP
in the regime the model is trained in, as the official code does. Its settings are grouped by concern, each section a
dataclass or a plain mapping of its fields (the run config YAML's sections, :mod:`eomt.data.config`):

* the keywords of :func:`train`: the data, the model size, the length of the schedule, the batch, the device, the run
  folder, resume / warm start;
* ``model`` (:class:`ModelConfig`): the architecture and loss weights of a new model;
* ``optim`` (:class:`OptimConfig`): optimizer, LR schedule, EMA and masked-attention annealing;
* ``aux`` (:class:`AuxConfig`): how the attribute heads train;
* ``aug`` (:class:`~eomt.data.transforms.AugConfig`): the augmentation;
* ``val`` (:class:`ValConfig`): validation and the metric ``best.pt`` is chosen on;
* ``multiview`` (:class:`MultiviewConfig`): the post-training on sets of photos of one subject — cross-view attention
  and / or a matching head that links instances across photos;
* ``attributes`` and ``inference``: which attribute heads train and how; the predict defaults saved with the model.

``device="0,1"`` trains on several GPUs, one process each (DistributedDataParallel).
"""

from __future__ import annotations

import contextlib
import csv
import dataclasses
import hashlib
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .._log import LOG
from ..aux_cls import aux_loss, gate_indices, xview_attention_loss
from ..lora import add_lora, has_lora, merge_state
from ..matcher import DEFAULT_MATCHER, set_match_loss
from ..config import DEFAULT_IMAGE_SIZE, INFERENCE_DEFAULTS, PATCH_SIZE, aux_specs_from_meta, normalize_loss_weights
from ..data import CocoDetection, CocoInstanceSeg, CocoSubjectSets, CocoValImages, collate_sets, collate_train
from ..data.coco import _attr_label
from ..data.subjects import SubjectSetConfig
from ..data.transforms import AugConfig, build_val_transform
from ..device import gpu_ids, resolve_device
from ..ema import ModelEMA
from ..model import DEFAULT_FPN_SCALES, DEFAULT_XVIEW, build_model, load_dinov2_backbone
from ..plotting import plot_aux_per_class, plot_metrics_csv
from ..postprocess import query_scores
from ..progress import progress
from ..serialization import load_raw, migrate_checkpoint, module_counts, resolve_checkpoint, save_checkpoint, wrap_checkpoint
from .distributed import is_main, launch, rank, sum_over_processes, world_size
from .validate import (
    _SEGM_KEYS,
    LOADER_TIMEOUT,
    attr_metric_names,
    aux_evaluate,
    evaluate,
    evaluate_detection,
    evaluate_sets,
    group_values,
    set_group_values,
    set_metric_keys,
)

# ---------------------------------------------------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """The architecture and loss weights of a new model. A warm start (``init_weights``) keeps its checkpoint's
    architecture (FPN, mask-head depth, attribute-head shape) and takes the loss weights from here; a resume or a
    multi-view post-training keeps everything of its checkpoint.

    ``fpn_scales``: the multi-scale pyramid read by the queries (``None`` = the original single-scale EoMT).
    ``num_upscale_blocks``: mask-head upsampling steps (``None`` = the size preset's, 2; 3 gives finer masks).
    The loss weights are those of the reference recipe; ``iou_aware_cls`` trains a matched query's class probability
    towards ``(1 + mask IoU) / 2`` so the score ranks masks by quality, and ``quality_weight > 0`` adds a head that
    predicts each mask's IoU (Mask Scoring R-CNN). ``l1_weight`` / ``giou_weight`` are the detect family's box terms.
    The attribute heads are MLPs of ``aux_head_layers`` layers (1 = linear) of width ``aux_head_hidden`` (``None`` = the
    encoder's) with ``aux_head_dropout``.
    """

    fpn_scales: tuple[float, ...] | None = DEFAULT_FPN_SCALES
    num_upscale_blocks: int | None = None
    no_object_weight: float = 0.1
    class_weight: float = 2.0
    mask_weight: float = 5.0
    dice_weight: float = 5.0
    iou_aware_cls: bool = True
    quality_weight: float = 2.0
    l1_weight: float = 5.0
    giou_weight: float = 2.0
    aux_head_layers: int = 2
    aux_head_hidden: int | None = None
    aux_head_dropout: float = 0.0

    def __post_init__(self) -> None:
        self.fpn_scales = tuple(float(s) for s in self.fpn_scales) if self.fpn_scales else None

    def loss_weights(self, family: str) -> dict:
        keys = ("no_object_weight", "class_weight") + (
            ("l1_weight", "giou_weight") if family == "detect"
            else ("mask_weight", "dice_weight", "iou_aware_cls", "quality_weight"))
        return normalize_loss_weights({k: getattr(self, k) for k in keys}, family=family)

    def aux_head_arch(self) -> dict:
        return {"layers": self.aux_head_layers, "hidden": self.aux_head_hidden, "dropout": self.aux_head_dropout}


@dataclass
class OptimConfig:
    """AdamW at ``lr0`` for the heads; the DINOv2 encoder at ``lr0 * backbone_lr_mult`` with layer-wise decay ``llrd``
    (the top block at that LR, each block below ``llrd`` times the one above), no weight decay on norms, biases and
    embeddings. The LR decays polynomially (``poly_power``) to 0 after a two-stage warmup in optimizer steps,
    ``warmup_steps = (head, vit)``: the head warms up while the encoder is held at 0, then the encoder ramps up.
    Gradients are clipped to ``clip_norm``. ``ema``: the validated and exported weights are an exponential moving
    average (``ema_decay``, ramped up over ``ema_tau`` steps). ``mask_anneal``: masked attention is annealed away block by
    block over the run, as in the reference recipe (the last sixth is mask-free); ``False`` trains mask-free from the
    first step, e.g. to fine-tune a model deployed mask-free. A short fine-tune wants a lower ``lr0``, a shorter
    warmup and a lower ``ema_decay``."""

    lr0: float = 1e-4
    weight_decay: float = 0.05
    llrd: float = 0.8
    backbone_lr_mult: float = 1.0
    poly_power: float = 0.9
    warmup_steps: tuple[int, int] = (500, 1000)
    clip_norm: float = 0.01
    ema: bool = True
    ema_decay: float = 0.9999
    ema_tau: float = 2000.0
    mask_anneal: bool = True

    def __post_init__(self) -> None:
        self.warmup_steps = tuple(int(x) for x in self.warmup_steps)
        if len(self.warmup_steps) != 2:
            raise ValueError("optim.warmup_steps must be (head_steps, vit_steps)")


@dataclass
class AuxConfig:
    """How the attribute heads train: on the queries the detector matched to an instance, kept when their mask IoU
    with it reaches ``iou_gate`` and (``class_gate``) their class is right. Each head's loss (cross-entropy, or Huber on
    a regression head's standardized value) is weighted by ``per_head`` (default 1) and the sum by ``w``.
    ``class_weights`` re-weights the categories of each categorical head by inverse-sqrt frequency."""

    w: float = 1.0
    per_head: dict[str, float] = field(default_factory=dict)
    class_weights: bool = False
    iou_gate: float = 0.5
    class_gate: bool = True


@dataclass
class ValConfig:
    """Validation every ``interval`` epochs, every epoch of the last ``final_epochs`` and after the last one, at
    ``imgsz`` (default: the training size), keeping
    every detection up to ``max_det`` per image scoring at least ``conf_thres`` (0: COCO mAP's convention).
    ``group_by`` (an image field, or several) adds the main metrics per value of that field. ``best_key`` is the
    ``metrics.csv`` validation column (without ``val/``) ``best.pt`` is chosen on, highest first (default: ``segm/mAP``,
    ``bbox/mAP`` for the detect family, ``sets/segm/mAP_set`` for a multi-view post-training, ``sets/match/f1`` for a
    matching head alone)."""

    interval: int = 2
    final_epochs: int = 0
    imgsz: int | None = None
    conf_thres: float = 0.0
    max_det: int = 100
    group_by: str | list[str] | None = None
    best_key: str | None = None


#: What trains in a multi-view post-training (``MultiviewConfig.trainable``): parameter group -> multiplier of
#: ``optim.lr0``. Every other parameter of the checkpoint stays frozen. "new": only the modules the post-training adds
#: (the cross-view sublayers, the set matcher and the LoRA factors), so a photo predicted alone gives the checkpoint's
#: result (without LoRA); "query_blocks": also the query blocks and the prediction heads, slower.
TRAINABLE_PRESETS: dict[str, dict[str, float]] = {
    "new": {"xview": 1.0, "matcher": 1.0, "lora": 1.0},
    "query_blocks": {"xview": 1.0, "matcher": 1.0, "lora": 1.0, "query_blocks": 0.4, "heads": 0.2},
}


@dataclass
class MultiviewConfig:
    """A post-training of an instance-segmentation checkpoint (``init_weights``) on sets of photos of one subject: the
    images' ``subject_id`` groups the photos, and the annotations' ``instance_id`` links the instances across them
    (:mod:`eomt.data.subjects`).

    ``xview``: cross-view attention, so that the photos of a set are predicted together, each photo's queries and
    (pooled) patches taking what the others show (a photo alone is predicted as before); ``attn_w`` weighs a loss that
    points the queries' cross-view attention at the same instance in the other photos (0: off). ``matching``: the set
    matcher (:class:`~eomt.matcher.SetMatcher`), which links the detections of the photos of a subject, trained on the
    ``instance_id`` links (``match_w`` weighs its loss), with or without ``xview``. ``gate`` (``"class"`` and
    categorical attribute heads) restricts the links to instances that agree on those, and the links' threshold is
    calibrated that way on the validation sets. ``lora``: ``{rank, blocks, alpha}`` adapts the attention of the
    ``blocks`` frozen ViT blocks below the query blocks (:mod:`eomt.lora`; ``None``: off).

    ``trainable``: what trains, a preset of :data:`TRAINABLE_PRESETS` or ``{group or parameter-name prefix: LR
    multiplier}`` (groups ``xview``, ``matcher``, ``lora``, ``query_blocks``, ``heads``; a prefix such as
    ``aux_heads.area`` wins over the groups, and 0 freezes it). ``sets``: how the sets are drawn
    (:class:`~eomt.data.subjects.SubjectSetConfig`; its ``max_views`` is the most photos predicted together).
    ``val_sets``: a file of fixed validation sets (every run writes its own as ``val_sets.json``), to score runs on the
    same sets.
    """

    xview: bool = True
    matching: bool = True
    match_w: float = 1.0
    attn_w: float = 0.0
    lora: dict | None = None
    gate: list[str] | None = None
    trainable: str | dict = "new"
    sets: SubjectSetConfig | dict = field(default_factory=SubjectSetConfig)
    val_sets: str | None = None

    def __post_init__(self) -> None:
        if not (self.xview or self.matching):
            raise ValueError("multiview: xview and matching are both off, nothing to train")
        if isinstance(self.sets, Mapping):
            self.sets = SubjectSetConfig(**self.sets)
        if isinstance(self.trainable, str) and self.trainable not in TRAINABLE_PRESETS:
            raise ValueError(f"multiview.trainable {self.trainable!r}: a preset of {sorted(TRAINABLE_PRESETS)} or a dict")
        if isinstance(self.gate, str):
            self.gate = [self.gate]

    def groups(self) -> dict[str, float]:
        """The LR multiplier of each trainable group or prefix, without the modules this post-training has not."""
        groups = dict(TRAINABLE_PRESETS[self.trainable]) if isinstance(self.trainable, str) else dict(self.trainable)
        return {g: m for g, m in groups.items() if (g != "xview" or self.xview) and (g != "matcher" or self.matching)
                and (g != "lora" or self.lora)}


def _section(cls, value, name: str):
    """``value`` (an instance, a mapping of its fields or ``None``) as a ``cls``; an unknown key stops the run."""
    if value is None:
        return cls()
    if isinstance(value, cls):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"{name}: a mapping of {cls.__name__} fields, got {type(value).__name__}")
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(value) - names
    if unknown:
        raise ValueError(f"{name}: unknown keys {sorted(unknown)} (valid: {', '.join(sorted(names))})")
    return cls(**value)


def _record(cfg):
    """A settings section as plain data (``args.yaml`` / the checkpoints' ``run_config``)."""
    if dataclasses.is_dataclass(cfg):
        return _record(dataclasses.asdict(cfg))
    if isinstance(cfg, Mapping):
        return {k: _record(v) for k, v in cfg.items()}
    if isinstance(cfg, (list, tuple)):
        return [_record(v) for v in cfg]
    return cfg


# ---------------------------------------------------------------------------------------------------------------------
# Optimizer and schedules
# ---------------------------------------------------------------------------------------------------------------------

_ENCODER_PREFIXES = ("eomt.embeddings", "eomt.layers", "eomt.layernorm")
# No weight decay on 1-D tensors (norms, biases) and token / positional embeddings: the usual ViT fine-tuning policy.
_NO_DECAY_TOKENS = ("position_embeddings", "cls_token", "register_tokens")


def _seed_everything(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _llrd_scale(name: str, num_layers: int, llrd: float) -> float:
    """Layer-wise LR decay factor of an encoder parameter, as in the reference EoMT recipe: block ``i`` scales by
    ``llrd ** (num_layers - 1 - i)`` (the top block at the base LR), the embeddings follow the first block and the
    final norm is at the base LR. The last blocks process the queries, so they must not be starved of LR."""
    if llrd >= 1.0 or num_layers <= 0:
        return 1.0
    if name.startswith("eomt.layers."):
        return llrd ** (num_layers - 1 - int(name.split("eomt.layers.")[1].split(".")[0]))
    if name.startswith("eomt.embeddings"):
        return llrd ** (num_layers - 1)
    return 1.0


def build_optimizer(model, lr: float, weight_decay: float, backbone_lr_mult: float, llrd: float = 1.0):
    """AdamW with layer-wise LR decay on the encoder, an optional encoder multiplier, and no weight decay on norms,
    biases and embeddings (see :class:`OptimConfig`). Each group is labelled ``is_backbone`` for the warmup."""
    num_layers = model.config.num_hidden_layers
    groups: dict[tuple, dict] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_backbone = name.startswith(_ENCODER_PREFIXES)
        lr_i = lr * backbone_lr_mult * _llrd_scale(name, num_layers, llrd) if is_backbone else lr
        wd_i = 0.0 if p.ndim <= 1 or any(tok in name for tok in _NO_DECAY_TOKENS) else weight_decay
        # ``is_backbone`` is part of the key: the top block and the final norm have the head's LR, but warm up with the
        # encoder (LR held at 0 during the head's warmup).
        key = (round(lr_i, 12), wd_i, is_backbone)
        groups.setdefault(key, {"params": [], "lr": lr_i, "weight_decay": wd_i, "is_backbone": is_backbone})["params"].append(p)
    opt = torch.optim.AdamW(list(groups.values()), lr=lr, weight_decay=weight_decay)
    for g in opt.param_groups:
        g["initial_lr"] = g["lr"]
    return opt


#: Parameter groups of a multi-view post-training -> name prefixes (the query blocks are added per model).
_MULTIVIEW_GROUPS = {
    "xview": ("eomt.xview.",),
    "matcher": ("matcher.",),
    "query_blocks": ("eomt.cross_blocks.", "eomt.query."),
    "heads": ("eomt.class_predictor.", "eomt.mask_head.", "eomt.upscale_block.", "eomt.quality_head.",
              "eomt.layernorm.", "aux_heads."),
}


def build_multiview_optimizer(model, lr: float, weight_decay: float, trainable: dict[str, float]):
    """Freeze every parameter outside the ``trainable`` groups and give each group ``lr`` times its multiplier (AdamW, no
    decay on 1-D tensors). A key is one of :data:`_MULTIVIEW_GROUPS` (the embeddings, the blocks before the queries and
    the FPN always stay frozen) or a parameter-name prefix, which wins over the groups (the longest prefix first):
    ``{"heads": 0.2, "aux_heads.area": 1.0}`` moves one head faster than the others, ``{"aux_heads.area": 0.0}`` freezes
    it. The query blocks warm up with the encoder (``is_backbone``), the rest with the heads."""
    n, nb = model.config.num_hidden_layers, model.config.num_blocks
    named = {**_MULTIVIEW_GROUPS, "query_blocks": _MULTIVIEW_GROUPS["query_blocks"]
             + tuple(f"eomt.layers.{i}." for i in range(n - nb, n)), "lora": ()}  # the LoRA factors: by name below
    names = [name for name, _ in model.named_parameters()]
    explicit = sorted((k for k in trainable if k not in named), key=len, reverse=True)
    unknown = [k for k in explicit if not any(nm == k or nm.startswith(k + ".") for nm in names)]
    if unknown:
        raise ValueError(f"multiview trainable {unknown}: neither a group ({sorted(named)}) nor a parameter-name prefix")

    def group_of(name: str):
        if ".parametrizations.weight.0.lora_" in name:  # a LoRA factor (eomt.lora)
            return "lora"
        hit = next((k for k in explicit if name == k or name.startswith(k + ".")), None)
        return hit if hit is not None else next((g for g, pre in named.items() if name.startswith(pre)), None)

    groups: dict[tuple, dict] = {}
    for name, p in model.named_parameters():
        g = group_of(name)
        on = g in trainable and float(trainable[g]) > 0
        p.requires_grad_(on)
        if not on:
            continue
        wd = 0.0 if p.ndim <= 1 else weight_decay
        groups.setdefault((g, wd), {"params": [], "lr": lr * float(trainable[g]), "weight_decay": wd,
                                    "is_backbone": name.startswith(named["query_blocks"])})["params"].append(p)
    if not groups:
        raise ValueError(f"multiview trainable {trainable}: no parameter of the model trains")
    opt = torch.optim.AdamW(list(groups.values()), lr=lr, weight_decay=weight_decay)
    for g in opt.param_groups:
        g["initial_lr"] = g["lr"]
    return opt


def _poly_two_stage_factors(opt_step, total_steps, warmup_steps, power):
    """``(head_factor, vit_factor)`` of the reference two-stage warmup + polynomial schedule: the head warms up linearly
    over ``warmup_steps[0]`` optimizer steps while the ViT LR is held at 0, the ViT then ramps up over
    ``warmup_steps[1]``; both then decay as ``(1 - progress) ** power`` to 0, each measured from the end of its warmup
    (``TwoStageWarmupPolySchedule`` of the official code)."""
    w_head, w_vit = int(warmup_steps[0]), int(warmup_steps[1])

    def poly(adjusted, span):
        return (1.0 - min(1.0, adjusted / max(1, span))) ** power

    head = opt_step / w_head if w_head > 0 and opt_step < w_head else poly(max(0, opt_step - w_head), total_steps - w_head)
    if opt_step < w_head:
        vit = 0.0
    elif w_vit > 0 and opt_step < w_head + w_vit:
        vit = (opt_step - w_head) / w_vit
    else:
        vit = poly(max(0, opt_step - w_head - w_vit), total_steps - w_head - w_vit)
    return head, vit


#: Per-block masked-attention annealing windows (fractions of training) of the reference recipe: the four query blocks
#: are annealed one after the other, and the last sixth of training is mask-free.
EOMT_MASK_ANNEAL_START = (1 / 6, 2 / 6, 3 / 6, 4 / 6)
EOMT_MASK_ANNEAL_END = (2 / 6, 3 / 6, 4 / 6, 5 / 6)


def _attn_mask_probs(it: int, total_iters: int, power: float = 0.9) -> list[float]:
    """Masked-attention probability of every query block at iteration ``it``: 1 until the start of the block's window,
    ``(1 - progress) ** power`` inside it, 0 after it."""
    out = []
    for a, b in zip(EOMT_MASK_ANNEAL_START, EOMT_MASK_ANNEAL_END):
        start, end = a * total_iters, b * total_iters
        out.append(1.0 if it < start else 0.0 if it >= end else (1.0 - (it - start) / max(1.0, end - start)) ** power)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Attribute-head helpers
# ---------------------------------------------------------------------------------------------------------------------


def _in_scope_labels(train_ds, spec) -> list:
    """Labels (ignored ones included) of the train annotations a head applies to (no image decoded)."""
    out = []
    for a in getattr(train_ds.coco, "dataset", {}).get("annotations", []) or []:
        cls = train_ds.cat2contig.get(a.get("category_id"))
        if cls is None or (spec.applies_to is not None and cls not in spec.applies_to):
            continue
        out.append(_attr_label(spec, train_ds._attr_id_maps, a.get("attributes", {}), cls))
    return out


def _valid_labels(spec, labels: list) -> list:
    return [v for v in labels if (v == v if spec.regression else v >= 0)]  # NaN != NaN


def _class_weights(train_ds) -> dict:
    """Inverse-sqrt-frequency CE weights per categorical head, mean-normalized to 1."""
    out: dict = {}
    for spec in train_ds.aux_specs:
        if spec.regression:
            continue
        counts = torch.zeros(spec.num_classes)
        for cid in _valid_labels(spec, _in_scope_labels(train_ds, spec)):
            counts[cid] += 1
        w = 1.0 / counts.clamp(min=1).sqrt()
        out[spec.name] = w * (spec.num_classes / w.sum())
    return out


def _head_banner(spec, names: dict, train_ds) -> str:
    """One head in the training banner: its kind, scope and how many train instances carry a value."""
    labels = _in_scope_labels(train_ds, spec)
    if spec.regression:
        crop = f", crop {spec.crop}" + (f" < {spec.min_visible:g}" if spec.crop == "ignore" else "") if spec.crop else ""
        kind = f"regression{', log' if spec.log else ''}{crop}, mean {spec.mean:.4g} std {spec.std:.4g}"
    else:
        pairs = [f"{spec.names.get(a, a)}<->{spec.names.get(b, b)}" for a, b in (spec.flip_aware or {}).items() if a < b]
        kind = str(spec.num_classes) + (f", flips {' '.join(pairs)}" if pairs else "")
    scope = ""
    if spec.applies_to is not None:
        scope = (f" on {len(spec.applies_to)} classes" if len(spec.applies_to) > 6
                 else " on {" + ",".join(sorted(names.get(c, str(c)) for c in spec.applies_to)) + "}")
    return f"{spec.name}({kind}){scope}: {len(_valid_labels(spec, labels))}/{len(labels)} tagged"


def _head_record(spec, class_names: dict) -> dict:
    """How one attribute head learns, as the run records it, in names rather than contiguous ids."""
    if spec.regression:
        rec = {"type": "regression", "log": spec.log, "range": list(spec.value_range) if spec.value_range else None,
               "crop": spec.crop, "min_visible": spec.min_visible if spec.crop == "ignore" else None,
               "bins": list(spec.bins) if spec.bins else None,
               "mean": round(float(spec.mean), 6), "std": round(float(spec.std), 6)}
    else:
        rec = {"type": "categorical", "categories": [spec.names.get(i, str(i)) for i in range(spec.num_classes)],
               "flip_aware": {spec.names.get(a, a): spec.names.get(b, b) for a, b in spec.flip_aware.items() if a < b}
               if spec.flip_aware else None}
    if spec.applies_to is not None:
        rec["applies_to"] = sorted(class_names.get(c, str(c)) for c in spec.applies_to)
    return {k: v for k, v in rec.items() if v is not None}


def _aux_metric(spec) -> str:
    """The matched-query metric of a head: ``aux_acc`` (accuracy), or ``aux_mae`` (mean absolute error, real units)."""
    return "aux_mae" if spec.regression else "aux_acc"


def _adopt_checkpoint_aux(ds, ckpt: dict) -> None:
    """Keep a post-trained checkpoint's attribute heads as they were trained: its standardization of the regression
    values, its flip-aware swaps and scopes, over the dataset's vocabularies (which must be the same)."""
    saved = {s.name: s for s in aux_specs_from_meta(ckpt.get("aux_heads"))}
    missing = set(saved) - {s.name for s in ds.aux_specs}
    if missing:  # the post-trained checkpoint would silently lose them
        raise ValueError(f"the dataset does not define the checkpoint's attribute heads {sorted(missing)}")
    for s in ds.aux_specs:
        ck = saved[s.name]
        if not s.regression and s.names != ck.names:
            raise ValueError(f"attribute {s.name!r}: the dataset's categories {s.names} differ from the checkpoint's {ck.names}")
        s.mean, s.std, s.value_range, s.flip_aware = ck.mean, ck.std, ck.value_range, ck.flip_aware
        s.log, s.applies_to = ck.log, ck.applies_to
        s.bins = s.bins or ck.bins  # reporting only: the run config's when it sets them


@torch.no_grad()
def _extra_negatives(model, out: dict, indices, extras: dict | None, conf: float) -> torch.Tensor:
    """``(B, Q)`` bool: the queries no target matched that score at least ``conf`` (as postprocess scores them), in a
    fully annotated photo and outside its ignore regions: the false detections and duplicates that make false links at
    inference, negatives of the matching loss."""
    cls, masks, quality = out["class_queries_logits"], out["masks_queries_logits"], out.get("quality_logits")
    unmatched = torch.ones(cls.shape[:2], dtype=torch.bool, device=cls.device)
    for b, (src, _) in enumerate(indices):
        unmatched[b, src.to(cls.device)] = False
    scores = torch.stack([query_scores(cls[b], masks[b], None if quality is None else quality[b])[0]
                          for b in range(cls.shape[0])])
    if extras is not None:
        unmatched &= model.eomt.criterion.negative_weights(masks, extras["ignore"], extras["neg_weight"]) >= 1.0
    return unmatched & (scores >= conf)


# ---------------------------------------------------------------------------------------------------------------------
# Validation of a run
# ---------------------------------------------------------------------------------------------------------------------


def build_val_parts(*, val_images, val_json, imgsz: int, aug_cfg, aux, is_detect: bool = False,
                    sets: SubjectSetConfig | None = None, fixed=None) -> dict:
    """The validation data of a run: ``photos`` (the val images one by one, COCO mAP) with ``photos_aux`` (a
    deterministic view of them for the matched-query attribute metrics, when there are heads) and, for a multi-view
    post-training (``sets``), ``sets``: fixed sets of the val photos of each subject, unaugmented (``fixed`` file names,
    else drawn with seed 0). ``aux`` is the ``(specs, id maps)`` the attribute labels are read with."""
    parts: dict = {}
    if not (val_images and val_json):
        return parts
    parts["photos"] = CocoValImages(val_images, val_json, imgsz=imgsz)
    if aux[0]:
        parts["photos_aux"] = (CocoDetection if is_detect else CocoInstanceSeg)(
            val_images, val_json, imgsz=imgsz, transform=build_val_transform(imgsz), shared_aux=aux)
    if sets is not None:
        parts["sets"] = CocoSubjectSets(val_images, val_json, imgsz=imgsz, aug=aug_cfg, seed=0, augment=False,
                                        sets={**dataclasses.asdict(sets), "p_single": 0.0, "p_hard": 0.0, "p_random": 0.0},
                                        shared_aux=aux, fixed=fixed)
    return parts


def val_metric_keys(parts: dict, specs, *, is_detect: bool = False, link_group_by=None, group_by=None) -> list[str]:
    """The keys :func:`validate_parts` returns for ``parts`` (the ``metrics.csv`` validation columns, without ``val/``)."""
    keys: list[str] = []
    if "photos" in parts:
        ds = parts["photos"]
        keys += [] if is_detect else [f"segm/{k}" for k in _SEGM_KEYS]
        keys += [f"bbox/{k}" for k in _SEGM_KEYS]
        if "photos_aux" in parts:
            keys += [n for s in specs for n in attr_metric_names(s)]
        for f, values in group_values(ds.coco, ds.ids, group_by).items():
            keys += [f"{f}={v}/{t}/mAP" for v in values for t in (("bbox",) if is_detect else ("segm", "bbox"))]
    if "sets" in parts:
        keys += set_metric_keys("sets", specs, gated=bool(link_group_by), groups=set_group_values(parts["sets"], group_by))
    return keys


def _fmt(x, nd: int = 3) -> str:
    return "-" if x is None or x != x else f"{x:.{nd}f}"


def validate_parts(model, parts: dict, *, device, batch: int = 4, workers: int = 4, conf_thres: float = 0.0,
                   max_det: int = 100, amp: bool = True, verbose: bool = True, is_detect: bool = False,
                   aux_iou_gate: float = 0.5, aux_class_gate: bool = True, link_group_by=None, link_conf: float = 0.3,
                   thres: float | None = None, group_by=None, tag: str = "[val]") -> tuple[dict, dict | None]:
    """Validate ``model`` on the ``parts`` of :func:`build_val_parts` as the trainer does at the end of an epoch:
    ``(metrics, per-class attribute counts)``, the metrics keyed as in ``metrics.csv`` without ``val/``
    (:func:`val_metric_keys`). The model is run as given (the trainer sets its masked-attention regime). ``thres`` is
    the stored link threshold the links are also scored at; ``link_group_by``, ``link_conf`` and ``group_by`` as in
    :func:`~eomt.engine.validate.evaluate_sets`. On several GPUs every process runs its share and the main process gets
    the metrics."""
    main = is_main()
    metrics: dict = {}
    aux_pc = None
    if "photos" in parts:
        fn = evaluate_detection if is_detect else evaluate
        metrics.update(fn(model, parts["photos"], device=device, batch_size=batch, num_workers=workers,
                          conf_thres=conf_thres, max_det=max_det, amp=amp, verbose=verbose and main, group_by=group_by))
        if main:
            t = "bbox" if is_detect else "segm"
            LOG.info(f"{tag} {t} mAP {metrics.get(f'{t}/mAP', 0):.4f} mAP50 {metrics.get(f'{t}/mAP50', 0):.4f}"
                  + ("" if is_detect else f" bbox mAP {metrics.get('bbox/mAP', 0):.4f}"))
        if "photos_aux" in parts:  # held-out matched-query accuracy per head
            val_aux, aux_pc = aux_evaluate(model, parts["photos_aux"], device=device, batch_size=batch,
                                           num_workers=workers, amp=amp, verbose=verbose and main,
                                           iou_gate=aux_iou_gate, class_gate=aux_class_gate)
            metrics.update(val_aux)
            if main:
                LOG.info(f"{tag} val aux: " + "  ".join(
                    f"{k.split('/', 1)[1]} {k.split('/', 1)[0][4:]}={v:.3f}" for k, v in val_aux.items()
                    if k.startswith(("aux_acc/", "aux_mae/")) and "@" not in k))
    if "sets" in parts:
        metrics.update(evaluate_sets(model, parts["sets"], device=device, num_workers=workers, amp=amp,
                                     verbose=verbose and main, link_group_by=link_group_by, link_conf=link_conf,
                                     thres=thres, group_by=group_by))
        if main:
            g = "_gated" if link_group_by else ""
            m = lambda k: metrics.get(f"sets/{k}", float("nan"))  # noqa: E731
            LOG.info(f"{tag} val sets: segm mAP {_fmt(m('segm/mAP_alone'), 4)} alone -> {_fmt(m('segm/mAP_set'), 4)} with "
                  f"the set (linked instances {_fmt(m('segm/mAP_linked_alone'), 4)} -> {_fmt(m('segm/mAP_linked_set'), 4)});"
                  f" links F1 {_fmt(m(f'match{g}/f1'))} at {_fmt(m(f'match{g}/thres'), 2)} (re-ID of the photos alone "
                  f"{_fmt(m(f'reid{g}/f1'))}), at the stored threshold {_fmt(m('match/f1_stored'))}; identities per "
                  f"subject (B-cubed F1) {_fmt(m('match/cluster_f1'))}")
    return metrics, aux_pc


# ---------------------------------------------------------------------------------------------------------------------
# Run bookkeeping
# ---------------------------------------------------------------------------------------------------------------------


def _sha256(path) -> str | None:
    p = Path(path) if path else None
    if p is None or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _same_categories(a, b, what: str) -> None:
    """Raise unless datasets ``a`` and ``b`` name the same primary classes in the same order."""
    na = a.names if isinstance(a.names, dict) else dict(enumerate(a.names))
    nb = b.names if isinstance(b.names, dict) else dict(enumerate(b.names))
    if na != nb or a.cat2contig != b.cat2contig:
        raise ValueError(f"{what}: the categories differ from the train data's ({nb} vs {na})")


class _CsvLogger:
    """Per-epoch metrics in ``<run_dir>/metrics.csv``, with fixed columns (a value an epoch lacks is ``nan``). A resumed
    run keeps the prior rows, rewritten under the new header when the columns changed."""

    def __init__(self, path: Path, fields: list[str]):
        self.path = path
        self.fields = fields
        if not path.exists() or path.stat().st_size == 0:
            with path.open("w", newline="") as f:
                csv.writer(f).writerow(fields)
            return
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            old, rows = reader.fieldnames or [], list(reader)
        if old != fields:
            with path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(fields)
                w.writerows([[r.get(k) if r.get(k) not in (None, "") else float("nan") for k in fields] for r in rows])

    def append(self, row: dict):
        with self.path.open("a", newline="") as f:
            csv.writer(f).writerow([row.get(k, float("nan")) for k in self.fields])


class _Forward(torch.nn.Module):
    """``fn(model, *args)`` as a module's forward, so that DistributedDataParallel sees every parameter a training step
    uses: the attribute, matcher and attention losses run on the model's outputs, outside ``model.forward``."""

    def __init__(self, model, fn):
        super().__init__()
        self.model, self.fn = model, fn

    def forward(self, *args):
        return self.fn(self.model, *args)


class _Logger:
    """Optional TensorBoard / Weights & Biases scalar logger."""

    def __init__(self, kind: str, logdir: Path, run_name: str):
        self.kind = kind
        self.writer = None
        if kind == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=str(logdir))
        elif kind == "wandb":
            import wandb

            self.writer = wandb
            wandb.init(project="attr-eomt", name=run_name, dir=str(logdir))
        elif kind != "none":
            raise ValueError(f"logger {kind!r}: 'none', 'tensorboard' or 'wandb'")

    def log(self, scalars: dict, step: int):
        if self.kind == "tensorboard":
            for k, v in scalars.items():
                self.writer.add_scalar(k, v, step)
        elif self.kind == "wandb":
            self.writer.log(scalars, step=step)

    def close(self):
        if self.kind == "tensorboard":
            self.writer.close()
        elif self.kind == "wandb":
            self.writer.finish()


#: Micro-steps between two refreshes of the progress bar (each reads the loss off the GPU, a synchronisation).
LOG_EVERY = 10


@contextlib.contextmanager
def _inference_numerics():
    """PyTorch's default numerics (no TF32 matmuls, no cuDNN autotuning), which training turns on for speed: validated
    in them, the trainer's metrics are exactly those a standalone validation, or the deployed model, reproduces."""
    b = torch.backends
    saved = (b.cuda.matmul.allow_tf32, b.cudnn.allow_tf32, b.cudnn.benchmark)
    b.cuda.matmul.allow_tf32, b.cudnn.allow_tf32, b.cudnn.benchmark = False, True, False
    try:
        yield
    finally:
        b.cuda.matmul.allow_tf32, b.cudnn.allow_tf32, b.cudnn.benchmark = saved


# ---------------------------------------------------------------------------------------------------------------------
# train()
# ---------------------------------------------------------------------------------------------------------------------


def train(
    *,
    train_images: str,
    train_json: str,
    val_images: str | None = None,
    val_json: str | None = None,
    size: str = "l",
    family: str = "instance",
    imgsz: int = DEFAULT_IMAGE_SIZE,
    epochs: int = 12,
    batch: int = 4,
    nominal_batch: int = 16,
    device: str = "auto",
    workers: int = 8,
    seed: int | None = None,
    amp: bool = True,
    project: str = "runs/train",
    name: str | None = None,
    resume: str | None = None,
    init_weights: str | None = None,
    pretrained: bool = True,
    keep_empty: bool = False,
    stop_after_epochs: int | None = None,
    logger: str = "none",
    model: ModelConfig | dict | None = None,
    optim: OptimConfig | dict | None = None,
    aux: AuxConfig | dict | None = None,
    aug: AugConfig | dict | None = None,
    val: ValConfig | dict | None = None,
    multiview: MultiviewConfig | dict | None = None,
    attributes: dict | None = None,
    inference: dict | None = None,
    train_transform=None,
) -> dict:
    """Train an EoMT model on a COCO-format dataset (the module docstring lists the settings sections).

    ``size`` (``"s"``, ``"b"``, ``"l"``) and ``family`` (``"instance"`` masks, ``"detect"`` boxes) define a new model,
    initialised from DINOv2 when ``pretrained``. ``init_weights`` warm-starts from a checkpoint instead (weights only, a
    new run from epoch 0); ``resume`` continues a run from its ``last.pt`` exactly (epoch, optimizer, schedule, EMA,
    best). The two exclude each other, and both keep their checkpoint's size, family and input size.

    ``epochs`` sets the schedule: the LR decay and the masked-attention annealing span it (12 is the official COCO
    instance recipe); ``stop_after_epochs`` ends the run earlier on the same schedule (truncated A/B runs). ``batch`` is
    the micro-batch per GPU and ``nominal_batch`` the global effective batch, the rest accumulated, so one GPU and
    several follow the same optimizer steps. ``workers`` are data-loader workers per GPU. ``amp`` trains in bf16 where
    the GPU has it (else fp16 with a gradient scaler). ``keep_empty`` keeps the images without annotations as negatives
    (instance family). ``train_transform`` replaces the built-in augmentation with your own ``(image, masks) -> (image,
    masks)`` callable. ``logger``: ``"tensorboard"`` or ``"wandb"`` (needs the ``logging`` extra).

    The run folder (``project/name``, default ``eomt-<size>``) receives ``args.yaml`` (the resolved run), ``metrics.csv``
    and ``metrics.png``, and ``weights/``: ``last.pt`` (the trainer state, written before each validation, so a crash
    there costs only the validation) and ``best.pt`` (the EMA weights, mask-free, of the best validation on
    ``val.best_key``; only once the masked-attention annealing is over, so it runs as predict runs it). Every checkpoint
    records its epoch, metrics row and the resolved run, with the warm-start weights and the data files and their
    SHA-256 (``run_config.provenance``).
    """
    gpus = gpu_ids(device)
    if len(gpus) > 1 and not dist.is_initialized():  # one process per GPU, each running this function on its own
        return launch(train, gpus, {k: v for k, v in locals().items() if k != "gpus"})
    mcfg, ocfg = _section(ModelConfig, model, "model"), _section(OptimConfig, optim, "optim")
    acfg, vcfg = _section(AuxConfig, aux, "aux"), _section(ValConfig, val, "val")
    mv = None if multiview is None else _section(MultiviewConfig, multiview, "multiview")
    aug_cfg = AugConfig.resolve(aug)
    if imgsz % PATCH_SIZE or (vcfg.imgsz and vcfg.imgsz % PATCH_SIZE):
        raise ValueError(f"imgsz={imgsz} and val.imgsz={vcfg.imgsz} must be divisible by {PATCH_SIZE} (DINOv2 grid)")
    if inference and set(inference) - set(INFERENCE_DEFAULTS):
        raise ValueError(f"inference: unknown keys {sorted(set(inference) - set(INFERENCE_DEFAULTS))} "
                         f"(allowed: {', '.join(INFERENCE_DEFAULTS)})")
    if resume and init_weights:
        raise ValueError("pass either resume (continue a run) or init_weights (warm-start a new run), not both")
    world, main = world_size(), is_main()
    if seed is not None:
        _seed_everything(seed + rank())  # DDP copies process 0's initial weights to the others
    elif world > 1:
        torch.seed()  # torch's default seed is the same in every process: draw distinct augmentations
    dev = resolve_device(device)
    if dev.type == "cuda":  # TF32 matmuls and the cuDNN autotuner: free speed, the input is a fixed square
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    accum_steps = max(1, round(nominal_batch / (batch * world)))
    eff_batch = batch * accum_steps * world
    LOG.info(f"[batch] micro={batch} x accum={accum_steps}" + (f" x {world} GPUs" if world > 1 else "")
          + f" -> effective {eff_batch}")

    # --- the checkpoint the run starts from: its size, family and input size, and (resume, multi-view) architecture
    resume_ckpt = init_ckpt = resume_path = init_path = None
    if resume:
        resume_path = resolve_checkpoint(resume, prefer="last")
        resume_ckpt = migrate_checkpoint(load_raw(resume_path))
        if "optimizer" not in resume_ckpt:
            raise ValueError("resume continues a run from its last.pt (it holds the trainer state); to fine-tune from a "
                             "best.pt pass it as init_weights")
    if init_weights:
        init_path = resolve_checkpoint(init_weights, prefer="best")
        init_ckpt = migrate_checkpoint(load_raw(init_path))
    src = resume_ckpt if resume_ckpt is not None else init_ckpt
    if src is not None:
        size, family, imgsz = src["size"], src["task"], int(src["imgsz"])
        LOG.info(f"[{'resume' if resume else 'init'}] {resume_path or init_path} (size={size}, family={family}, "
              f"imgsz={imgsz})" + ("" if resume else ": weights only, a new run from epoch 0"))
    is_detect = family == "detect"
    name = name or f"eomt-{size}"
    run_dir = Path(project) / name
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # --- architecture: the model section for a new model; a checkpoint's own for a resume or a multi-view post-training;
    # a warm start keeps its checkpoint's architecture and takes the loss weights of the model section
    loss_weights = mcfg.loss_weights(family)
    aux_head_arch, fpn_scales, num_upscale_blocks = mcfg.aux_head_arch(), mcfg.fpn_scales, mcfg.num_upscale_blocks
    if src is not None:
        aux_head_arch = src.get("aux_head_arch") or aux_head_arch
        fpn_scales, num_upscale_blocks = src.get("fpn_scales"), src.get("num_upscale_blocks")
        if resume or mv is not None:
            loss_weights = normalize_loss_weights(src.get("loss_weights"), family=family)
    mask_anneal = ocfg.mask_anneal and not is_detect and mv is None
    if mv is not None:
        if src is None or is_detect:
            raise ValueError("multiview post-trains an instance-segmentation checkpoint: pass init_weights (or resume)")
        if train_transform is not None:
            raise ValueError("multiview uses the built-in augmentation: train_transform is not supported")
        if inference is None:  # the predict defaults the checkpoint was saved with
            inference = src.get("inference")
    # the detections whose links are trained against and scored: those inference keeps
    link_conf = float((inference or {}).get("conf_thres", INFERENCE_DEFAULTS["conf_thres"]))

    # --- data
    if train_transform is not None:
        LOG.info("[aug] a custom train_transform: the aug settings are ignored")
    else:
        LOG.info(f"[aug] active: {', '.join(aug_cfg.active())}; LSJ {aug_cfg.min_scale}-{aug_cfg.max_scale}")
    if keep_empty and is_detect:
        raise ValueError("keep_empty applies to the instance family only")
    if mv is not None:  # sets of photos of one subject, with the checkpoint's attribute heads
        ck_heads = [h["name"] for h in src.get("aux_heads") or []]
        train_ds = CocoSubjectSets(train_images, train_json, imgsz=imgsz, aug=aug_cfg, sets=mv.sets,
                                   attributes={n: attributes.get(n) for n in ck_heads} if attributes else ck_heads)
        _adopt_checkpoint_aux(train_ds, src)
        from ..reid import validate_group_by

        mv.gate = list(validate_group_by(mv.gate, train_ds.aux_specs)) or None
        LOG.info(f"[data] train: {len(train_ds)} sets of up to {mv.sets.max_views} photos, {train_ds.num_subjects} "
              f"subjects; {mv.sets}")
    else:
        train_ds = (CocoDetection if is_detect else CocoInstanceSeg)(
            train_images, train_json, imgsz=imgsz, transform=train_transform, aug=aug_cfg,
            attributes=True if attributes is None else attributes, **({"keep_empty": True} if keep_empty else {}))
        n_neg = sum(1 for i in train_ds.ids if not train_ds.coco.getAnnIds(imgIds=i, iscrowd=False)) if keep_empty else 0
        LOG.info(f"[data] train: {len(train_ds)} images" + (f" (incl. {n_neg} negatives)" if keep_empty else ""))
    nc, names, aux_specs = train_ds.num_classes, train_ds.names, train_ds.aux_specs
    LOG.info(f"[data] {nc} classes" + ("; attribute heads: " + "; ".join(_head_banner(s, names, train_ds) for s in aux_specs)
                                    if aux_specs else ""))
    sampler = DistributedSampler(train_ds, shuffle=True, seed=seed or 0, drop_last=True) if world > 1 else None
    train_loader = DataLoader(
        train_ds, batch_size=batch, shuffle=sampler is None, sampler=sampler, num_workers=workers,
        collate_fn=collate_sets if mv is not None else collate_train, pin_memory=dev.type == "cuda", drop_last=True,
        persistent_workers=workers > 0, prefetch_factor=4 if workers > 0 else None,
        timeout=LOADER_TIMEOUT if workers > 0 else 0,
    )
    val_size = vcfg.imgsz or imgsz
    fixed = CocoSubjectSets.read_sets(mv.val_sets) if mv is not None and mv.val_sets else None
    val_parts = build_val_parts(val_images=val_images, val_json=val_json, imgsz=val_size, aug_cfg=aug_cfg,
                                aux=(aux_specs, train_ds._attr_id_maps), is_detect=is_detect,
                                sets=mv.sets if mv is not None else None, fixed=fixed)
    if "photos" in val_parts:
        LOG.info(f"[data] val: {len(val_parts['photos'])} images" + (f" at {val_size} px" if val_size != imgsz else ""))
    if "sets" in val_parts:
        vs = val_parts["sets"]
        _same_categories(train_ds, vs, "val sets")
        got = f" from {mv.val_sets} ({vs.fixed_report})" if fixed is not None else ""
        LOG.info(f"[data] val sets: {len(vs)} fixed sets of {vs.num_subjects} subjects{got}")
        if main:  # the sets, so that another run or model can be scored on the same ones (multiview.val_sets)
            vs.write_sets(run_dir / "val_sets.json")
    aux_class_weight = _class_weights(train_ds) if aux_specs and acfg.class_weights else None

    # --- the cross-view sublayers and the set matcher of a multi-view post-training
    xview_cfg = matcher_cfg = None
    if mv is not None:
        if mv.xview:
            xview_cfg = {**DEFAULT_XVIEW, **(src.get("xview") or {}), "max_views": mv.sets.max_views}
        if mv.matching:  # a checkpoint's matcher as it is, else the default one
            matcher_cfg = {**DEFAULT_MATCHER, **(src.get("matcher") or {})}

    # --- the run, as args.yaml and every checkpoint's run_config record it
    run_cfg = {
        "size": size, "family": family, "imgsz": imgsz, "nc": nc,
        "train_images": train_images, "train_json": train_json, "val_images": val_images, "val_json": val_json,
        "epochs": epochs, "stop_after_epochs": stop_after_epochs, "gpus": world, "batch": batch, "accum": accum_steps,
        "effective_batch": eff_batch, "workers": workers, "seed": seed, "amp": amp, "pretrained": pretrained,
        "keep_empty": keep_empty,
        "model": {**_record(mcfg), "fpn_scales": list(fpn_scales) if fpn_scales else None,
                  "num_upscale_blocks": num_upscale_blocks, "aux_head_arch": aux_head_arch, "loss_weights": loss_weights},
        "optim": {**_record(ocfg), "mask_anneal": mask_anneal},
        "aux": _record(acfg),
        "aug": aug_cfg.to_dict() if train_transform is None else "custom train_transform",
        "val": {**_record(vcfg), "imgsz": val_size},
        "multiview": _record(mv) if mv is not None else None,
        "attributes": {s.name: _head_record(s, names) for s in aux_specs},
        "inference": inference,
    }
    if main:  # where the run started from and what it read: the files and their SHA-256
        init_src = resume_path or init_path
        run_cfg["provenance"] = {
            "init_weights": str(Path(init_src).resolve()) if init_src else None,
            "init_sha256": _sha256(init_src) if init_src else None,
            "data": {k: {"path": str(Path(v).resolve()), "sha256": _sha256(v)}
                     for k, v in (("train_json", train_json), ("val_json", val_json),
                                  ("val_sets", mv.val_sets if mv is not None else None)) if v},
        }
        import yaml

        with (run_dir / "args.yaml").open("w") as f:
            yaml.safe_dump(run_cfg, f, sort_keys=False)

    # --- model
    net = build_model(size, nc=nc, imgsz=imgsz, names=names, family=family, aux_heads=aux_specs,
                      aux_head_arch=aux_head_arch, loss_weights=loss_weights, num_upscale_blocks=num_upscale_blocks,
                      fpn_scales=fpn_scales, xview=xview_cfg, matcher=matcher_cfg).to(dev)
    LOG.info(f"[model] loss weights: {loss_weights}; upscale blocks {net.num_upscale_blocks}"
          + (f"; FPN {net.fpn_scales}" if net.fpn_scales else "")
          + (f"; attribute heads {net.aux_head_arch}" if aux_specs else ""))
    start_epoch, best_metric = 0, float("-inf")
    lora_cfg = None
    if resume_ckpt is None and init_ckpt is not None:
        # Weights only; a tensor whose shape differs (another number of classes, a new head) keeps its fresh init.
        cur = net.state_dict()
        ckpt_sd = init_ckpt["model"]
        if has_lora(ckpt_sd):  # an exported training state: its LoRA factors merged
            lora = init_ckpt.get("lora") or {}
            ckpt_sd = merge_state(ckpt_sd, scale=float(lora.get("alpha", 1)) / float(lora.get("rank", 1)))
        filtered = {k: v for k, v in ckpt_sd.items() if k in cur and cur[k].shape == v.shape}
        skipped = [k for k, v in ckpt_sd.items() if k in cur and cur[k].shape != v.shape]
        missing, unexpected = net.load_state_dict(filtered, strict=False)
        if skipped:
            LOG.info(f"[init] re-initialized {len(skipped)} shape-mismatched tensors: {module_counts(skipped)}")
        if missing:
            LOG.info(f"[init] fresh (not in the checkpoint): {module_counts(missing)}")
        if unexpected:
            LOG.info(f"[init] in the checkpoint, not in the model: {module_counts(unexpected)}")
        if net.eomt.xview is not None and not any(k.startswith("eomt.xview.") for k in ckpt_sd):
            net.init_xview()  # each cross-view sublayer starts as a copy of its query block's attention
    elif resume_ckpt is None and pretrained:
        load_dinov2_backbone(net)
        net.to(dev)
    if mv is not None and mv.lora:  # after the weights (a resumed run's come with its factors): they start at 0
        lora_cfg = add_lora(net, **mv.lora)

    def make_optimizer():
        if mv is not None:
            return build_multiview_optimizer(net, ocfg.lr0, ocfg.weight_decay, mv.groups())
        return build_optimizer(net, ocfg.lr0, ocfg.weight_decay, ocfg.backbone_lr_mult, ocfg.llrd)

    optimizer = make_optimizer()
    if mv is not None:
        n_train = sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6
        LOG.info(f"[multiview] {'cross-view attention' if net.eomt.xview is not None else 'no cross-view attention'}, "
              + (f"set matcher {net.matcher_cfg}" if net.matcher is not None else "no set matcher")
              + (f", LoRA {lora_cfg} on blocks {sorted({int(n.split('.')[2]) for n, _ in net.named_parameters() if '.lora_' in n})}"
                 if lora_cfg else "")
              + f"; training {mv.groups()} ({n_train:.1f} M parameters), the rest of the checkpoint is frozen")
    if resume_ckpt is not None:
        net.load_state_dict(resume_ckpt["model"])
        optimizer.load_state_dict(resume_ckpt["optimizer"])
        start_epoch = int(resume_ckpt.get("epoch", -1)) + 1
        # best.pt may be newer than last.pt (written after it, before a crash): never overwrite it with a worse one
        best_path = weights_dir / "best.pt"
        best_metric = max(float(resume_ckpt.get("best_metric", float("-inf"))),
                          float(load_raw(best_path).get("best_metric", float("-inf"))) if best_path.is_file()
                          else float("-inf"))
        LOG.info(f"[resume] at epoch {start_epoch} (best {best_metric:.4f})")
    cross_view = net.eomt.xview is not None
    geom_key = "box_labels" if is_detect else "mask_labels"

    def micro_step(m, pixel_values, geom_labels, class_labels, aux_labels, set_ids=None, inst_ids=None, extras=None):
        """One micro-batch: the model's loss (detection, in the forward) plus the matching and attribute losses on the
        final prediction's matching, gated. Returns ``(loss, stats)``, the stats as device tensors."""
        partial = {} if extras is None else {"ignore_masks": extras["ignore"], "neg_weight": extras["neg_weight"]}
        attn_w = mv.attn_w if mv is not None and cross_view else 0.0
        out = m(pixel_values, class_labels=class_labels, set_ids=set_ids if cross_view else None,
                keep_xview_attn=attn_w > 0, **{geom_key: geom_labels}, **partial)
        loss, stats = out["loss"], {}
        gated = None
        matching = m.matcher is not None and set_ids is not None
        if aux_specs or matching or attn_w > 0:
            gated = gate_indices(out, out["indices"], geom_labels, class_labels, iou_thr=acfg.iou_gate,
                                 require_class=acfg.class_gate)
        if matching:
            negatives = _extra_negatives(m, out, out["indices"], extras, link_conf)
            mstats: dict = {}
            m_loss = set_match_loss(m, out, gated, inst_ids, set_ids, negatives=negatives, stats=mstats)
            loss = loss + mv.match_w * m_loss
            stats["match"] = (m_loss.detach(), mstats)
        if attn_w > 0:
            grid = (pixel_values.shape[-2] // m.patch_size, pixel_values.shape[-1] // m.patch_size)
            x_loss = xview_attention_loss(m, out, gated, inst_ids, geom_labels, grid)
            loss = loss + attn_w * x_loss
            stats["attn"] = x_loss.detach()
        if aux_specs:
            astats: dict = {}
            a_loss, _ = aux_loss(m, out, geom_labels, class_labels, aux_labels, weights=acfg.per_head, indices=gated,
                                 class_weights=aux_class_weight, stats=astats)
            loss = loss + acfg.w * a_loss
            stats["aux"] = astats
        if world > 1 or not loss.requires_grad:
            # every trainable parameter joins the graph (a module that took no part this step, e.g. the cross-view
            # sublayers for a photo alone, gets a zero gradient): DDP then needs no search for unused parameters,
            # and a step where nothing trained took part still runs its backward
            loss = loss + sum(p.sum() for p in m.parameters() if p.requires_grad) * 0.0
        return loss, stats

    # Precision: bf16 where the GPU has it (no gradient scaler), else fp16 with one.
    amp_on = amp and dev.type == "cuda"
    autocast_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if amp_on else torch.float32
    scaler = torch.amp.GradScaler("cuda", enabled=autocast_dtype == torch.float16)
    LOG.info(f"[amp] {'off' if not amp_on else str(autocast_dtype).split('.')[-1]}"
          + (" (+ gradient scaler)" if scaler.is_enabled() else ""))

    step_model = _Forward(net, micro_step)
    if world > 1:  # the masked-attention probabilities are set alike on every process
        step_model = DistributedDataParallel(step_model, device_ids=[dev.index], broadcast_buffers=False,
                                             gradient_as_bucket_view=True)  # the gradients live in DDP's buckets
    ema_model = ModelEMA(net, decay=ocfg.ema_decay, tau=ocfg.ema_tau, device=dev) if ocfg.ema else None
    if ema_model is not None and resume_ckpt is not None and resume_ckpt.get("ema"):
        ema = dict(resume_ckpt["ema"])
        ema["module"] = migrate_checkpoint({**resume_ckpt, "model": ema["module"]})["model"]
        ema_model.load_state_dict(ema)
        LOG.info(f"[resume] restored EMA ({ema_model.updates} updates)")

    iters_per_epoch = max(1, len(train_loader))
    total_iters = epochs * iters_per_epoch
    total_opt_steps = max(1, math.ceil(total_iters / accum_steps))
    if main and "photos" in val_parts and mask_anneal:
        free = max(EOMT_MASK_ANNEAL_END) * epochs
        windows = ", ".join(f"{a * epochs:.1f}-{b * epochs:.1f}" for a, b in zip(EOMT_MASK_ANNEAL_START, EOMT_MASK_ANNEAL_END))
        LOG.info(f"[val] every {vcfg.interval} epoch(s) and after the last, in the training regime (as the official EoMT): "
              f"masked attention at the current annealing (query-block windows at epochs {windows}). best.pt runs "
              f"mask-free: it is chosen from epoch {free:.1f} on.")
    head_group = next((g for g in reversed(optimizer.param_groups) if not g.get("is_backbone")), optimizer.param_groups[-1])
    tb = _Logger(logger, run_dir, name) if main else None

    # The links' calibration: the threshold of the linking embedding and the gate keys, from the last validation of
    # the sets (a post-trained checkpoint's until then).
    link = {"match_thres": (src or {}).get("match_thres") if net.matcher is not None or cross_view else None,
            "link_group_by": list(mv.gate) if mv is not None and mv.gate else None}

    def _save(path: Path, *, state_dict, trainer_state: bool, row: dict, best: float):
        extra = {"epoch": epoch, "best_metric": best, "best_key": best_key, "metrics": row}
        if trainer_state:
            extra["optimizer"] = optimizer.state_dict()
            if ema_model is not None:
                extra["ema"] = ema_model.state_dict()
            extra["lora"] = lora_cfg  # the factors stay apart in the trainer state (resume)
        elif lora_cfg:  # an exported checkpoint: the LoRA factors merged into the weights
            state_dict = merge_state(state_dict, scale=lora_cfg["alpha"] / lora_cfg["rank"])
        save_checkpoint(wrap_checkpoint(
            state_dict, size=size, nc=nc, imgsz=imgsz, names=names, task=family, aux_heads=aux_specs,
            aux_head_arch=aux_head_arch, letterbox=True, loss_weights=loss_weights,
            num_upscale_blocks=net.num_upscale_blocks, fpn_scales=net.fpn_scales, norm_mean=net.pixel_mean,
            norm_std=net.pixel_std, patch_size=net.patch_size, inference=inference, run_config=run_cfg,
            xview=net.xview_cfg, matcher=net.matcher_cfg, **link,
            **extra,
        ), path)

    # metrics.csv: the train loss and each attribute head's matched-query accuracy (MAE for a regression head), the
    # matching loss and what it sees, the share of time spent waiting for data, then the validation columns
    metric_of = {s.name: _aux_metric(s) for s in aux_specs}
    csv_fields = ["epoch", "train/loss"] + [f"train/{metric_of[s.name]}/{s.name}" for s in aux_specs]
    matching = net.matcher is not None
    attn_on = mv is not None and cross_view and mv.attn_w > 0
    if matching:
        csv_fields += ["train/match_loss"] + [f"train/match/{k}" for k in ("pos_p", "paired")]
    if attn_on:
        csv_fields.append("train/attn_loss")
    csv_fields.append("train/data_wait")
    val_keys = val_metric_keys(val_parts, aux_specs, is_detect=is_detect, group_by=vcfg.group_by,
                               link_group_by=mv.gate if mv is not None else None)
    csv_fields += [f"val/{k}" for k in val_keys]
    csv_log = _CsvLogger(run_dir / "metrics.csv", csv_fields) if main else None
    best_key = vcfg.best_key or ("sets/segm/mAP_set" if "sets" in val_parts and cross_view else "sets/match/f1"
                                 if "sets" in val_parts else "bbox/mAP" if is_detect else "segm/mAP")
    if val_keys and best_key not in val_keys:
        raise ValueError(f"val.best_key {best_key!r}: not a validation metric of this run (e.g. {val_keys[:3]} ...)")

    # --- epochs
    epoch = start_epoch
    for epoch in range(start_epoch, epochs):
        step_model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)
        zero = lambda: torch.zeros((), device=dev)  # noqa: E731
        running, running_match, running_attn = zero(), zero(), zero()
        match_sums = {k: [zero(), zero()] for k in ("pos_p", "paired")}  # sum, count of the finite
        aux_sums = {s.name: [zero(), zero()] for s in aux_specs}  # correct (abs. error), counted
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
        epoch_t0 = data_t = time.perf_counter()
        data_wait = 0.0
        pbar = progress(train_loader, desc=f"epoch {epoch}/{epochs - 1}", unit="batch", disable=not main)
        for step, batch_data in enumerate(pbar):
            data_wait += time.perf_counter() - data_t
            pixel_values, geom_labels, class_labels, aux_labels = batch_data[:4]
            set_ids, inst_ids, extras = batch_data[4:7] if mv is not None else (None, None, None)
            it = epoch * iters_per_epoch + step
            head_factor, vit_factor = _poly_two_stage_factors(it // accum_steps, total_opt_steps, ocfg.warmup_steps,
                                                              ocfg.poly_power)
            for g in optimizer.param_groups:
                g["lr"] = g["initial_lr"] * (vit_factor if g.get("is_backbone") else head_factor)
            if mask_anneal:
                net.eomt.set_attn_mask_probs(_attn_mask_probs(it, total_iters, ocfg.poly_power))
            pixel_values = pixel_values.to(dev, non_blocking=True)
            geom_labels = [g.to(dev, non_blocking=True) for g in geom_labels]
            class_labels = [c.to(dev, non_blocking=True) for c in class_labels]
            aux_labels = {k: [t.to(dev, non_blocking=True) for t in v] for k, v in aux_labels.items()}
            if extras is not None:
                extras = {"ignore": [None if t is None else t.to(dev, non_blocking=True) for t in extras["ignore"]],
                          "neg_weight": extras["neg_weight"].to(dev)}

            # gradient accumulation: zero at the start of each window, step on its boundary (or the epoch's last batch);
            # on several GPUs the gradients are averaged over them on the boundary only
            if step % accum_steps == 0:
                optimizer.zero_grad(set_to_none=True)
            boundary = (step + 1) % accum_steps == 0 or (step + 1) == iters_per_epoch
            with step_model.no_sync() if world > 1 and not boundary else contextlib.nullcontext():
                with torch.amp.autocast("cuda", enabled=amp_on, dtype=autocast_dtype):
                    loss, stats = step_model(pixel_values, geom_labels, class_labels, aux_labels, set_ids, inst_ids,
                                             extras)
                scaler.scale(loss / accum_steps).backward()
            if boundary:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), ocfg.clip_norm)
                scaler.step(optimizer)
                scaler.update()
                if ema_model is not None:
                    ema_model.update(net)

            running += loss.detach()
            if "match" in stats:
                running_match += stats["match"][0]
                for k, v in stats["match"][1].items():
                    ok = torch.isfinite(v)
                    match_sums[k][0] += torch.where(ok, v, 0.0)
                    match_sums[k][1] += ok.float()
            if "attn" in stats:
                running_attn += stats["attn"]
            for n, (hit, cnt) in stats.get("aux", {}).items():
                aux_sums[n][0] += hit
                aux_sums[n][1] += cnt
            if main and (step % LOG_EVERY == 0 or step + 1 == iters_per_epoch):  # the only reads of the GPU in a step
                postfix = {"loss": f"{float(loss.detach()):.3f}", "avg": f"{float(running) / (step + 1):.3f}",
                           "lr": f"{head_group['lr']:.2e}"}
                if aux_specs:
                    postfix["aux"] = " ".join(f"{n}={_fmt(float(s) / float(c) if float(c) else float('nan'), 2)}"
                                              for n, (s, c) in aux_sums.items())
                if matching:
                    postfix["match"] = f"{float(running_match) / (step + 1):.3f}"
                if attn_on:
                    postfix["attn"] = f"{float(running_attn) / (step + 1):.3f}"
                pbar.set_postfix(postfix)
                if tb is not None:
                    tb.log({"train/loss": float(loss.detach()), "lr/head": head_group["lr"]}, it)
            data_t = time.perf_counter()

        # --- the epoch's train metrics, summed over the GPUs on the main process
        epoch_s = time.perf_counter() - epoch_t0
        totals = sum_over_processes((float(running), float(running_match),
                                     {k: [float(s), float(c)] for k, (s, c) in match_sums.items()}, data_wait,
                                     {n: [float(s), float(c)] for n, (s, c) in aux_sums.items()}, float(running_attn)))
        row: dict = {"epoch": epoch}
        if main:
            running_f, match_f, match_tot, wait_f, aux_tot, attn_f = totals
            row["train/loss"] = running_f / (iters_per_epoch * world)
            row["train/data_wait"] = wait_f / world / max(epoch_s, 1e-9)
            aux_acc = {n: s / c if c else float("nan") for n, (s, c) in aux_tot.items()}
            row.update({f"train/{metric_of[n]}/{n}": v for n, v in aux_acc.items()})
            LOG.info(f"[epoch {epoch}] mean loss {row['train/loss']:.4f}; waited for data {100 * row['train/data_wait']:.0f} "
                  "% of the epoch" + ("; attribute heads (train) " + "  ".join(
                      f"{n} {metric_of[n][4:]}={v:.3f}" for n, v in aux_acc.items()) if aux_specs else ""))
            if matching:
                seen = {k: s / c if c else float("nan") for k, (s, c) in match_tot.items()}
                row["train/match_loss"] = match_f / (iters_per_epoch * world)
                row.update({f"train/match/{k}": v for k, v in seen.items()})
                LOG.info(f"[epoch {epoch}] set matcher loss {row['train/match_loss']:.4f}: true pairs at probability "
                         f"{_fmt(seen['pos_p'])}, {_fmt(100 * seen['paired'], 0)} % of the instances of a set of several "
                         "photos with a partner in it")
            if attn_on:
                row["train/attn_loss"] = attn_f / (iters_per_epoch * world)
                LOG.info(f"[epoch {epoch}] cross-view attention loss {row['train/attn_loss']:.4f}")
            if tb is not None and aux_specs:
                tb.log({f"train/{metric_of[n]}/{n}": v for n, v in aux_acc.items()}, epoch)

        # --- validation, in the regime the model is trained in (official EoMT): masked attention at the annealing
        # probabilities of the epoch's end; the EMA is what is validated and exported as best.pt
        eval_model = ema_model.module if ema_model is not None else net
        val_probs = (_attn_mask_probs((epoch + 1) * iters_per_epoch, total_iters, ocfg.poly_power) if mask_anneal
                     else [0.0] * net.config.num_blocks)
        deployable = not any(val_probs)  # best.pt must run mask-free, as predict runs it
        last_epoch = epoch + 1 in (epochs, stop_after_epochs)
        if main:  # the trainer state first: a crash in the validation then costs only the validation
            _save(weights_dir / "last.pt", state_dict=net.state_dict(), trainer_state=True, row=row, best=best_metric)
        metrics: dict = {}
        aux_pc = None
        if val_parts and (last_epoch or (epoch + 1) % max(1, vcfg.interval) == 0
                          or epoch >= epochs - vcfg.final_epochs):
            eval_model.eomt.set_attn_mask_probs(val_probs)
            if dev.type == "cuda":
                train_peak = torch.cuda.max_memory_allocated(dev)
                torch.cuda.reset_peak_memory_stats(dev)
            # a block mid-window unmasks a random share of its queries: the same ones at every validation
            with torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []), _inference_numerics():
                torch.manual_seed(0)
                metrics, aux_pc = validate_parts(
                    eval_model, val_parts, device=dev, batch=batch, workers=workers, conf_thres=vcfg.conf_thres,
                    max_det=vcfg.max_det, amp=amp, verbose=main, is_detect=is_detect, aux_iou_gate=acfg.iou_gate,
                    aux_class_gate=acfg.class_gate, link_group_by=link["link_group_by"], link_conf=link_conf,
                    thres=link["match_thres"], group_by=vcfg.group_by, tag=f"[epoch {epoch}]")
            eval_model.eomt.set_attn_mask_probs([0.0] * net.config.num_blocks)
            gs = "_gated" if link["link_group_by"] else ""
            if main and math.isfinite(metrics.get(f"sets/match{gs}/thres", float("nan"))):
                link["match_thres"] = metrics[f"sets/match{gs}/thres"]  # the links' calibration
            if tb is not None:
                tb.log({f"val/{k}": v for k, v in metrics.items()}, epoch)
            if dev.type == "cuda":  # validation's peak, then its cache back to training
                val_peak = torch.cuda.max_memory_allocated(dev)
                torch.cuda.empty_cache()
                if main:
                    LOG.info(f"[epoch {epoch}] peak GPU memory: training {train_peak / 2**30:.1f} GiB, validation "
                          f"{val_peak / 2**30:.1f} GiB")

        if main:  # the main process alone writes the checkpoints, the metrics row and the plots
            row.update({f"val/{k}": v for k, v in metrics.items()})
            row = {k: v for k, v in row.items() if k in csv_log.fields}
            cur = metrics.get(best_key)
            if deployable and cur is not None and math.isfinite(cur) and cur > best_metric:
                best_metric = cur
                _save(weights_dir / "best.pt", state_dict=eval_model.state_dict(), trainer_state=False, row=row,
                      best=best_metric)
                LOG.info(f"[epoch {epoch}] new best {best_key} {best_metric:.4f} -> best.pt")
            csv_log.append(row)
            try:  # plots never stop a run
                plot_metrics_csv(run_dir / "metrics.csv", run_dir / "metrics.png")
                if aux_pc:
                    plot_aux_per_class(aux_pc, names, run_dir / "aux_per_class.png",
                                       regression={n for n in aux_pc if metric_of[n] == "aux_mae"})
            except Exception as e:  # noqa: BLE001
                LOG.info(f"[plot] failed: {e}")
        if stop_after_epochs is not None and epoch + 1 >= stop_after_epochs:
            LOG.info(f"[stop] stopping after {epoch + 1} epoch(s) as requested; the schedule spans {epochs}")
            break

    if tb is not None:
        tb.close()
    best = weights_dir / "best.pt"
    return {"best_metric": best_metric, "best_key": best_key, "last": str(weights_dir / "last.pt"),
            "best": str(best) if best.exists() else None, "weights_dir": str(weights_dir)}

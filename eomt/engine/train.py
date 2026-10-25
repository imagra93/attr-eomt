"""Training loop for EoMT instance segmentation.

Initializes the encoder from DINOv2, fine-tunes with AdamW (a low LR multiplier
on the pretrained encoder, full LR on the mask/class head), a cosine schedule
with warmup, AMP and gradient clipping. The masked-attention probability is
annealed 1->0 over training (EoMT recipe) so the model converges to efficient
mask-free inference. Validates every ``val_interval`` epochs with COCO segm mAP
(in the training regime, at the current annealing, as the official EoMT does) and
keeps both ``last.pt`` and the ``best.pt`` (highest mask-free ``segm/mAP``, so only
once the annealing is over). The EoMT segmentation loss is computed inside the HF model.
``device="0,1"`` trains on several GPUs, one process each (DistributedDataParallel).
"""

from __future__ import annotations

import contextlib
import csv
import math
import warnings
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from ..aux_cls import (
    aux_accuracy,
    aux_accuracy_by_primary,
    aux_loss,
    gate_indices,
    match_loss,
    match_queries,
)
from ..config import INFERENCE_DEFAULTS, aux_specs_from_meta, normalize_loss_weights
from ..data import CocoDetection, CocoInstanceSeg, CocoValImages, CocoViewSets, collate_sets, collate_train
from ..data.coco import _attr_label
from ..data.transforms import AugConfig, build_val_transform
from ..ema import ModelEMA, _unwrap
from ..model import DEFAULT_FPN_SCALES, _interp_pos_grid, build_model, load_dinov2_backbone
from ..plotting import plot_aux_per_class, plot_metrics_csv
from ..device import gpu_ids, resolve_device
from ..serialization import load_raw, resolve_checkpoint, save_checkpoint, wrap_checkpoint
from .distributed import is_main, launch, rank, sum_over_processes, world_size
from .validate import _SEGM_KEYS, aux_evaluate, evaluate, evaluate_detection, evaluate_sets

_ENCODER_PREFIXES = ("eomt.embeddings", "eomt.layers", "eomt.layernorm")
# Parameters excluded from weight decay regardless of group: all 1-D tensors
# (LayerNorm weights, every bias) plus token/positional embeddings — the standard
# ViT fine-tuning policy (decaying these hurts).
_NO_DECAY_TOKENS = ("position_embeddings", "cls_token", "register_tokens")


def _seed_everything(seed: int) -> None:
    """Seed Python / NumPy / Torch RNGs for reproducible runs."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _count_encoder_layers(model) -> int:
    """Number of transformer blocks in the encoder (max ``eomt.layers.<i>`` + 1)."""
    ids = [
        int(n.split("eomt.layers.")[1].split(".")[0])
        for n, _ in model.named_parameters()
        if n.startswith("eomt.layers.")
    ]
    return (max(ids) + 1) if ids else 0


def _llrd_scale(name: str, num_layers: int, llrd: float) -> float:
    """Layer-wise LR decay factor for a backbone param, as in the reference EoMT recipe.

    Block ``i`` scales by ``llrd ** (num_layers - 1 - i)``: the **top block gets exactly the
    base LR** and each block below it is ``llrd`` times the one above. The patch / position
    embeddings follow the first block, and the post-encoder norm gets the base LR. In EoMT the
    last ``num_blocks`` transformer blocks are the ones that process the queries and predict the
    masks, so they must not be starved of learning rate.
    """
    if llrd >= 1.0 or num_layers <= 0:
        return 1.0
    if name.startswith("eomt.layers."):
        return llrd ** (num_layers - 1 - int(name.split("eomt.layers.")[1].split(".")[0]))
    if name.startswith("eomt.embeddings"):
        return llrd ** (num_layers - 1)
    return 1.0  # post-encoder layernorm / anything else in the backbone


def build_optimizer(
    model, lr: float, weight_decay: float, backbone_lr_mult: float, llrd: float = 1.0
):
    """AdamW with layer-wise LR decay, an optional backbone multiplier, and no-WD on norms/biases.

    Param groups are keyed by ``(lr, weight_decay)``. The task head gets ``lr``. The DINOv2
    encoder gets ``lr * backbone_lr_mult`` scaled per block by ``llrd`` (see :func:`_llrd_scale`:
    the top block is at ``lr * backbone_lr_mult``). The reference EoMT recipe uses
    ``backbone_lr_mult=1.0`` and ``llrd=0.8``; ``llrd=1.0`` is a flat encoder LR. 1-D tensors and
    positional/token embeddings are placed in weight-decay-free groups (the reference decays
    everything; this is the usual ViT fine-tuning policy and a second-order difference).
    """
    num_layers = _count_encoder_layers(model)
    groups: dict[tuple, dict] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_backbone = name.startswith(_ENCODER_PREFIXES)
        base = lr * backbone_lr_mult if is_backbone else lr
        lr_i = base * (_llrd_scale(name, num_layers, llrd) if is_backbone else 1.0)
        no_decay = p.ndim <= 1 or any(tok in name for tok in _NO_DECAY_TOKENS)
        wd_i = 0.0 if no_decay else weight_decay
        # ``is_backbone`` is part of the key: with the reference recipe the top ViT block and the final
        # norm have exactly the head's LR, and a shared group would give head parameters the ViT warmup
        # (LR held at 0) or the other way round. ``is_backbone`` labels the whole group for the schedule.
        key = (round(lr_i, 12), wd_i, is_backbone)
        groups.setdefault(
            key, {"params": [], "lr": lr_i, "weight_decay": wd_i, "is_backbone": is_backbone}
        )["params"].append(p)
    opt = torch.optim.AdamW(list(groups.values()), lr=lr, weight_decay=weight_decay)
    for g in opt.param_groups:
        g["initial_lr"] = g["lr"]
    return opt


#: Multi-photo post-training (``train(multiview=...)``): the cross-view modules (``xview``, see
#: :data:`eomt.model.DEFAULT_XVIEW`), the matching head's width (0 = none), its contrastive loss weight and
#: temperature, the LR multiplier (of ``lr0``) of each parameter group that trains — every other parameter of the
#: checkpoint stays frozen — and how the synthetic sets are drawn (:class:`~eomt.data.sets.SetConfig`).
DEFAULT_MULTIVIEW: dict = {
    "xview": {"mode": "blocks"},
    "match_dim": 128,
    "match_w": 1.0,
    "match_tau": 0.1,
    "trainable": {"xview": 1.0, "match_head": 1.0},
    "sets": {},
}
#: Parameter groups of the multi-photo post-training -> name prefixes (the query blocks are added per model).
_MULTIVIEW_GROUPS = {
    "xview": ("eomt.xview.", "eomt.refiner."),
    "match_head": ("match_head.",),
    "query_blocks": ("eomt.cross_blocks.", "eomt.query."),
    "heads": ("eomt.class_predictor.", "eomt.mask_head.", "eomt.upscale_block.", "eomt.quality_head.",
              "eomt.layernorm.", "aux_heads."),
}


def build_multiview_optimizer(model, lr: float, weight_decay: float, trainable: dict[str, float]):
    """Freeze every parameter outside the ``trainable`` groups (:data:`_MULTIVIEW_GROUPS`; the embeddings, the layers
    before the queries and the FPN always stay frozen) and give each group ``lr`` times its multiplier (AdamW, no decay
    on 1-D tensors). The query blocks warm up with the ViT (``is_backbone``), the others with the head."""
    unknown = set(trainable) - set(_MULTIVIEW_GROUPS)
    if unknown:
        raise ValueError(f"multiview trainable groups {sorted(unknown)}: choose from {sorted(_MULTIVIEW_GROUPS)}")
    n, nb = model.config.num_hidden_layers, model.config.num_blocks
    prefixes = {**_MULTIVIEW_GROUPS, "query_blocks": _MULTIVIEW_GROUPS["query_blocks"]
                + tuple(f"eomt.layers.{i}." for i in range(n - nb, n))}
    groups: dict[tuple, dict] = {}
    for name, p in model.named_parameters():
        g = next((g for g, pre in prefixes.items() if name.startswith(pre)), None)
        p.requires_grad_(g in trainable)
        if g not in trainable:
            continue
        wd = 0.0 if p.ndim <= 1 else weight_decay
        groups.setdefault((g, wd), {"params": [], "lr": lr * float(trainable[g]), "weight_decay": wd,
                                    "is_backbone": g == "query_blocks"})["params"].append(p)
    opt = torch.optim.AdamW(list(groups.values()), lr=lr, weight_decay=weight_decay)
    for g in opt.param_groups:
        g["initial_lr"] = g["lr"]
    return opt


def _adopt_checkpoint_aux(ds, ckpt: dict) -> None:
    """Keep a post-trained checkpoint's attribute heads as they were trained: its standardization of the regression
    values and its flip-aware swaps (the dataset JSON may lack them, or yield other statistics), over the dataset's
    vocabularies (which must be the same) and training-only settings (``crop``)."""
    saved = {s.name: s for s in aux_specs_from_meta(ckpt.get("aux_heads"))}
    missing = set(saved) - {s.name for s in ds.aux_specs}
    if missing:  # the post-trained checkpoint would silently lose them
        raise ValueError(f"the dataset does not define the checkpoint's attribute heads {sorted(missing)}")
    for s in ds.aux_specs:
        ck = saved[s.name]
        if not s.regression and s.names != ck.names:
            raise ValueError(f"attribute {s.name!r}: the dataset's categories {s.names} differ from the checkpoint's {ck.names}")
        s.mean, s.std, s.value_range, s.flip_aware = ck.mean, ck.std, ck.value_range, ck.flip_aware
        s.log, s.applies_to = ck.log, ck.applies_to  # a data-only dataset leaves these to the run config


def _sets_fields(cases, aux_specs) -> list[str]:
    """The ``metrics.csv`` columns of :func:`~eomt.engine.validate.evaluate_sets`."""
    keys = [f"segm/{k}_{m}" for k in ("mAP", "mAP50", "mAP_small", "AR100") for m in ("alone", "set")]
    keys += [f"{c}/mAP_{m}" for c in cases for m in ("alone", "set")]
    keys += [f"{_aux_metric(s)}/{s.name}_{m}" for s in aux_specs for m in ("alone", "set")]
    keys += [f"{k}/{n}" for k in ("match", "reid") for n in ("f1", "precision", "recall", "thres")]
    return [f"val/sets/{k}" for k in keys]


def _poly_two_stage_factors(opt_step, total_steps, warmup_steps, power):
    """``(head_factor, vit_factor)`` of the reference two-stage warmup + polynomial schedule.

    ``warmup_steps = (head, vit)`` in optimizer steps. The head warms up linearly while the ViT
    LR is held at **0**; the ViT then ramps up linearly over its own warmup. Both then decay as
    ``(1 - progress) ** power`` to 0 over the remaining steps (each group measured from the end of
    its own warmup), exactly like ``TwoStageWarmupPolySchedule`` in the official EoMT code.
    """
    w_head, w_vit = int(warmup_steps[0]), int(warmup_steps[1])

    def poly(adjusted, span):
        return (1.0 - min(1.0, adjusted / max(1, span))) ** power

    if w_head > 0 and opt_step < w_head:
        head = opt_step / w_head
    else:
        head = poly(max(0, opt_step - w_head), total_steps - w_head)
    if opt_step < w_head:
        vit = 0.0
    elif w_vit > 0 and opt_step < w_head + w_vit:
        vit = (opt_step - w_head) / w_vit
    else:
        vit = poly(max(0, opt_step - w_head - w_vit), total_steps - w_head - w_vit)
    return head, vit


def _attn_mask_prob(it, total_iters, start_frac, end_frac, power=1.0):
    """Masked-attention annealing schedule of one query block (EoMT, CVPR 2025).

    Returns the probability of applying masked attention to each query at
    iteration ``it``: held at 1.0 until the start of the block's window, then
    ``(1 - progress) ** power`` down to 0.0 at its end (the reference recipe uses the
    polynomial power 0.9), so the final stretch trains mask-free and matches efficient
    (mask-less) inference.
    """
    start = start_frac * total_iters
    end = end_frac * total_iters
    if it < start:
        return 1.0
    if it >= end:
        return 0.0
    return (1.0 - (it - start) / max(1.0, end - start)) ** power


#: Per-block masked-attention annealing windows (fractions of training) of the reference recipe: the
#: four query blocks are annealed one after the other, and the last sixth of training is mask-free.
EOMT_MASK_ANNEAL_START = (1 / 6, 2 / 6, 3 / 6, 4 / 6)
EOMT_MASK_ANNEAL_END = (2 / 6, 3 / 6, 4 / 6, 5 / 6)


#: The high-resolution final phase (``hires_epochs``) is a short fine-tune of the averaged weights: its own short warmup
#: and a faster EMA, as a warm-started run of a few epochs would have (the main run's 0.9999 would barely move).
HIRES_WARMUP_STEPS = (100, 200)
HIRES_EMA_DECAY, HIRES_EMA_TAU = 0.999, 500.0
#: Share of a GPU's usable memory a high-resolution training step may take; the rest covers fragmentation and the
#: peaks one probe step does not hit.
HIRES_MEMORY_BUDGET = 0.9
_POS_KEY = "eomt.embeddings.position_embeddings.weight"


def _state_at_imgsz(state_dict: dict, from_imgsz: int, to_imgsz: int, patch: int) -> dict:
    """The state of a model built at ``from_imgsz`` as one built at ``to_imgsz``: its position-embedding table resized
    as the model resizes it on the fly for a ``to_imgsz`` input, so both predict the same at that size."""
    if from_imgsz == to_imgsz or _POS_KEY not in state_dict:
        return state_dict
    g0, g1 = from_imgsz // patch, to_imgsz // patch
    return {**state_dict, _POS_KEY: _interp_pos_grid(state_dict[_POS_KEY], (g0, g0), (g1, g1))}


def _fit_batch(requested: int, fits) -> int:
    """The largest micro-batch from ``requested`` down (halving) for which ``fits(batch)`` holds; 0 when none does."""
    b = requested
    while b >= 1:
        if fits(b):
            return b
        b //= 2
    return 0


def _per_block(value, n_blocks: int, what: str) -> list[float]:
    """One fraction of training per query block."""
    vals = [float(v) for v in value]
    if len(vals) != n_blocks:
        raise ValueError(f"{what} needs {n_blocks} values (one per query block), got {len(vals)}")
    return vals


def _attn_mask_probs(it, total_iters, starts, ends, power=1.0) -> list[float]:
    """Masked-attention probability of every query block at iteration ``it`` (one window each)."""
    return [_attn_mask_prob(it, total_iters, a, b, power) for a, b in zip(starts, ends)]


def _aux_w_factor(it, total_iters, warmup_frac):
    """Linear 0->1 ramp of the aux-loss weight over the first ``warmup_frac`` of training.

    Keeps the early backbone driven by the segmentation loss; ``0`` disables warmup
    (full weight from step 0).
    """
    if warmup_frac <= 0:
        return 1.0
    return min(1.0, it / max(1.0, warmup_frac * total_iters))


def _in_scope_labels(train_ds, spec) -> list:
    """Labels (:func:`eomt.data.coco._attr_label`, ignored ones included) of the train annotations a head applies to."""
    anns = getattr(train_ds.coco, "dataset", {}).get("annotations", []) or []
    out = []
    for a in anns:
        cls = train_ds.cat2contig.get(a.get("category_id"))
        # Out-of-scope instances never reach a class-scoped head's loss, so they count for neither its coverage nor
        # its weights.
        if cls is None or (spec.applies_to is not None and cls not in spec.applies_to):
            continue
        out.append(_attr_label(spec, train_ds._attr_id_maps, a.get("attributes", {}), cls))
    return out


def _valid_labels(spec, labels: list) -> list:
    return [v for v in labels if (v == v if spec.regression else v >= 0)]  # NaN != NaN


def _density_bins(spec, values: list, n_bins: int = 50):
    """Inverse-sqrt smoothed-density weights of a regression head (LDS, Yang et al. 2021) as ``(edges, weights)`` over
    its standardized values, mean-normalized over the training values; ``None`` when the values do not spread."""
    z = spec.encode(torch.tensor(values, dtype=torch.float64)).float()
    if z.numel() < 2 or float(z.max() - z.min()) < 1e-6:
        return None
    edges = torch.linspace(float(z.min()), float(z.max()), n_bins + 1)
    counts = torch.histc(z, bins=n_bins, min=float(edges[0]), max=float(edges[-1]))
    kernel = torch.exp(-0.5 * (torch.arange(-2.0, 3.0) / 2.0) ** 2)  # the paper's Gaussian, 5 bins, sigma 2
    density = torch.nn.functional.conv1d(counts[None, None], (kernel / kernel.sum())[None, None], padding=2)[0, 0]
    w = density.clamp(min=1e-6).rsqrt()
    return edges, w / w[torch.bucketize(z, edges[1:-1])].mean()


def _compute_aux_class_weights(train_ds) -> dict:
    """Inverse-sqrt-frequency CE weights per aux head (mean-normalized to ~1); for a regression head, inverse-sqrt
    smoothed-density weights of its values (see :func:`_density_bins`).

    Counts each head's contiguous-class frequency from the in-memory train
    annotations (no image decode); missing / out-of-vocab values are ignored.
    """
    out: dict = {}
    for spec in train_ds.aux_specs:
        labels = _valid_labels(spec, _in_scope_labels(train_ds, spec))
        if spec.regression:
            bins = _density_bins(spec, labels)
            if bins is not None:
                out[spec.name] = bins
            continue
        counts = torch.zeros(spec.num_classes)
        for cid in labels:
            counts[cid] += 1
        w = 1.0 / counts.clamp(min=1).sqrt()
        out[spec.name] = w * (spec.num_classes / w.sum())  # mean weight ~ 1.0
    return out


def _aux_coverage(train_ds) -> dict[str, tuple[int, int]]:
    """Per head ``{name: (supervised, in_scope)}`` instance counts from train annotations.

    ``in_scope`` counts instances whose primary class the head applies to (every
    instance for an unscoped head); ``supervised`` counts those of them carrying a
    tagged, valid value (not ignored). Diagnostic only — no image is decoded.
    """
    out: dict[str, tuple[int, int]] = {}
    for spec in train_ds.aux_specs:
        labels = _in_scope_labels(train_ds, spec)
        out[spec.name] = (len(_valid_labels(spec, labels)), len(labels))
    return out


def _aux_kind_str(spec) -> str:
    """``num_classes`` of a categorical head, the target space of a regression head (banner)."""
    if not spec.regression:
        pairs = [f"{spec.names.get(a, a)}<->{spec.names.get(b, b)}" for a, b in (spec.flip_aware or {}).items() if a < b]
        return str(spec.num_classes) + (f", flips {' '.join(pairs)}" if pairs else "")
    crop = f", crop {spec.crop}" + (f" < {spec.min_visible:g}" if spec.crop == "ignore" else "") if spec.crop else ""
    return f"regression{', log' if spec.log else ''}{crop}, mean {spec.mean:.4g} std {spec.std:.4g}"


def _head_record(spec, class_names: dict) -> dict:
    """How one attribute head learns, as the run records it (``args.yaml`` / ``run_config``): its type and categories
    or regression space, and its behavior under the augmentation, in names rather than contiguous ids."""
    if spec.regression:
        rec = {"type": "regression", "log": spec.log, "range": list(spec.value_range) if spec.value_range else None,
               "crop": spec.crop, "min_visible": spec.min_visible if spec.crop == "ignore" else None,
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


def _aux_scope_str(spec, names: dict) -> str:
    """Human-readable ``applies_to`` scope for the banner ("" when unscoped)."""
    if spec.applies_to is None:
        return ""
    if len(spec.applies_to) > 6:
        return f" on {len(spec.applies_to)} classes"
    labels = sorted(names.get(c, str(c)) for c in spec.applies_to)
    return " on {" + ",".join(labels) + "}"


def _write_run_config(path: Path, cfg: dict) -> None:
    """Persist the resolved run hyper-parameters (incl. model ``size``) to YAML.

    Written once at startup so the run directory records how it was trained even
    if training is interrupted before the first checkpoint (by the main process on several GPUs).
    """
    import yaml

    if not is_main():
        return
    with path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


#: Iterations between two measurements of the attribute heads' pull on the shared queries (``train/aux_grad``).
AUX_GRAD_EVERY = 50


def _aux_grad_ratios(out: dict, per_head: dict, aux_w_eff: float, weights: dict | None, log_vars) -> dict[str, float]:
    """Each attribute head's pull on the shared queries: the norm of the gradient of its weighted loss on the query rows
    of the final hidden state, over that of the segmentation loss. Well above 1 the head steers the encoder more than
    the segmentation does; far below it only reads the encoder's features."""
    h = out["last_hidden_state"]
    nq = out["query_embed"].shape[1]
    seg = float(torch.autograd.grad(out["loss"], h, retain_graph=True)[0][:, :nq].float().norm())
    ratios = {}
    for name, loss in per_head.items():
        if not loss.requires_grad:  # no query matched in this micro-batch
            continue
        w = aux_w_eff * float((weights or {}).get(name, 1.0))
        if log_vars is not None:
            w *= float(torch.exp(-log_vars[name]))
        g = torch.autograd.grad(loss, h, retain_graph=True)[0][:, :nq]
        ratios[name] = w * float(g.float().norm()) / max(seg, 1e-12)
    return ratios


def _safe_acc(hit: int, tot: int) -> float:
    """Matched-query accuracy, or NaN when no query matched this epoch (no info)."""
    return hit / tot if tot else float("nan")


class _CsvLogger:
    """Append per-epoch metrics to ``<run_dir>/metrics.csv``.

    Columns are fixed up front: segmentation always (train loss + val COCO segm/bbox
    mAP), plus one matched-query-accuracy column per secondary head when the dataset
    has them. Any metric without a value for an epoch — no validation that epoch, or a
    head with no matched queries — is written as ``nan`` rather than skipped, so every
    row has the same columns.
    """

    def __init__(self, path: Path, fields: list[str]):
        self.path = path
        self.fields = fields
        # write the header on a fresh run; keep prior rows when resuming
        if not path.exists() or path.stat().st_size == 0:
            with path.open("w", newline="") as f:
                csv.writer(f).writerow(fields)

    def append(self, row: dict):
        with self.path.open("a", newline="") as f:
            csv.writer(f).writerow([row.get(k, float("nan")) for k in self.fields])


class _Forward(torch.nn.Module):
    """``fn(model, *args)`` as a module's forward, so that DistributedDataParallel sees every parameter a training
    step uses: the attribute heads run on the model's outputs, outside ``model.forward``, and get no gradient when no
    query matches, which DDP tolerates only for parameters used inside its forward (``find_unused_parameters``)."""

    def __init__(self, model, fn, extra=None):
        super().__init__()
        self.model, self.fn = model, fn
        self.extra = extra  # parameters used by ``fn`` outside the model (the attribute heads' learned log-variances)

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

    def log(self, scalars: dict, step: int):
        if self.kind == "tensorboard" and self.writer is not None:
            for k, v in scalars.items():
                self.writer.add_scalar(k, v, step)
        elif self.kind == "wandb" and self.writer is not None:
            self.writer.log(scalars, step=step)

    def close(self):
        if self.kind == "tensorboard" and self.writer is not None:
            self.writer.close()
        elif self.kind == "wandb" and self.writer is not None:
            self.writer.finish()


def train(
    *,
    train_images: str,
    train_json: str,
    val_images: str | None = None,
    val_json: str | None = None,
    size: str = "l",
    family: str = "instance",
    imgsz: int = 644,
    epochs: int = 12,
    batch: int = 4,
    nominal_batch: int = 16,
    accum: int = 0,
    lr0: float = 1e-4,
    weight_decay: float = 0.05,
    backbone_lr_mult: float = 1.0,
    llrd: float = 0.8,
    poly_power: float = 0.9,
    warmup_steps: tuple[int, int] = (500, 1000),
    ema: bool = True,
    ema_decay: float = 0.9999,
    ema_tau: float = 2000.0,
    mask_anneal: bool = True,
    mask_anneal_start: Sequence[float] = EOMT_MASK_ANNEAL_START,
    mask_anneal_end: Sequence[float] = EOMT_MASK_ANNEAL_END,
    clip_norm: float = 0.01,
    workers: int = 8,
    prefetch: int = 4,
    device: str = "auto",
    amp: bool = True,
    amp_dtype: str = "auto",
    tf32: bool = True,
    seed: int | None = None,
    pretrained: bool = True,
    flip_prob: float | None = None,
    min_scale: float | None = None,
    max_scale: float | None = None,
    aug: AugConfig | dict | None = None,
    train_transform=None,
    keep_empty: bool = False,
    letterbox: bool = True,
    project: str = "runs/train",
    name: str | None = None,
    val_interval: int = 2,
    conf_thres: float = 0.0,
    max_det: int = 100,
    aux_w: float = 1.0,
    aux_w_warmup: float = 0.0,
    aux_w_per_head: dict[str, float] | None = None,
    aux_class_weights: bool = False,
    aux_iou_gate: float = 0.5,
    aux_class_gate: bool = True,
    aux_balance: str = "static",
    aux_head_layers: int = 2,
    aux_head_hidden: int | None = None,
    aux_head_dropout: float = 0.0,
    no_object_weight: float = 0.1,
    class_weight: float = 2.0,
    mask_weight: float = 5.0,
    dice_weight: float = 5.0,
    l1_weight: float = 5.0,
    giou_weight: float = 2.0,
    iou_aware_cls: bool = False,
    quality_weight: float = 0.0,
    num_upscale_blocks: int | None = None,
    fpn_scales: tuple[float, ...] | list[float] | None = DEFAULT_FPN_SCALES,
    box_head: bool = False,
    deep_supervision: bool = True,
    stop_after_epochs: int | None = None,
    val_imgsz: int | None = None,
    hires_epochs: int = 0,
    hires_imgsz: int | None = None,
    hires_batch: int | None = None,
    hires_lr: float | None = None,
    hires_aug: dict | None = None,
    attributes: dict | None = None,
    inference: dict | None = None,
    logger: str = "none",
    resume: str | None = None,
    init_weights: str | None = None,
    multiview: dict | None = None,
) -> dict:
    """Fine-tune an EoMT instance-segmentation model on a COCO-format dataset.

    ``resume`` continues a run (restores epoch, optimizer, LR schedule, EMA, best).
    ``init_weights`` is a *warm start* (fine-tune): it loads only the model weights
    from a checkpoint and then trains from epoch 0 with a fresh optimizer, LR
    schedule and EMA — use it to retrain with changed hyper-parameters (e.g.
    ``flip_prob=0``) while keeping the learned weights. The two are mutually
    exclusive.

    ``fpn_scales`` defaults to the B1 multi-scale recipe (SimpleFPN + query
    cross-attention, :data:`~eomt.model.DEFAULT_FPN_SCALES`); pass ``None`` for the
    original single-scale model. Resuming ignores this and keeps the checkpoint's
    architecture; warm-starting (``init_weights``) honors it, so fine-tuning a
    single-scale checkpoint into a multi-scale one is the default warm-start path.

    **Architecture options** (instance family). ``box_head=True`` adds an auxiliary box head trained with L1 +
    GIoU (``l1_weight`` / ``giou_weight``) whose boxes also enter the matching cost. ``deep_supervision=False``
    stops supervising a query block's prediction once that block's masked attention is annealed away; while the
    block masks, its prediction builds the attention mask and stays supervised.
    With the default windows the blocks drop out at 2/6, 3/6, 4/6 and 5/6 of training and only the
    last sixth runs final-only; the reference recipe
    supervises every block throughout. Resuming keeps the checkpoint's ``box_head``; warm starts honor the argument.

    **Attribute heads.** A dataset attribute is categorical (cross-entropy) or ``"type": "regression"`` (a real
    number: Huber loss on its standardized value, ``log1p`` first with ``"log": true``; validated as the MAE in real
    units, ``val/aux_mae/<name>``). Both losses are O(1), so equal weights are a sane start: ``aux_w`` scales them all,
    ``aux_w_per_head`` each one, and ``aux_class_weights=True`` re-weights by inverse-sqrt class frequency (by
    inverse-sqrt smoothed density for a regression head). ``aux_balance="uncertainty"`` learns one weight per head
    instead (``exp(-s) * loss + s``, Kendall et al. 2018; the segmentation loss stays the fixed anchor). Every
    epoch prints each head's pull on the shared queries, the norm of its weighted loss's gradient on them over the
    segmentation loss's (``train/aux_grad/<name>`` in the CSV): well above 1 the head steers the encoder, far below it
    only reads its features.

    **Class targets** (instance family; persisted with the loss weights). ``iou_aware_cls=True`` trains a matched query's
    class probability towards ``(1 + IoU) / 2`` of its mask rather than 1, so the score ranks masks by their quality.
    ``quality_weight > 0`` adds a mask-quality head that predicts each matched query's mask IoU (Mask Scoring R-CNN);
    the score at inference becomes class probability x predicted IoU.

    ``val_imgsz`` validates (and picks ``best.pt``) at another input size than ``imgsz``; ``None`` = ``imgsz``. Scale
    augmentation makes the objects the model trains on larger, on average, than they are in a val image resized to
    ``imgsz`` (``min_scale``..``max_scale`` of ``imgsz``); validating at ``imgsz`` times the mean training scale measures
    the model at the scale it learned.

    **Several GPUs.** ``device="0,1"`` (GPU indices) runs one process per GPU with DistributedDataParallel;
    ``"auto"`` is GPU 0 alone. ``batch`` is per GPU and ``nominal_batch`` stays the global effective batch, so the
    accumulation shrinks (``nominal_batch / (batch x GPUs)``) and the optimizer steps, LR schedule and mask annealing
    are those of one GPU. Gradients are averaged over the GPUs on the last micro-step of
    each accumulation window. Each GPU draws its own share of the images and augmentations (seed + rank); validation
    splits the val images between the GPUs and the main process scores them all; it alone logs and writes
    checkpoints and plots. ``workers`` is per GPU. Every argument must be picklable (the processes are spawned).

    ``stop_after_epochs`` ends the run once that many epochs have completed (counted from the start of the run,
    so it composes with ``resume``) while the LR schedule and mask annealing still span ``epochs``: it is for
    truncated A/B comparisons that must follow exactly the schedule of a full run.

    **High-resolution final phase.** ``hires_epochs`` > 0 trains the last ``hires_epochs`` of ``epochs`` at
    ``hires_imgsz`` (e.g. the native size of the photos a model trained at 644 is deployed on). The main schedule (LR,
    warmup, masked-attention annealing) spans the first ``epochs - hires_epochs`` epochs exactly as a run of that length
    would; the final phase then starts from the EMA weights as a short fine-tune of its own: the same augmentation
    settings at the new size, with ``hires_aug`` (:class:`~eomt.data.transforms.AugConfig` fields) overriding them for the
    phase, e.g. a ``zoom_scale`` scaled by ``imgsz / hires_imgsz`` to keep the close-ups at the same magnification of the
    photo (the instance-crop rarity weights stay those of ``aug``); ``hires_lr`` (default ``lr0 / 5``) with a
    :data:`HIRES_WARMUP_STEPS` warmup and
    polynomial decay, a fresh EMA (:data:`HIRES_EMA_DECAY`), mask-free, validated every epoch at ``hires_imgsz``. Its
    micro-batch is ``hires_batch`` (default ``batch``), the accumulation adjusted to keep the effective batch. Before
    training starts, one real training step at the high resolution is run on every GPU: when it does not fit in
    :data:`HIRES_MEMORY_BUDGET` of the memory (with the optimizer, EMA and DDP buffers added), the micro-batch is halved
    down to 1, and the run stops at once if even 1 does not fit. Each phase keeps its own best, ``best_<imgsz>.pt``
    (saved at that size: its position-embedding table resized so it loads as a model of that size, same outputs);
    ``best.pt`` is the best of the whole run, each phase validated at its own size; ``metrics.csv`` gains an
    ``imgsz`` column. Instance family, built-in augmentation only.

    **Run config.** ``attributes`` (``{name: behavior}``, the ``attributes`` section of the run config YAML, see
    :mod:`eomt.data.config`) picks the attributes of the data that get a head and sets how each learns
    (:data:`~eomt.data.coco.BEHAVIOR_KEYS`: ``flip_aware``, ``crop``, ``log``, ``range``, ...) over what the data
    declares; ``None`` trains every attribute as the data declares it. ``inference`` (:data:`~eomt.config.INFERENCE_DEFAULTS`
    keys) is saved in every checkpoint as the model's predict / track defaults. ``args.yaml`` and every checkpoint
    (``run_config``) record the resolved run: these, the augmentation and every hyper-parameter.

    **Augmentation.** ``aug`` is an :class:`~eomt.data.transforms.AugConfig` (or a plain mapping, e.g. the ``train_aug``
    block of the dataset YAML) holding every augmentation knob; the defaults are a strong general recipe (flip,
    large-scale jitter, small rotation / shear / perspective, gamma, grayscale, blur, noise, JPEG, glare and
    "safe" erasing; mosaic / mixup / copy-paste and the instance-aware crop are off). ``flip_prob`` / ``min_scale``
    / ``max_scale`` override it when given. ``aug={"preset": "legacy"}`` restores the original flip + LSJ + crop +
    colour-jitter recipe with hard masks. ``train_transform`` replaces the whole built-in pipeline with your own
    ``(image, masks) -> (image, masks)`` callable (``aug`` is then ignored for that dataset).
    ``keep_empty=True`` trains on the images that have no annotations too (negatives: zero targets, every query is
    pushed to "no object"); by default they are dropped. Instance family only.

    **Optimisation recipe** (the reference EoMT recipe): AdamW at ``lr0`` for the head; the DINOv2
    encoder uses layer-wise LR decay ``llrd`` (0.8) with the top block at the base LR
    (``backbone_lr_mult=1.0``, no extra backbone multiplier), so the last blocks, which process the
    queries and predict the masks, learn as fast as the head. The LR follows a polynomial decay
    (``poly_power`` 0.9) to 0 with a two-stage warmup in optimizer steps, ``warmup_steps=(head, vit)``
    = (500, 1000): the head warms up while the ViT LR is held at 0, then the ViT ramps up. Masked
    attention is annealed per query block, one window after the other (``mask_anneal_start`` /
    ``mask_anneal_end``: one fraction of training per block; ``(1 - progress) ** poly_power`` inside a
    window), and the last sixth of training is mask-free. ``epochs=12`` and ``val_interval=2`` are the official COCO
    instance recipe (about 89k optimizer steps at batch 16).

    **Validation** (every ``val_interval`` epochs and after the last one) runs the model in the regime it is trained in,
    as the official EoMT does: masked attention at the annealing probabilities of the epoch's end. A model that still
    masks is far worse without its masks, so a mask-free curve before the last sixth follows the annealing, not the
    model. ``best.pt`` must be deployable mask-free, so it is only chosen once every block's annealing is over (a run
    stopped earlier has ``last.pt`` alone); every checkpoint is saved mask-free.
    ``amp_dtype`` selects the autocast dtype: ``"auto"`` uses bf16 on hardware that supports it (no GradScaler, more
    stable), else fp16; force with ``"bf16"`` / ``"fp16"``.

    **Several photos predicted together.** ``multiview`` (see :data:`DEFAULT_MULTIVIEW`) post-trains the
    checkpoint given as ``init_weights`` so that the photos of one subject are predicted in one pass, each photo's
    queries taking what the others show: it adds the cross-view modules and a matching head and trains them on
    synthetic sets made from single annotated photos (:class:`~eomt.data.sets.CocoViewSets`; ``batch`` counts sets).
    Only the groups listed in ``trainable`` learn (stage A: the new modules, so a single photo is predicted exactly as
    before; stage B adds ``query_blocks`` at a low multiplier). The checkpoint's architecture, loss weights,
    attribute-head definitions and ``inference`` settings are kept, masked attention and the high-resolution phase
    stay off, and ``best.pt`` is chosen on the hard photos of fixed validation sets predicted together with their set
    (``sets/segm/mAP_set``, see :func:`~eomt.engine.validate.evaluate_sets`).
    """
    if len(warmup_steps) != 2:
        raise ValueError("warmup_steps must be (head_steps, vit_steps)")
    if aux_balance not in ("static", "uncertainty"):
        raise ValueError(f"aux_balance={aux_balance!r}: 'static' or 'uncertainty'")
    if imgsz % 14 or (val_imgsz is not None and val_imgsz % 14):
        raise ValueError(f"imgsz={imgsz} and val_imgsz={val_imgsz} must be divisible by 14 (DINOv2 grid).")
    if multiview is not None and hires_epochs:  # e.g. a run config's `train` section written for the main training
        print("[xview] the high-resolution phase stays off: the post-training runs at the checkpoint's input size")
        hires_epochs = 0
    if hires_epochs:
        if not 0 < hires_epochs < epochs:
            raise ValueError(f"hires_epochs={hires_epochs} must leave some of epochs={epochs} at imgsz")
        if not hires_imgsz or hires_imgsz % 14:
            raise ValueError(f"hires_imgsz={hires_imgsz} must be set and divisible by 14 (DINOv2 grid).")
        if family == "detect" or train_transform is not None:
            raise ValueError("hires_epochs needs the instance family and the built-in augmentation")
    if inference and set(inference) - set(INFERENCE_DEFAULTS):
        raise ValueError(f"inference: unknown keys {sorted(set(inference) - set(INFERENCE_DEFAULTS))} "
                         f"(allowed: {', '.join(INFERENCE_DEFAULTS)})")

    gpus = gpu_ids(device)
    if len(gpus) > 1 and not dist.is_initialized():  # one process per GPU, each running this function on its own
        return launch(train, gpus, {k: v for k, v in locals().items() if k != "gpus"})
    world, main = world_size(), is_main()

    if seed is not None:
        _seed_everything(seed + rank())  # DDP copies process 0's initial weights to the others
    elif world > 1:
        torch.seed()  # torch's default seed is the same constant in every process: draw distinct augmentations

    dev = resolve_device(device)
    lo_epochs = epochs - hires_epochs  # the main schedule spans these
    hires_lr = lr0 / 5 if hires_lr is None else hires_lr

    # Speed knobs (no accuracy cost): TF32 matmul/conv on Ampere+, and cudnn
    # autotuner — safe because the input is a fixed-size square every step.
    if dev.type == "cuda":
        if tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # Gradient accumulation: reach an effective (nominal) batch of ``nominal_batch``
    # without the memory of a true large batch. EoMT is a ViT (LayerNorm, no
    # BatchNorm) so accumulating ``accum`` micro-batches is ~equivalent to one large
    # batch. ``accum`` overrides ``nominal_batch`` when > 0. On several GPUs each
    # accumulates its own micro-batches and the gradients are averaged over the GPUs.
    if accum > 0:
        accum_steps = accum
    elif nominal_batch > 0:
        accum_steps = max(1, round(nominal_batch / (batch * world)))
    else:
        accum_steps = 1
    eff_batch = batch * accum_steps * world
    gpus_str = f" x {world} GPUs" if world > 1 else ""
    print(f"[batch] micro={batch} x accum={accum_steps}{gpus_str} -> effective {eff_batch}")

    # Resume may point at a checkpoint file OR a run/weights folder. Resolve it
    # and recover the model size/imgsz from its metadata BEFORE building the model
    # so the architecture matches the saved weights regardless of --size.
    if resume and init_weights:
        raise ValueError(
            "pass either --resume (continue a run) or --weights (warm-start a fresh "
            "run from a checkpoint), not both."
        )
    resume_ckpt = None
    if resume:
        resume_path = resolve_checkpoint(resume, prefer="last")
        resume_ckpt = load_raw(resume_path)
        size = resume_ckpt.get("size", size)
        family = resume_ckpt.get("task", family)
        if resume_ckpt.get("imgsz"):
            imgsz = int(resume_ckpt["imgsz"])
        print(f"[resume] checkpoint {resume_path} (size={size}, family={family}, imgsz={imgsz})")

    # Warm start (fine-tune): recover the architecture from the checkpoint here so
    # the model is built to match, but the trainer state (epoch/optimizer/EMA/best)
    # is NOT restored below — training begins fresh at epoch 0.
    init_ckpt = None
    if init_weights:
        init_path = resolve_checkpoint(init_weights, prefer="best")
        init_ckpt = load_raw(init_path)
        size = init_ckpt.get("size", size)
        family = init_ckpt.get("task", family)
        if init_ckpt.get("imgsz"):
            imgsz = int(init_ckpt["imgsz"])
        print(
            f"[finetune] warm-start weights from {init_path} (size={size}, imgsz={imgsz}); "
            "fresh optimizer/LR schedule/EMA, training from epoch 0."
        )

    # Run directory. Default the name to ``eomt-<size>`` so different sizes don't
    # overwrite each other; resolved after resume so it tracks the checkpoint's size.
    if name is None:
        name = f"eomt-{size}"
    run_dir = Path(project) / name
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Secondary-head architecture. New runs use the CLI flags (a small MLP by
    # default). Resuming keeps the architecture recorded in the checkpoint so the
    # saved head weights load 1:1; checkpoints predating this metadata only ever
    # had single Linear heads, so default to that for them.
    aux_head_arch = {
        "layers": aux_head_layers,
        "hidden": aux_head_hidden,
        "dropout": aux_head_dropout,
    }
    if resume_ckpt is not None:
        aux_head_arch = resume_ckpt.get("aux_head_arch") or {"layers": 1}
    elif init_ckpt is not None:
        aux_head_arch = init_ckpt.get("aux_head_arch") or {"layers": 1}

    # Segmentation-loss weights + mask-head depth. New runs use the CLI flags.
    # Resuming MUST keep the checkpoint's objective and head shape (an inconsistent
    # objective or upscale depth would invalidate the optimizer/EMA state and break
    # the 1:1 weight load). Warm-starting (fine-tune) keeps the CLI loss weights —
    # the whole point is to retrain with a tuned objective — but inherits the
    # checkpoint's num_upscale_blocks because it is architecture-bound (the saved
    # upscale_block tensors must match the rebuilt head).
    is_detect = family == "detect"
    if is_detect:
        # Box head has no masks to focus attention on — masked-attention annealing
        # is inert; force it off so the (instance-only) buffer logic is skipped.
        mask_anneal = False
        cli_loss_weights = {
            "no_object_weight": no_object_weight,
            "class_weight": class_weight,
            "l1_weight": l1_weight,
            "giou_weight": giou_weight,
        }
    else:
        cli_loss_weights = {
            "no_object_weight": no_object_weight,
            "class_weight": class_weight,
            "mask_weight": mask_weight,
            "dice_weight": dice_weight,
            "l1_weight": l1_weight,
            "giou_weight": giou_weight,
            "iou_aware_cls": iou_aware_cls,
            "quality_weight": quality_weight,
        }
    loss_weights = normalize_loss_weights(cli_loss_weights, family=family)
    if resume_ckpt is not None:
        loss_weights = normalize_loss_weights(resume_ckpt.get("loss_weights"), family=family)
        num_upscale_blocks = resume_ckpt.get("num_upscale_blocks", num_upscale_blocks)
        # Resuming MUST keep the checkpoint's architecture (the saved FPN/cross-attn
        # tensors must match the rebuilt modules); the CLI flag/default is ignored.
        # An absent key means the checkpoint is single-scale — do NOT fall back to the
        # (now multi-scale) default, or resuming an old run would build a mismatched head.
        fpn_scales = resume_ckpt.get("fpn_scales")
        box_head = bool(resume_ckpt.get("aux_box_head"))
    elif init_ckpt is not None and init_ckpt.get("num_upscale_blocks") is not None:
        num_upscale_blocks = int(init_ckpt["num_upscale_blocks"])
    # Warm-start (init_weights) deliberately keeps the CLI ``fpn_scales``: enabling
    # multi-scale by fine-tuning a single-scale checkpoint is the intended workflow —
    # the new FPN/cross-attn modules init random (shape-mismatched/missing tensors are
    # skipped by the warm-start loader below) while the ViT + heads load 1:1.

    # Multi-photo post-training: the architecture of the checkpoint it starts from, kept as trained.
    mv = src_ckpt = None
    if multiview is not None:
        mv = {**DEFAULT_MULTIVIEW, **multiview}
        src_ckpt = resume_ckpt if resume_ckpt is not None else init_ckpt
        if src_ckpt is None or is_detect:
            raise ValueError("multiview post-trains an instance-segmentation checkpoint: pass init_weights (or resume).")
        fpn_scales = src_ckpt.get("fpn_scales")
        box_head = bool(src_ckpt.get("aux_box_head"))
        loss_weights = normalize_loss_weights(src_ckpt.get("loss_weights"), family=family)
        if src_ckpt.get("xview"):  # post-training a post-trained checkpoint (stage B): its modules load 1:1
            mv["xview"], mv["match_dim"] = src_ckpt["xview"], int(src_ckpt.get("match_dim") or 0)
        if mask_anneal:
            print("[xview] masked attention stays off: the checkpoint runs mask-free")
            mask_anneal = False
        if inference is None:  # the predict / track defaults the checkpoint was saved with
            inference = src_ckpt.get("inference")

    # --- data ---
    aug_cfg = AugConfig.resolve(aug, flip_prob=flip_prob, min_scale=min_scale, max_scale=max_scale)
    active = aug_cfg.active()
    if is_detect:
        if aug_cfg.mosaic_prob or aug_cfg.mixup_prob or aug_cfg.copy_paste_prob:
            warnings.warn("mosaic / mixup / copy-paste are instance-family augmentations; ignored for family='detect'.", stacklevel=2)
        active = [a for a in active if a not in ("mosaic", "mixup", "copy_paste")]
    if train_transform is not None:
        print("[aug] using the custom train_transform; the `aug` settings are ignored")
    else:
        masks = "" if is_detect else f"; masks: {aug_cfg.mask_resize}"
        print(f"[aug] active: {', '.join(active)}; LSJ {aug_cfg.min_scale}-{aug_cfg.max_scale}{masks}")
    Dataset = CocoDetection if is_detect else CocoInstanceSeg
    if keep_empty and is_detect:
        raise ValueError("keep_empty=True is only supported for family='instance'.")
    if mv is not None:  # sets of photos made from single photos, with the checkpoint's attribute heads only
        ck_heads = [h["name"] for h in src_ckpt.get("aux_heads") or []]
        # how each head learns (crop, ...): the run config's `attributes` where it sets it, else the data's
        train_ds = CocoViewSets(train_images, train_json, imgsz=imgsz, aug=aug_cfg, sets=mv["sets"],
                                attributes={n: attributes.get(n) for n in ck_heads} if attributes else ck_heads)
        _adopt_checkpoint_aux(train_ds, src_ckpt)
        print(f"[data] sets of up to {train_ds.sets.max_views} photos from single photos: {train_ds.sets}")
    else:
        train_ds = Dataset(
            train_images,
            train_json,
            imgsz=imgsz,
            transform=train_transform,
            aug=aug_cfg,
            attributes=True if attributes is None else attributes,
            **({"keep_empty": True} if keep_empty else {}),
        )
    nc, names = train_ds.num_classes, train_ds.names
    aux_specs = train_ds.aux_specs
    n_neg = sum(1 for i in train_ds.ids if not train_ds.coco.getAnnIds(imgIds=i, iscrowd=False)) if keep_empty else 0
    print(f"[data] train: {len(train_ds)} images, {nc} classes" + (f" (incl. {n_neg} negatives)" if keep_empty else ""))
    if aux_specs:
        cov = _aux_coverage(train_ds)
        print(
            "[data] aux heads: "
            + "; ".join(
                f"{s.name}({_aux_kind_str(s)}){_aux_scope_str(s, train_ds.names)}: "
                f"{cov[s.name][0]}/{cov[s.name][1]} tagged"
                for s in aux_specs
            )
        )
    def make_loader(ds, micro_batch: int):
        # Several GPUs: each process draws a disjoint share of every epoch's shuffle (same seed and epoch everywhere).
        smp = DistributedSampler(ds, shuffle=True, seed=seed or 0, drop_last=True) if world > 1 else None
        return smp, DataLoader(
            ds,
            batch_size=micro_batch,
            shuffle=smp is None,
            sampler=smp,
            num_workers=workers,
            collate_fn=collate_sets if mv is not None else collate_train,
            pin_memory=True,
            drop_last=True,
            persistent_workers=workers > 0,
            prefetch_factor=prefetch if workers > 0 else None,
        )

    def make_val(size: int):
        if not (val_images and val_json):
            return None, None
        vds = CocoValImages(val_images, val_json, imgsz=size, letterbox=letterbox)
        print(f"[data] val:   {len(vds)} images" + (f" at {size} px" if size != imgsz else ""))
        # Held-out aux accuracy needs GT geometry/attrs in the train id space: a
        # deterministic (no-crop) view of the val split, sharing train's maps.
        vseg = Dataset(
            val_images,
            val_json,
            imgsz=size,
            transform=build_val_transform(size, letterbox=letterbox),
            shared_aux=(aux_specs, train_ds._attr_id_maps),
        ) if aux_specs else None
        return vds, vseg

    sampler, train_loader = make_loader(train_ds, batch)
    val_size = val_imgsz or imgsz
    val_ds, val_seg_ds = make_val(val_size)
    val_sets_ds = None
    if mv is not None and val_ds is not None:  # the same sets at every validation, none of a single photo
        val_sets_ds = CocoViewSets(val_images, val_json, imgsz=val_size, aug=aug_cfg,
                                   sets={**mv["sets"], "p_single": 0.0}, seed=0,
                                   shared_aux=(aux_specs, train_ds._attr_id_maps))
        print(f"[data] val sets: {len(val_sets_ds)} fixed sets (one per val image)")

    # Optional per-class CE weights for imbalanced attributes (computed once).
    aux_class_weight = _compute_aux_class_weights(train_ds) if (aux_specs and aux_class_weights) else None

    # Record how this run was trained (model size, hyper-parameters, augmentation, how each head learns, inference
    # settings): args.yaml, and every checkpoint as `run_config`.
    run_cfg = {
        "size": size,
        "family": family,
        "imgsz": imgsz,
        "val_imgsz": val_size,
        "nc": nc,
        "train_images": train_images,
        "train_json": train_json,
        "val_images": val_images,
        "val_json": val_json,
        "epochs": epochs,
        "gpus": world,
        "batch": batch,
        "accum": accum_steps,
        "effective_batch": eff_batch,
        "lr0": lr0,
        "weight_decay": weight_decay,
        "backbone_lr_mult": backbone_lr_mult,
        "llrd": llrd,
        "poly_power": poly_power,
        "warmup_steps": list(warmup_steps),
        "clip_norm": clip_norm,
        "ema": ema,
        "ema_decay": ema_decay,
        "ema_tau": ema_tau,
        "amp": amp,
        "amp_dtype": amp_dtype,
        "tf32": tf32,
        "seed": seed,
        "pretrained": pretrained,
        "mask_anneal": mask_anneal,
        "mask_anneal_start": list(mask_anneal_start),
        "mask_anneal_end": list(mask_anneal_end),
        "flip_prob": aug_cfg.flip_prob,
        "min_scale": aug_cfg.min_scale,
        "max_scale": aug_cfg.max_scale,
        "aug": aug_cfg.to_dict() if train_transform is None else "custom train_transform",
        "keep_empty": keep_empty,
        "letterbox": letterbox,
        "val_interval": val_interval,
        "conf_thres": conf_thres,
        "max_det": max_det,
        "aux_w": aux_w,
        "aux_w_warmup": aux_w_warmup,
        "aux_w_per_head": aux_w_per_head,
        "aux_class_weights": aux_class_weights,
        "aux_iou_gate": aux_iou_gate,
        "aux_class_gate": aux_class_gate,
        "aux_balance": aux_balance,
        "aux_head_arch": aux_head_arch,
        "aux_heads": {s.name: ("regression" if s.regression else s.num_classes) for s in aux_specs},
        "loss_weights": loss_weights,
        "num_upscale_blocks": num_upscale_blocks,
        "fpn_scales": list(fpn_scales) if fpn_scales else None,
        "box_head": box_head,
        "deep_supervision": deep_supervision,
        "hires_epochs": hires_epochs,
        "hires_imgsz": hires_imgsz,
        "hires_batch": hires_batch,
        "hires_lr": hires_lr if hires_epochs else None,
        "hires_aug": hires_aug,
        "multiview": mv,
        "attributes": {s.name: _head_record(s, names) for s in aux_specs},
        "inference": inference,
    }
    _write_run_config(run_dir / "args.yaml", run_cfg)

    # --- model ---
    model = build_model(
        size,
        nc=nc,
        imgsz=imgsz,
        names=names,
        family=family,
        aux_heads=aux_specs,
        aux_head_arch=aux_head_arch,
        loss_weights=loss_weights,
        num_upscale_blocks=num_upscale_blocks,
        fpn_scales=fpn_scales,
        aux_box_head=box_head,
        xview=mv["xview"] if mv is not None else None,
        match_dim=mv["match_dim"] if mv is not None else 0,
    ).to(dev)
    model.eomt.deep_supervision = bool(deep_supervision)
    print(f"[model] loss weights: {loss_weights}; upscale blocks: {model.num_upscale_blocks}")
    if model.fpn_scales:
        print(f"[model] multi-scale FPN enabled: scales={model.fpn_scales}")
    if model.aux_box_head:
        print(f"[model] auxiliary box head: L1 {loss_weights['l1_weight']} / GIoU {loss_weights['giou_weight']} (also in the matching cost)")
    if loss_weights.get("quality_weight"):
        print(f"[model] mask-quality head (predicted IoU scores the masks): weight {loss_weights['quality_weight']}")
    if loss_weights.get("iou_aware_cls"):
        print("[model] IoU-aware class targets: a matched query's target is (1 + mask IoU) / 2")
    if not deep_supervision:
        print("[model] deep supervision off: a query block is supervised only while it uses masked attention")
    if aux_specs:
        print(f"[model] aux head arch: {model.aux_head_arch}")
    start_epoch, best_metric, phase_best = 0, -1.0, -1.0
    # Uncertainty weighting: one learned log-variance per attribute head, outside the model so that inference
    # checkpoints stay as they are (saved with the trainer state in last.pt). Its own group: head LR, no decay.
    aux_log_vars = None
    if aux_balance == "uncertainty" and aux_specs:
        aux_log_vars = torch.nn.ParameterDict({s.name: torch.nn.Parameter(torch.zeros((), device=dev)) for s in aux_specs})

    def make_optimizer(lr: float):
        if mv is not None:
            opt = build_multiview_optimizer(model, lr, weight_decay, mv["trainable"])
        else:
            opt = build_optimizer(model, lr, weight_decay, backbone_lr_mult, llrd)
        if aux_log_vars is not None:
            opt.add_param_group(
                {"params": list(aux_log_vars.values()), "lr": lr, "initial_lr": lr, "weight_decay": 0.0, "is_backbone": False}
            )
        return opt

    optimizer = make_optimizer(lr0)
    if mv is not None:
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        print(f"[xview] {model.xview_cfg['mode']} cross-view modules, match head {model.match_dim}-d; training "
              f"{mv['trainable']} ({n_train:.1f} M parameters), the rest of the checkpoint is frozen")

    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model"], strict=False)
        if aux_log_vars is not None and resume_ckpt.get("aux_log_vars"):
            aux_log_vars.load_state_dict(resume_ckpt["aux_log_vars"])
        if "optimizer" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
        start_epoch = int(resume_ckpt.get("epoch", -1)) + 1
        best_metric = float(resume_ckpt.get("best_metric", -1.0))
        phase_best = float(resume_ckpt.get("phase_best", -1.0))
        print(f"[resume] at epoch {start_epoch} (best segm mAP {best_metric:.4f})")
    elif init_ckpt is not None:
        # Warm start: load weights only. Optimizer/EMA/epoch/best stay fresh, so the
        # LR schedule and EMA restart from these weights at epoch 0.
        #
        # "Lazy" load: drop any checkpoint tensor whose shape doesn't match the current
        # model so weights from a different num-classes (e.g. an 81-class COCO detect
        # head into a 2-class fuel head) still transfer the backbone/queries/decoder.
        # ``strict=False`` skips missing/unexpected keys but NOT shape mismatches, which
        # would otherwise raise; those skipped tensors keep their fresh init.
        cur = model.state_dict()
        ckpt_sd = init_ckpt["model"]
        filtered = {k: v for k, v in ckpt_sd.items() if k in cur and cur[k].shape == v.shape}
        skipped = [k for k, v in ckpt_sd.items() if k in cur and cur[k].shape != v.shape]
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        if skipped:
            print(f"[finetune] re-initialized {len(skipped)} shape-mismatched tensors: {skipped}")
        print(
            f"[finetune] loaded warm-start weights ({len(missing)} missing / "
            f"{len(unexpected)} unexpected keys); training from epoch 0."
        )
        if mv is not None and not any(k.startswith("eomt.xview.") for k in ckpt_sd):
            model.init_xview()  # each cross-view sublayer starts as a copy of its query block's attention
    elif pretrained:
        load_dinov2_backbone(model)
        model.to(dev)

    # ``geom_labels`` carries instance masks (instance family) or normalized
    # cxcywh boxes (detect family); route it to the matching forward kwarg.
    geom_key = "box_labels" if is_detect else "mask_labels"

    def micro_step(m, pixel_values, geom_labels, class_labels, aux_labels, aux_w_eff, probe=False, set_ids=None,
                   inst_ids=None):
        """One micro-batch's loss: the segmentation loss computed in the model plus the attribute heads' loss (and,
        for sets of photos, the matching head's, kept in ``out["match_loss"]``). With ``probe``, also each head's pull
        on the shared queries (see :func:`_aux_grad_ratios`)."""
        out = m(pixel_values, class_labels=class_labels, set_ids=set_ids, **{geom_key: geom_labels})
        loss, aux_indices, aux_gated, grads = out["loss"], None, None, None
        if aux_specs or "match_embed" in out:
            # Match once per step, then gate to well-localized (IoU) and
            # correctly-classified queries so the attribute trains only on
            # instances the detector actually got right. Reuse for accuracy.
            aux_indices = match_queries(m, out, geom_labels, class_labels)
            aux_gated = gate_indices(
                out, aux_indices, geom_labels, class_labels,
                iou_thr=aux_iou_gate, require_class=aux_class_gate,
            )
        if "match_embed" in out and set_ids is not None:
            out["match_loss"] = match_loss(out, aux_gated, inst_ids, set_ids, mv["match_tau"])
            loss = loss + mv["match_w"] * out["match_loss"]
        if aux_specs:
            a_loss, per_head = aux_loss(
                m, out, geom_labels, class_labels, aux_labels,
                weights=aux_w_per_head, indices=aux_gated,
                class_weights=aux_class_weight, log_vars=aux_log_vars,
            )
            if probe and out["last_hidden_state"].requires_grad:  # frozen queries (a photo alone, stage A): no pull
                grads = _aux_grad_ratios(out, per_head, aux_w_eff, aux_w_per_head, aux_log_vars)
            loss = loss + aux_w_eff * a_loss
        if not loss.requires_grad:  # nothing that trains took part (a photo alone, stage A): a zero step, so the
            # backward (and on several GPUs its gradient averaging) still runs
            loss = loss + sum(p.sum() * 0.0 for p in m.parameters() if p.requires_grad)
        return loss, out, aux_indices, aux_gated, grads

    # Precision: bf16 (Ampere+/Blackwell) needs no GradScaler and is more stable
    # than fp16; "auto" picks bf16 when supported, else fp16. The GradScaler is
    # only ever enabled for fp16 — with bf16 its scale/unscale/step/update calls
    # below become no-ops, leaving the loop logic unchanged.
    amp_on = amp and dev.type == "cuda"
    if not amp_on:
        autocast_dtype, use_fp16 = torch.float32, False
    elif amp_dtype == "fp16":
        autocast_dtype, use_fp16 = torch.float16, True
    elif amp_dtype == "bf16":
        autocast_dtype, use_fp16 = torch.bfloat16, False
    else:  # "auto"
        if torch.cuda.is_bf16_supported():
            autocast_dtype, use_fp16 = torch.bfloat16, False
        else:
            autocast_dtype, use_fp16 = torch.float16, True
    print(f"[amp] {'off' if not amp_on else str(autocast_dtype).split('.')[-1]}"
          f"{' (+GradScaler)' if use_fp16 else ''}")

    # High-resolution phase: its micro-batch, checked on every GPU before anything trains (here, before DDP: a probe
    # that runs out of memory on one GPU must not leave the others waiting in a gradient all-reduce).
    hires_b = hires_batch or batch
    if hires_epochs and start_epoch < epochs:
        hires_train_ds = train_ds.at_imgsz(hires_imgsz, aug=AugConfig.resolve(aug_cfg, **(hires_aug or {})))
        stored = resume_ckpt.get("hires_batch") if resume_ckpt is not None else None
        if dev.type == "cuda":
            # the images with the most instances: the loss's memory grows with them
            busiest = sorted(range(len(train_ds)), key=lambda i: -len(train_ds.coco.imgToAnns.get(train_ds.ids[i], [])))
            # what the probe step does not allocate yet: the AdamW moments, DDP's gradient buckets, the EMA copy
            extra = sum(p.numel() * p.element_size() for p in model.parameters()) * (2 + (world > 1) + bool(ema))

            def hires_fits(b: int) -> bool:
                """One real training micro-step at ``hires_imgsz``, micro-batch ``b``, mask-free, on this GPU: does it fit
                in the budget with ``extra`` added?"""
                pv, geom, cls, attrs = collate_train([hires_train_ds[i] for i in busiest[:b]])
                pv, geom, cls = pv.to(dev), [g.to(dev) for g in geom], [c.to(dev) for c in cls]
                attrs = {k: [t.to(dev) for t in v] for k, v in attrs.items()}
                saved, loss, peak = model.eomt.attn_mask_probs.clone(), None, None
                model.eomt.attn_mask_probs.zero_()
                model.train()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(dev)
                try:
                    with torch.amp.autocast("cuda", enabled=amp_on, dtype=autocast_dtype):
                        loss = micro_step(model, pv, geom, cls, attrs, aux_w)[0]
                    loss.backward()
                    peak = torch.cuda.max_memory_allocated(dev)
                except torch.cuda.OutOfMemoryError:
                    pass
                finally:
                    loss = None
                    model.zero_grad(set_to_none=True)
                    model.eomt.attn_mask_probs.copy_(saved)
                    torch.cuda.empty_cache()
                usable = torch.cuda.mem_get_info(dev)[0] + torch.cuda.memory_reserved(dev)
                ok = peak is not None and peak + extra <= HIRES_MEMORY_BUDGET * usable
                print(f"[hires] probe on GPU {dev.index}: micro-batch {b} at {hires_imgsz} px "
                      + ("runs out of memory" if peak is None else f"needs {(peak + extra) / 2**30:.1f} of {usable / 2**30:.1f} GiB")
                      + ("" if ok else f": over the {HIRES_MEMORY_BUDGET:.0%} budget"))
                return ok

            # a resumed run keeps its micro-batch: its LR schedule depends on it
            chosen = (stored if hires_fits(stored) else 0) if stored else _fit_batch(hires_b, hires_fits)
            if world > 1:
                t = torch.tensor([chosen], device=dev)
                dist.all_reduce(t, op=dist.ReduceOp.MIN)
                chosen = int(t)
            if not chosen:
                raise RuntimeError(f"hires phase: a micro-batch of {stored or 1} at {hires_imgsz} px does not fit on every GPU; "
                                   "free memory or lower hires_imgsz")
            hires_b = chosen
        else:
            hires_b = stored or hires_b
        if hires_b != (hires_batch or batch):
            print(f"[hires] micro-batch {hires_batch or batch} does not fit at {hires_imgsz} px: {hires_b}, more accumulation")

    # Several GPUs: DDP copies process 0's weights to the others here and averages the gradients in backward. The
    # buffers (the annealed masked-attention probabilities) are set identically on every process, not broadcast.
    step_model = _Forward(model, micro_step, aux_log_vars)
    if world > 1:
        step_model = DistributedDataParallel(
            step_model, device_ids=[dev.index], broadcast_buffers=False, find_unused_parameters=True
        )

    # EMA of the weights, built after init/resume (and DDP's copy) so it starts from real weights.
    # Restored from the checkpoint when resuming so the average isn't lost.
    ema_model = ModelEMA(model, decay=ema_decay, tau=ema_tau, device=dev) if ema else None
    if ema_model is not None and resume_ckpt is not None and resume_ckpt.get("ema"):
        ema_model.load_state_dict(resume_ckpt["ema"])
        print(f"[resume] restored EMA ({ema_model.updates} updates)")

    core = _unwrap(model)
    # NOTE: torch.compile is intentionally not used. The encoder's masked-attention
    # path has data-dependent control flow and shapes — ``_disable_attention_mask``
    # boolean-indexes a random subset of queries, and the per-block branches read the
    # ``attn_mask_probs`` buffer (mutated every step by mask annealing). Compiling it
    # produced constant graph breaks/recompiles: no net speedup and highly variable
    # step times. Keep the encoder eager.

    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    iters_per_epoch = max(1, len(train_loader))
    total_iters = lo_epochs * iters_per_epoch  # the main schedule spans the epochs before a high-resolution phase
    total_opt_steps = max(1, math.ceil(total_iters / accum_steps))
    # the current phase: its schedule (warmup, first micro-iteration and epoch), input size and micro-batch
    sched_warmup, phase_it0, phase_epoch0, hires_on, cur_imgsz, cur_batch = warmup_steps, 0, 0, False, imgsz, batch
    n_q_blocks = int(core.eomt.attn_mask_probs.numel())
    anneal_starts = _per_block(mask_anneal_start, n_q_blocks, "mask_anneal_start")
    anneal_ends = _per_block(mask_anneal_end, n_q_blocks, "mask_anneal_end")
    if main and val_ds is not None and mask_anneal:
        mask_free_epoch = max(anneal_ends) * lo_epochs
        windows = ", ".join(f"{a * lo_epochs:.1f}-{b * lo_epochs:.1f}" for a, b in zip(anneal_starts, anneal_ends))
        print(
            f"[val] every {val_interval} epoch(s) and after the last, in the training regime (as the official EoMT): "
            f"masked attention at the current annealing (query-block windows at epochs {windows}; mask-free from "
            f"epoch {mask_free_epoch:.1f})."
        )
        print(
            "[val] A model that still masks scores far lower without its masks, so a mask-free curve would follow the "
            f"annealing, not the model. best.pt runs mask-free: it is only chosen from epoch {mask_free_epoch:.1f} on, "
            "and every checkpoint is saved mask-free."
        )
        if stop_after_epochs is not None and stop_after_epochs < mask_free_epoch:
            print(f"[val] stop_after_epochs={stop_after_epochs} ends before the annealing: no best.pt, use last.pt")

    # shown in the log; a multi-photo post-training of the query blocks alone has no head group
    head_group = next((g for g in reversed(optimizer.param_groups) if not g.get("is_backbone")), optimizer.param_groups[-1])

    def lr_factors(it: int) -> tuple[float, float]:
        """(head, backbone) multiplier of each group's ``initial_lr`` at micro-iteration ``it``, on the schedule of the
        current phase."""
        return _poly_two_stage_factors((it - phase_it0) // accum_steps, total_opt_steps, sched_warmup, poly_power)

    tb = _Logger(logger, run_dir, name) if logger != "none" and main else None

    def _save(path: Path, *, state_dict, with_trainer_state: bool, at_imgsz: int = imgsz):
        """``at_imgsz``: save as a model of that input size (inference checkpoints of the high-resolution phase); the
        trainer state stays at ``imgsz``, the size the model is rebuilt at to resume."""
        extra = {}
        if with_trainer_state:
            extra = {
                "epoch": epoch,
                "best_metric": best_metric,
                "optimizer": optimizer.state_dict(),
            }
            if ema_model is not None:
                extra["ema"] = ema_model.state_dict()
            if aux_log_vars is not None:
                extra["aux_log_vars"] = aux_log_vars.state_dict()
            if hires_epochs:
                extra.update(phase_best=phase_best, hires_batch=hires_b)
        ckpt = wrap_checkpoint(
            _state_at_imgsz(state_dict, imgsz, at_imgsz, int(getattr(core, "patch_size", 14))),
            size=size,
            nc=nc,
            imgsz=at_imgsz,
            names=names,
            task=family,
            aux_heads=aux_specs,
            aux_head_arch=aux_head_arch,
            letterbox=letterbox,
            loss_weights=loss_weights,
            num_upscale_blocks=core.num_upscale_blocks,
            fpn_scales=core.fpn_scales,
            aux_box_head=core.aux_box_head,
            norm_mean=getattr(core, "pixel_mean", None),
            norm_std=getattr(core, "pixel_std", None),
            patch_size=getattr(core, "patch_size", None),
            inference=inference,
            run_config=run_cfg,
            xview=core.xview_cfg,
            match_dim=core.match_dim or None,
            match_thres=match_thres,
            **extra,
        )
        save_checkpoint(ckpt, path)

    # The matching head's similarity threshold for "same instance": the best-F1 one of the last validation sets.
    match_thres = None

    # Per-epoch metrics CSV (segmentation always; per aux head if present: its matched-query accuracy or MAE, and its
    # pull on the shared queries).
    metric_of = {s.name: _aux_metric(s) for s in aux_specs}
    csv_fields = ["epoch", "train/loss"]
    csv_fields += [f"train/{metric_of[s.name]}/{s.name}" for s in aux_specs]
    csv_fields += [f"train/aux_grad/{s.name}" for s in aux_specs]
    if val_ds is not None:
        if not is_detect:
            csv_fields += [f"val/segm/{k}" for k in _SEGM_KEYS]
        csv_fields += [f"val/bbox/{k}" for k in _SEGM_KEYS]
        csv_fields += [f"val/{metric_of[s.name]}/{s.name}" for s in aux_specs]
    if hires_epochs:
        csv_fields.append("imgsz")
    if mv is not None:
        csv_fields.insert(2, "train/match_loss")
        if val_sets_ds is not None:
            csv_fields += _sets_fields(list(val_sets_ds.sets.cases) + (["locate"] if val_sets_ds.sets.p_focus > 0 else []),
                                       aux_specs)
    csv_log = _CsvLogger(run_dir / "metrics.csv", csv_fields) if main else None

    # Per-primary aux accuracy is computed held-out in aux_evaluate when a val split
    # exists; otherwise fall back to accumulating it over the train epoch.
    track_pc_train = bool(aux_specs) and val_seg_ds is None

    # --- epochs ---
    for epoch in range(start_epoch, epochs):
        if hires_epochs and epoch >= lo_epochs and not hires_on:
            # The high-resolution phase. Entered at its first epoch it starts from the EMA weights (what best.pt
            # holds) with a fresh optimizer and EMA, as a warm-started fine-tune; a run resumed inside it already has
            # them back from last.pt.
            fresh = epoch == lo_epochs
            hires_on, cur_imgsz, cur_batch = True, hires_imgsz, hires_b
            phase_it0, phase_epoch0 = lo_epochs * iters_per_epoch, lo_epochs
            accum_steps = max(1, round(eff_batch / (cur_batch * world)))
            sampler, train_loader = make_loader(hires_train_ds, cur_batch)
            iters_per_epoch = max(1, len(train_loader))
            total_opt_steps = max(1, math.ceil(hires_epochs * iters_per_epoch / accum_steps))
            sched_warmup = HIRES_WARMUP_STEPS
            val_ds, val_seg_ds = make_val(hires_imgsz)
            core.eomt.attn_mask_probs.zero_()
            if fresh:
                if ema_model is not None:
                    core.load_state_dict(ema_model.module.state_dict())
                optimizer = make_optimizer(hires_lr)
                if ema:
                    ema_model = ModelEMA(model, decay=HIRES_EMA_DECAY, tau=HIRES_EMA_TAU, device=dev)
                phase_best = -1.0
            head_group = next(g for g in reversed(optimizer.param_groups) if not g.get("is_backbone"))
            print(f"[hires] epochs {lo_epochs}-{epochs - 1} at {hires_imgsz} px: micro-batch {cur_batch} x accum "
                  f"{accum_steps}{f' x {world} GPUs' if world > 1 else ''} -> effective {cur_batch * accum_steps * world}, "
                  f"LR {hires_lr:g} (warmup {HIRES_WARMUP_STEPS}), EMA {HIRES_EMA_DECAY}, mask-free"
                  + (f", aug overrides {hires_aug}" if hires_aug else "") + ("; from the EMA weights" if fresh else "; resumed"))
        step_model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)
        running = running_match = 0.0
        # matched-query accuracy per aux head, accumulated over the epoch
        aux_hits = {s.name: 0 for s in aux_specs}
        aux_tot = {s.name: 0 for s in aux_specs}
        # per-primary-class aux accuracy (only accumulated as the no-val fallback)
        aux_pc: dict[str, dict[int, list[int]]] = {s.name: {} for s in aux_specs}
        aux_grad: dict[str, list[float]] = {s.name: [] for s in aux_specs}  # the heads' pull, measured on the main process
        desc = f"epoch {epoch}/{epochs - 1}" + (f" @{cur_imgsz}" if hires_epochs else "")
        pbar = tqdm(train_loader, desc=desc, unit="batch", disable=not main)
        for step, batch_data in enumerate(pbar):
            # sets of photos also carry the set of each photo and the instance ids of its targets
            pixel_values, geom_labels, class_labels, aux_labels = batch_data[:4]
            set_ids, inst_ids = batch_data[4:] if mv is not None else (None, None)
            it = phase_it0 + (epoch - phase_epoch0) * iters_per_epoch + step
            head_factor, vit_factor = lr_factors(it)
            for g in optimizer.param_groups:
                g["lr"] = g["initial_lr"] * (vit_factor if g.get("is_backbone") else head_factor)

            if mask_anneal and not hires_on:
                probs = _attn_mask_probs(it, total_iters, anneal_starts, anneal_ends, poly_power)
                core.eomt.set_attn_mask_probs(probs)

            pixel_values = pixel_values.to(dev)
            geom_labels = [g.to(dev) for g in geom_labels]
            class_labels = [c.to(dev) for c in class_labels]
            aux_labels = {k: [t.to(dev) for t in v] for k, v in aux_labels.items()}

            # Gradient accumulation: zero grads at the start of each window, divide
            # the loss by ``accum_steps`` (so summed micro-batch grads average), and
            # only step/clip/update on the window boundary (or the epoch's last batch).
            # On several GPUs the gradients are averaged over them on that boundary only.
            if step % accum_steps == 0:
                optimizer.zero_grad(set_to_none=True)
            boundary = (step + 1) % accum_steps == 0 or (step + 1) == iters_per_epoch
            aux_w_eff = aux_w * _aux_w_factor(it, total_iters, aux_w_warmup)
            no_sync = step_model.no_sync() if world > 1 and not boundary else contextlib.nullcontext()
            with no_sync:
                with torch.amp.autocast("cuda", enabled=amp_on, dtype=autocast_dtype):
                    probe = bool(aux_specs) and main and it % AUX_GRAD_EVERY == 0
                    loss, out, aux_indices, aux_gated, grads = step_model(
                        pixel_values, geom_labels, class_labels, aux_labels, aux_w_eff, probe, set_ids, inst_ids
                    )
                for n, r in (grads or {}).items():
                    aux_grad[n].append(r)
                loss_item = float(loss.detach())  # unscaled, for logging
                if "match_loss" in out:
                    running_match += float(out["match_loss"].detach())
                scaler.scale(loss / accum_steps).backward()
            if boundary:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(core.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
                if ema_model is not None:
                    ema_model.update(model)

            if aux_specs:
                for name, (hit, tot) in aux_accuracy(
                    model, out, geom_labels, class_labels, aux_labels, indices=aux_gated
                ).items():
                    aux_hits[name] += hit
                    aux_tot[name] += tot
                if track_pc_train:  # per-primary diagnostic when there is no val set
                    iou_only = gate_indices(
                        out, aux_indices, geom_labels, class_labels,
                        iou_thr=aux_iou_gate, require_class=False,
                    )
                    for name, buckets in aux_accuracy_by_primary(
                        model, out, geom_labels, class_labels, aux_labels, indices=iou_only
                    ).items():
                        for cls_id, (c, t) in buckets.items():
                            acc = aux_pc[name].setdefault(cls_id, [0, 0])
                            acc[0] += c
                            acc[1] += t

            running += loss_item
            avg = running / (step + 1)
            postfix = {
                "loss": f"{loss_item:.3f}",
                "avg": f"{avg:.3f}",
                "lr": f"{head_group['lr']:.2e}",
            }
            if aux_specs:  # running matched-query accuracy (MAE for a regression head) per head
                postfix["aux"] = " ".join(
                    f"{n}={_safe_acc(aux_hits[n], aux_tot[n]):.2f}" for n in aux_hits
                )
            if mv is not None:
                postfix["match"] = f"{running_match / (step + 1):.3f}"
            pbar.set_postfix(postfix)
            if tb is not None and step % 20 == 0:
                scalars = {"train/loss": loss_item, "lr/head": head_group["lr"]}
                if mask_anneal:
                    scalars["train/attn_mask_prob"] = float(core.eomt.attn_mask_probs[0])
                tb.log(scalars, it)

        # Several GPUs: the epoch's loss and accuracy counts summed over all of them, on the main process.
        totals = sum_over_processes((running, running_match, aux_hits, aux_tot, aux_pc))
        if main:
            running, running_match, aux_hits, aux_tot, aux_pc = totals
            epoch_loss = running / (iters_per_epoch * world)
            epoch_match = running_match / (iters_per_epoch * world)
            print(f"[epoch {epoch}] mean loss {epoch_loss:.4f}"
                  + (f" (matching head {epoch_match:.4f})" if mv is not None else ""))
            aux_acc = {n: _safe_acc(aux_hits[n], aux_tot[n]) for n in aux_hits}
            aux_pull = {n: sum(v) / len(v) if v else float("nan") for n, v in aux_grad.items()}
            if aux_specs:
                acc_str = "  ".join(f"{n} {metric_of[n][4:]}={aux_acc[n]:.3f}" for n in aux_acc)
                print(f"[epoch {epoch}] aux train: {acc_str}")
                print(
                    f"[epoch {epoch}] aux pull on the queries (gradient norm / segmentation's): "
                    + "  ".join(f"{n}={g:.3g}" for n, g in aux_pull.items())
                )
                if aux_log_vars is not None:
                    print(
                        f"[epoch {epoch}] aux learned weights: "
                        + "  ".join(f"{n}={math.exp(-float(v)):.3g}" for n, v in aux_log_vars.items())
                    )
                if tb is not None:
                    tb.log({f"train/{metric_of[n]}/{n}": aux_acc[n] for n in aux_acc}, epoch)
                    tb.log({f"train/aux_grad/{n}": g for n, g in aux_pull.items()}, epoch)

        # Validate in the regime the model is trained in (official EoMT): masked attention at the annealing
        # probabilities of the epoch's end. The EMA copy is what gets validated and exported as best.pt.
        # When annealing is disabled, leave the buffers as they are.
        eval_model = ema_model.module if ema_model is not None else core
        val_probs = [0.0] * n_q_blocks if hires_on else _attn_mask_probs(
            (epoch + 1) * iters_per_epoch, total_iters, anneal_starts, anneal_ends, poly_power)
        if mask_anneal:
            eval_model.eomt.set_attn_mask_probs(val_probs)
        # best.pt must run mask-free (as predict/val run it), so it is only chosen once the annealing is over.
        deployable = not mask_anneal or not any(val_probs)

        # --- validation (several GPUs: each runs its share, the main process gets the metrics) ---
        metrics = {}
        epoch_aux_pc: dict[str, dict[int, tuple[int, int]]] | None = None
        last_epoch = epoch + 1 in (epochs, stop_after_epochs)
        # A block mid-window unmasks a random share of its queries: seed it so every validation draws the same
        # queries, without touching the training RNG. The high-resolution phase is short: every epoch of it, and the
        # last one before it, are validated.
        if val_ds is not None and ((epoch + 1) % val_interval == 0 or last_epoch or hires_on or epoch + 1 == lo_epochs):
            with torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []):
                torch.manual_seed(0)
                if is_detect:
                    metrics = evaluate_detection(
                        eval_model, val_ds, device=dev, batch_size=cur_batch,
                        num_workers=workers, conf_thres=conf_thres, max_det=max_det,
                        amp=amp, verbose=main,
                    )
                    print(
                        f"[epoch {epoch}] bbox mAP {metrics.get('bbox/mAP', 0):.4f} "
                        f"mAP50 {metrics.get('bbox/mAP50', 0):.4f}"
                    )
                else:
                    metrics = evaluate(
                        eval_model, val_ds, device=dev, batch_size=cur_batch,
                        num_workers=workers, conf_thres=conf_thres, max_det=max_det,
                        amp=amp, verbose=main,
                    )
                    print(
                        f"[epoch {epoch}] segm mAP {metrics.get('segm/mAP', 0):.4f} "
                        f"mAP50 {metrics.get('segm/mAP50', 0):.4f} "
                        f"bbox mAP {metrics.get('bbox/mAP', 0):.4f}"
                    )
                if val_seg_ds is not None:  # held-out matched-query accuracy per head
                    val_aux, epoch_aux_pc = aux_evaluate(
                        eval_model, val_seg_ds, device=dev, batch_size=cur_batch,
                        num_workers=workers, amp=amp, verbose=main,
                        iou_gate=aux_iou_gate, class_gate=aux_class_gate,
                    )
                    metrics.update({f"{metric_of[n]}/{n}": v for n, v in val_aux.items()})
                    print(
                        f"[epoch {epoch}] val aux: "
                        + "  ".join(f"{n} {metric_of[n][4:]}={v:.3f}" for n, v in val_aux.items())
                    )
                if val_sets_ds is not None:  # the hard photo of each fixed set, alone and with its set
                    metrics.update(evaluate_sets(eval_model, val_sets_ds, device=dev, num_workers=workers, amp=amp,
                                                 verbose=main))
                    if main:
                        print(
                            f"[epoch {epoch}] val sets: hard photo segm mAP {metrics['sets/segm/mAP_alone']:.4f} alone"
                            f" -> {metrics['sets/segm/mAP_set']:.4f} with its set; links F1 "
                            f"{metrics['sets/match/f1']:.3f} at {metrics['sets/match/thres']:.2f} "
                            f"(re-ID of the photos alone {metrics['sets/reid/f1']:.3f})"
                        )
                        if math.isfinite(metrics["sets/match/thres"]):
                            match_thres = metrics["sets/match/thres"]
                if tb is not None:
                    tb.log({f"val/{k}": v for k, v in metrics.items()}, epoch)

        # Checkpoints are saved mask-free, as predict/val run them; the next step sets the training probabilities again.
        if mask_anneal:
            core.eomt.attn_mask_probs.zero_()
            eval_model.eomt.attn_mask_probs.zero_()

        if main:  # the main process alone writes the checkpoints, the metrics row and the plots
            # --- checkpoints ---
            # last.pt holds the live weights + trainer state (optimizer/EMA) for resume;
            # best.pt holds the eval weights (EMA when enabled) for inference.
            # The best metric is updated BEFORE last.pt is written, so a resumed run starts from the right value
            # (it used to lag one epoch and could overwrite a better best.pt).
            best_key = "sets/segm/mAP_set" if val_sets_ds is not None else "bbox/mAP" if is_detect else "segm/mAP"
            cur = metrics.get(best_key, None)
            improved = deployable and cur is not None and cur > best_metric
            # with a high-resolution phase, each phase also keeps its own best (best_<imgsz>.pt)
            phase_improved = bool(hires_epochs) and deployable and cur is not None and cur > phase_best
            if improved:
                best_metric = cur
            if phase_improved:
                phase_best = cur
            _save(weights_dir / "last.pt", state_dict=core.state_dict(), with_trainer_state=True)
            if improved:
                _save(weights_dir / "best.pt", state_dict=eval_model.state_dict(), with_trainer_state=False, at_imgsz=cur_imgsz)
                print(f"[epoch {epoch}] new best {best_key} {best_metric:.4f} -> best.pt")
            if phase_improved:
                _save(weights_dir / f"best_{cur_imgsz}.pt", state_dict=eval_model.state_dict(), with_trainer_state=False,
                      at_imgsz=cur_imgsz)
                print(f"[epoch {epoch}] new best {best_key} at {cur_imgsz} px {phase_best:.4f} -> best_{cur_imgsz}.pt")

            # --- per-epoch metrics row (missing values -> nan) ---
            row = {"epoch": epoch, "train/loss": epoch_loss, "train/match_loss": epoch_match, "imgsz": cur_imgsz}
            row.update({f"train/{metric_of[n]}/{n}": aux_acc[n] for n in aux_acc})
            row.update({f"train/aux_grad/{n}": g for n, g in aux_pull.items()})
            row.update({f"val/{k}": v for k, v in metrics.items()})
            csv_log.append(row)

            # --- per-epoch plots (overwrite each round; never fatal) ---
            try:
                plot_metrics_csv(run_dir / "metrics.csv", run_dir / "metrics.png")
            except Exception as e:  # noqa: BLE001 - plotting must never crash training
                print(f"[plot] metrics.png failed: {e}")
            if aux_specs:
                if epoch_aux_pc is None and track_pc_train:
                    epoch_aux_pc = {
                        n: {c: (v[0], v[1]) for c, v in buckets.items()}
                        for n, buckets in aux_pc.items()
                    }
                if epoch_aux_pc is not None:
                    try:  # accuracy bars for the categorical heads, MAE bars for the regression ones
                        plot_aux_per_class(
                            epoch_aux_pc, names, run_dir / "aux_per_class.png",
                            regression={n for n in epoch_aux_pc if metric_of[n] == "aux_mae"},
                        )
                    except Exception as e:  # noqa: BLE001
                        print(f"[plot] aux_per_class.png failed: {e}")

        if stop_after_epochs is not None and epoch + 1 >= stop_after_epochs:
            print(f"[stop] stopping after {epoch + 1} epoch(s) as requested; the schedule spans {epochs}")
            break

    if tb is not None:
        tb.close()

    return {
        "best_metric": best_metric,
        "last": str(weights_dir / "last.pt"),
        "best": str(weights_dir / "best.pt") if (weights_dir / "best.pt").exists() else None,
        "weights_dir": str(weights_dir),
    }

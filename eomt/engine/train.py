"""Training loop for EoMT instance segmentation.

Initializes the encoder from DINOv2, fine-tunes with AdamW (a low LR multiplier
on the pretrained encoder, full LR on the mask/class head), a cosine schedule
with warmup, AMP and gradient clipping. The masked-attention probability is
annealed 1->0 over training (EoMT recipe) so the model converges to efficient
mask-free inference. Validates every ``val_interval`` epochs with COCO segm mAP
(in the mask-free regime) and keeps both ``last.pt`` and the ``best.pt`` (highest
``segm/mAP``). The EoMT segmentation loss is computed inside the HF model.
"""

from __future__ import annotations

import csv
import math
import warnings
from collections.abc import Sequence
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..aux_cls import (
    aux_accuracy,
    aux_accuracy_by_primary,
    aux_loss,
    gate_indices,
    match_queries,
)
from ..config import normalize_loss_weights
from ..data import CocoDetection, CocoInstanceSeg, CocoValImages, collate_train
from ..data.transforms import AugConfig, build_val_transform
from ..ema import ModelEMA, _unwrap
from ..model import DEFAULT_FPN_SCALES, build_model, load_dinov2_backbone
from ..plotting import plot_aux_per_class, plot_metrics_csv
from ..device import resolve_device
from ..serialization import load_raw, resolve_checkpoint, save_checkpoint, wrap_checkpoint
from .validate import _SEGM_KEYS, aux_evaluate, evaluate, evaluate_detection

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


def _compute_aux_class_weights(train_ds) -> dict:
    """Inverse-sqrt-frequency CE weights per aux head (mean-normalized to ~1).

    Counts each head's contiguous-class frequency from the in-memory train
    annotations (no image decode); missing / out-of-vocab values are ignored.
    """
    anns = getattr(train_ds.coco, "dataset", {}).get("annotations", []) or []
    cat2contig = train_ds.cat2contig
    out: dict[str, torch.Tensor] = {}
    for spec in train_ds.aux_specs:
        id_map = train_ds._attr_id_maps[spec.name]
        counts = torch.zeros(spec.num_classes)
        for a in anns:
            # Skip out-of-scope instances for class-scoped heads: they are routed to
            # -100 and never contribute to the head's loss, so they must not skew its
            # class-frequency weights either.
            if spec.applies_to is not None:
                cls = cat2contig.get(a.get("category_id"))
                if cls is None or cls not in spec.applies_to:
                    continue
            cid = id_map.get(a.get("attributes", {}).get(spec.name), -100)
            if 0 <= cid < spec.num_classes:
                counts[cid] += 1
        w = 1.0 / counts.clamp(min=1).sqrt()
        out[spec.name] = w * (spec.num_classes / w.sum())  # mean weight ~ 1.0
    return out


def _aux_coverage(train_ds) -> dict[str, tuple[int, int]]:
    """Per head ``{name: (supervised, in_scope)}`` instance counts from train annotations.

    ``in_scope`` counts instances whose primary class the head applies to (every
    instance for an unscoped head); ``supervised`` counts those of them carrying a
    tagged, in-vocab value (not ignored as ``-100``). Diagnostic only — mirrors the
    routing / ignore logic in :func:`eomt.data.coco._attr_label` without decoding images.
    """
    anns = getattr(train_ds.coco, "dataset", {}).get("annotations", []) or []
    cat2contig = train_ds.cat2contig
    out: dict[str, tuple[int, int]] = {}
    for spec in train_ds.aux_specs:
        id_map = train_ds._attr_id_maps[spec.name]
        supervised = in_scope = 0
        for a in anns:
            cls = cat2contig.get(a.get("category_id"))
            if cls is None:
                continue
            if spec.applies_to is not None and cls not in spec.applies_to:
                continue
            in_scope += 1
            if id_map.get(a.get("attributes", {}).get(spec.name), -100) >= 0:
                supervised += 1
        out[spec.name] = (supervised, in_scope)
    return out


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
    if training is interrupted before the first checkpoint.
    """
    import yaml

    with path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


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
    epochs: int = 50,
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
    val_interval: int = 1,
    conf_thres: float = 0.0,
    max_det: int = 100,
    aux_w: float = 1.0,
    aux_w_warmup: float = 0.0,
    aux_w_per_head: dict[str, float] | None = None,
    aux_class_weights: bool = False,
    aux_iou_gate: float = 0.5,
    aux_class_gate: bool = True,
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
    logger: str = "none",
    resume: str | None = None,
    init_weights: str | None = None,
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
    block masks, its prediction builds the attention mask and stays supervised (unsupervised, it made noise masks).
    With the default windows the blocks drop out at 2/6, 3/6, 4/6 and 5/6 of training (~40 % fewer intermediate
    predictions overall) and only the last sixth runs final-only (~25 % faster steps); the reference recipe
    supervises every block throughout. Resuming keeps the checkpoint's ``box_head``; warm starts honor the argument.

    **Class targets** (instance family; persisted with the loss weights). ``iou_aware_cls=True`` trains a matched query's
    class probability towards ``(1 + IoU) / 2`` of its mask rather than 1, so the score ranks masks by their quality.
    ``quality_weight > 0`` adds a mask-quality head that predicts each matched query's mask IoU (Mask Scoring R-CNN);
    the score at inference becomes class probability x predicted IoU.

    ``val_imgsz`` validates (and picks ``best.pt``) at another input size than ``imgsz``; ``None`` = ``imgsz``. Scale
    augmentation makes the objects the model trains on larger, on average, than they are in a val image resized to
    ``imgsz`` (``min_scale``..``max_scale`` of ``imgsz``); validating at ``imgsz`` times the mean training scale measures
    the model at the scale it learned.

    ``stop_after_epochs`` ends the run once that many epochs have completed (counted from the start of the run,
    so it composes with ``resume``) while the LR schedule and mask annealing still span ``epochs``: it is for
    truncated A/B comparisons that must follow exactly the schedule of a full run.

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
    window), and the last sixth of training is mask-free. ``amp_dtype`` selects the autocast dtype:
    ``"auto"`` uses bf16 on hardware that supports it (no GradScaler, more stable),
    else fp16; force with ``"bf16"`` / ``"fp16"``.
    """
    if len(warmup_steps) != 2:
        raise ValueError("warmup_steps must be (head_steps, vit_steps)")
    if imgsz % 14 or (val_imgsz is not None and val_imgsz % 14):
        raise ValueError(f"imgsz={imgsz} and val_imgsz={val_imgsz} must be divisible by 14 (DINOv2 grid).")

    if seed is not None:
        _seed_everything(seed)

    dev = resolve_device(device)

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
    # batch. ``accum`` overrides ``nominal_batch`` when > 0.
    if accum > 0:
        accum_steps = accum
    elif nominal_batch > 0:
        accum_steps = max(1, round(nominal_batch / batch))
    else:
        accum_steps = 1
    eff_batch = batch * accum_steps
    print(f"[batch] micro={batch} x accum={accum_steps} -> effective {eff_batch}")

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
    train_ds = Dataset(
        train_images,
        train_json,
        imgsz=imgsz,
        transform=train_transform,
        aug=aug_cfg,
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
                f"{s.name}({s.num_classes}){_aux_scope_str(s, train_ds.names)}: "
                f"{cov[s.name][0]}/{cov[s.name][1]} tagged"
                for s in aux_specs
            )
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch,
        shuffle=True,
        num_workers=workers,
        collate_fn=collate_train,
        pin_memory=True,
        drop_last=True,
        persistent_workers=workers > 0,
        prefetch_factor=prefetch if workers > 0 else None,
    )

    val_ds = None
    val_seg_ds = None
    val_size = val_imgsz or imgsz
    if val_images and val_json:
        val_ds = CocoValImages(val_images, val_json, imgsz=val_size, letterbox=letterbox)
        print(f"[data] val:   {len(val_ds)} images" + (f" at {val_size} px" if val_size != imgsz else ""))
        # Held-out aux accuracy needs GT geometry/attrs in the train id space: a
        # deterministic (no-crop) view of the val split, sharing train's maps.
        if aux_specs:
            val_seg_ds = Dataset(
                val_images,
                val_json,
                imgsz=val_size,
                transform=build_val_transform(val_size, letterbox=letterbox),
                shared_aux=(aux_specs, train_ds._attr_id_maps),
            )

    # Optional per-class CE weights for imbalanced attributes (computed once).
    aux_class_weight = _compute_aux_class_weights(train_ds) if (aux_specs and aux_class_weights) else None

    # Record how this run was trained (model size + key hyper-parameters).
    _write_run_config(
        run_dir / "args.yaml",
        {
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
            "aux_head_arch": aux_head_arch,
            "aux_heads": {s.name: s.num_classes for s in aux_specs},
            "loss_weights": loss_weights,
            "num_upscale_blocks": num_upscale_blocks,
            "fpn_scales": list(fpn_scales) if fpn_scales else None,
            "box_head": box_head,
            "deep_supervision": deep_supervision,
        },
    )

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
    start_epoch, best_metric = 0, -1.0
    optimizer = build_optimizer(model, lr0, weight_decay, backbone_lr_mult, llrd)

    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model"], strict=False)
        if "optimizer" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
        start_epoch = int(resume_ckpt.get("epoch", -1)) + 1
        best_metric = float(resume_ckpt.get("best_metric", -1.0))
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
    elif pretrained:
        load_dinov2_backbone(model)
        model.to(dev)

    # EMA of the weights, built after init/resume so it starts from real weights.
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
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    iters_per_epoch = max(1, len(train_loader))
    total_iters = epochs * iters_per_epoch
    total_opt_steps = max(1, math.ceil(total_iters / accum_steps))
    n_q_blocks = int(core.eomt.attn_mask_probs.numel())
    anneal_starts = _per_block(mask_anneal_start, n_q_blocks, "mask_anneal_start")
    anneal_ends = _per_block(mask_anneal_end, n_q_blocks, "mask_anneal_end")

    head_group = next(g for g in reversed(optimizer.param_groups) if not g.get("is_backbone"))  # shown in the log

    def lr_factors(it: int) -> tuple[float, float]:
        """(head, backbone) multiplier of each group's ``initial_lr`` at micro-iteration ``it``."""
        return _poly_two_stage_factors(it // accum_steps, total_opt_steps, warmup_steps, poly_power)

    tb = _Logger(logger, run_dir, name) if logger != "none" else None

    def _save(path: Path, *, state_dict, with_trainer_state: bool):
        extra = {}
        if with_trainer_state:
            extra = {
                "epoch": epoch,
                "best_metric": best_metric,
                "optimizer": optimizer.state_dict(),
            }
            if ema_model is not None:
                extra["ema"] = ema_model.state_dict()
        ckpt = wrap_checkpoint(
            state_dict,
            size=size,
            nc=nc,
            imgsz=imgsz,
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
            **extra,
        )
        save_checkpoint(ckpt, path)

    # Per-epoch metrics CSV (segmentation always; one column per aux head if present).
    csv_fields = ["epoch", "train/loss"]
    csv_fields += [f"train/aux_acc/{s.name}" for s in aux_specs]
    if val_ds is not None:
        if not is_detect:
            csv_fields += [f"val/segm/{k}" for k in _SEGM_KEYS]
        csv_fields += [f"val/bbox/{k}" for k in _SEGM_KEYS]
        csv_fields += [f"val/aux_acc/{s.name}" for s in aux_specs]
    csv_log = _CsvLogger(run_dir / "metrics.csv", csv_fields)

    # Per-primary aux accuracy is computed held-out in aux_evaluate when a val split
    # exists; otherwise fall back to accumulating it over the train epoch.
    track_pc_train = bool(aux_specs) and val_seg_ds is None

    # --- epochs ---
    for epoch in range(start_epoch, epochs):
        model.train()
        running = 0.0
        # matched-query accuracy per aux head, accumulated over the epoch
        aux_hits = {s.name: 0 for s in aux_specs}
        aux_tot = {s.name: 0 for s in aux_specs}
        # per-primary-class aux accuracy (only accumulated as the no-val fallback)
        aux_pc: dict[str, dict[int, list[int]]] = {s.name: {} for s in aux_specs}
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs - 1}", unit="batch")
        for step, (pixel_values, geom_labels, class_labels, aux_labels) in enumerate(pbar):
            it = epoch * iters_per_epoch + step
            head_factor, vit_factor = lr_factors(it)
            for g in optimizer.param_groups:
                g["lr"] = g["initial_lr"] * (vit_factor if g.get("is_backbone") else head_factor)

            if mask_anneal:
                probs = _attn_mask_probs(it, total_iters, anneal_starts, anneal_ends, poly_power)
                core.eomt.set_attn_mask_probs(probs)

            pixel_values = pixel_values.to(dev)
            geom_labels = [g.to(dev) for g in geom_labels]
            class_labels = [c.to(dev) for c in class_labels]
            aux_labels = {k: [t.to(dev) for t in v] for k, v in aux_labels.items()}

            # Gradient accumulation: zero grads at the start of each window, divide
            # the loss by ``accum_steps`` (so summed micro-batch grads average), and
            # only step/clip/update on the window boundary (or the epoch's last batch).
            if step % accum_steps == 0:
                optimizer.zero_grad(set_to_none=True)
            aux_gated = None
            # ``geom_labels`` carries instance masks (instance family) or normalized
            # cxcywh boxes (detect family); route it to the matching forward kwarg.
            geom_kw = {"box_labels": geom_labels} if is_detect else {"mask_labels": geom_labels}
            with torch.amp.autocast("cuda", enabled=amp_on, dtype=autocast_dtype):
                out = model(pixel_values, class_labels=class_labels, **geom_kw)
                loss = out["loss"]
                if aux_specs:
                    # Match once per step, then gate to well-localized (IoU) and
                    # correctly-classified queries so the attribute trains only on
                    # instances the detector actually got right. Reuse for accuracy.
                    aux_indices = match_queries(model, out, geom_labels, class_labels)
                    aux_gated = gate_indices(
                        out, aux_indices, geom_labels, class_labels,
                        iou_thr=aux_iou_gate, require_class=aux_class_gate,
                    )
                    a_loss, _ = aux_loss(
                        model, out, geom_labels, class_labels, aux_labels,
                        weights=aux_w_per_head, indices=aux_gated,
                        class_weights=aux_class_weight,
                    )
                    aux_w_eff = aux_w * _aux_w_factor(it, total_iters, aux_w_warmup)
                    loss = loss + aux_w_eff * a_loss
            loss_item = float(loss.detach())  # unscaled, for logging
            scaler.scale(loss / accum_steps).backward()
            if (step + 1) % accum_steps == 0 or (step + 1) == iters_per_epoch:
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
            if aux_specs:  # running matched-query accuracy per head
                postfix["aux_acc"] = " ".join(
                    f"{n}={_safe_acc(aux_hits[n], aux_tot[n]):.2f}" for n in aux_hits
                )
            pbar.set_postfix(postfix)
            if tb is not None and step % 20 == 0:
                scalars = {"train/loss": loss_item, "lr/head": head_group["lr"]}
                if mask_anneal:
                    scalars["train/attn_mask_prob"] = float(core.eomt.attn_mask_probs[0])
                tb.log(scalars, it)

        epoch_loss = running / iters_per_epoch
        print(f"[epoch {epoch}] mean loss {epoch_loss:.4f}")
        aux_acc = {n: _safe_acc(aux_hits[n], aux_tot[n]) for n in aux_hits}
        if aux_specs:
            acc_str = "  ".join(f"{n}={aux_acc[n]:.3f}" for n in aux_acc)
            print(f"[epoch {epoch}] aux train acc: {acc_str}")
            if tb is not None:
                tb.log({f"train/aux_acc/{n}": aux_acc[n] for n in aux_acc}, epoch)

        # Evaluate and checkpoint in the deployment regime: masked attention off
        # (deterministic, mask-free) so val mAP and the saved buffer match how the
        # model is later run via predict/val. Training stochasticity resumes on the
        # next epoch's first step. When annealing is disabled, leave the buffer as is.
        # The EMA copy is what gets validated and exported as best.pt, so zero its
        # buffer too (its buffers track the live model, which was just annealing).
        eval_model = ema_model.module if ema_model is not None else core
        if mask_anneal:
            core.eomt.attn_mask_probs.zero_()
            if ema_model is not None:
                eval_model.eomt.attn_mask_probs.zero_()

        # --- validation ---
        metrics = {}
        epoch_aux_pc: dict[str, dict[int, tuple[int, int]]] | None = None
        if val_ds is not None and (epoch + 1) % val_interval == 0:
            if is_detect:
                metrics = evaluate_detection(
                    eval_model, val_ds, device=dev, batch_size=batch,
                    num_workers=workers, conf_thres=conf_thres, max_det=max_det,
                    amp=amp, verbose=True,
                )
                print(
                    f"[epoch {epoch}] bbox mAP {metrics.get('bbox/mAP', 0):.4f} "
                    f"mAP50 {metrics.get('bbox/mAP50', 0):.4f}"
                )
            else:
                metrics = evaluate(
                    eval_model, val_ds, device=dev, batch_size=batch,
                    num_workers=workers, conf_thres=conf_thres, max_det=max_det,
                    amp=amp, verbose=True,
                )
                print(
                    f"[epoch {epoch}] segm mAP {metrics.get('segm/mAP', 0):.4f} "
                    f"mAP50 {metrics.get('segm/mAP50', 0):.4f} "
                    f"bbox mAP {metrics.get('bbox/mAP', 0):.4f}"
                )
            if val_seg_ds is not None:  # held-out matched-query accuracy per head
                val_aux, epoch_aux_pc = aux_evaluate(
                    eval_model, val_seg_ds, device=dev, batch_size=batch,
                    num_workers=workers, amp=amp,
                    iou_gate=aux_iou_gate, class_gate=aux_class_gate,
                )
                metrics.update({f"aux_acc/{n}": v for n, v in val_aux.items()})
                print(
                    f"[epoch {epoch}] val aux acc: "
                    + "  ".join(f"{n}={v:.3f}" for n, v in val_aux.items())
                )
            if tb is not None:
                tb.log({f"val/{k}": v for k, v in metrics.items()}, epoch)

        # --- checkpoints ---
        # last.pt holds the live weights + trainer state (optimizer/EMA) for resume;
        # best.pt holds the eval weights (EMA when enabled) for inference.
        # The best metric is updated BEFORE last.pt is written, so a resumed run starts from the right value
        # (it used to lag one epoch and could overwrite a better best.pt).
        best_key = "bbox/mAP" if is_detect else "segm/mAP"
        cur = metrics.get(best_key, None)
        improved = cur is not None and cur > best_metric
        if improved:
            best_metric = cur
        _save(weights_dir / "last.pt", state_dict=core.state_dict(), with_trainer_state=True)
        if improved:
            _save(weights_dir / "best.pt", state_dict=eval_model.state_dict(), with_trainer_state=False)
            print(f"[epoch {epoch}] new best {best_key} {best_metric:.4f} -> best.pt")

        # --- per-epoch metrics row (missing values -> nan) ---
        row = {"epoch": epoch, "train/loss": epoch_loss}
        row.update({f"train/aux_acc/{n}": aux_acc[n] for n in aux_acc})
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
                try:
                    plot_aux_per_class(epoch_aux_pc, names, run_dir / "aux_per_class.png")
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

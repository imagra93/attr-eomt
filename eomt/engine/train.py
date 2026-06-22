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
from ..data import CocoInstanceSeg, CocoValImages, collate_train
from ..data.transforms import build_val_transform
from ..ema import ModelEMA, _unwrap
from ..model import build_model, load_dinov2_backbone
from ..plotting import plot_aux_per_class, plot_metrics_csv
from ..serialization import load_raw, resolve_checkpoint, save_checkpoint, wrap_checkpoint
from .validate import _SEGM_KEYS, aux_evaluate, evaluate

_ENCODER_PREFIXES = ("eomt.embeddings", "eomt.layers", "eomt.layernorm")
# Parameters excluded from weight decay regardless of group: all 1-D tensors
# (LayerNorm weights, every bias) plus token/positional embeddings — the standard
# ViT fine-tuning policy (decaying these hurts).
_NO_DECAY_TOKENS = ("position_embeddings", "cls_token", "register_tokens")


def _resolve_device(device: str) -> torch.device:
    if device in ("", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


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
    """Layer-wise LR decay factor for a backbone param (1.0 at the top layer).

    Maps embeddings→0, ``eomt.layers.i``→i+1, post-encoder norm→num_layers+1, then
    scales by ``llrd ** (depth from the top)`` so deeper (later) layers — closer to
    the task head — get a higher LR than the early, generic DINOv2 layers.
    """
    if llrd >= 1.0 or num_layers <= 0:
        return 1.0
    top = num_layers + 1
    if name.startswith("eomt.embeddings"):
        layer_id = 0
    elif name.startswith("eomt.layers."):
        layer_id = int(name.split("eomt.layers.")[1].split(".")[0]) + 1
    else:  # post-encoder layernorm / anything else in the backbone
        layer_id = top
    return llrd ** (top - layer_id)


def build_optimizer(
    model, lr: float, weight_decay: float, backbone_lr_mult: float, llrd: float = 1.0
):
    """AdamW with backbone LR scaling, layer-wise LR decay, and no-WD on norms/biases.

    Param groups are keyed by ``(lr, weight_decay)``: the DINOv2 encoder gets
    ``lr * backbone_lr_mult`` further scaled per layer by ``llrd`` (``1.0`` = the
    legacy flat multiplier); the task head gets full ``lr``. 1-D tensors and
    positional/token embeddings are placed in weight-decay-free groups.
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
        key = (round(lr_i, 12), wd_i)
        groups.setdefault(key, {"params": [], "lr": lr_i, "weight_decay": wd_i})["params"].append(p)
    opt = torch.optim.AdamW(list(groups.values()), lr=lr, weight_decay=weight_decay)
    for g in opt.param_groups:
        g["initial_lr"] = g["lr"]
    return opt


def _lr_factor(it, total_iters, warmup_iters, warmup_start_factor, min_ratio):
    if it < warmup_iters:
        return warmup_start_factor + (1.0 - warmup_start_factor) * it / max(1, warmup_iters)
    progress = (it - warmup_iters) / max(1, total_iters - warmup_iters)
    cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return min_ratio + (1.0 - min_ratio) * cos


def _attn_mask_prob(it, total_iters, start_frac, end_frac):
    """Masked-attention annealing schedule (EoMT, CVPR 2025).

    Returns the probability of applying masked attention to each query at
    iteration ``it``: held at 1.0, then linearly annealed to 0.0 across the
    ``[start_frac, end_frac]`` fraction of training so the final stretch trains
    mask-free and matches efficient (mask-less) inference.
    """
    start = start_frac * total_iters
    end = end_frac * total_iters
    if it <= start:
        return 1.0
    if it >= end:
        return 0.0
    return 1.0 - (it - start) / max(1.0, end - start)


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
    out: dict[str, torch.Tensor] = {}
    for spec in train_ds.aux_specs:
        id_map = train_ds._attr_id_maps[spec.name]
        counts = torch.zeros(spec.num_classes)
        for a in anns:
            cid = id_map.get(a.get("attributes", {}).get(spec.name), -100)
            if 0 <= cid < spec.num_classes:
                counts[cid] += 1
        w = 1.0 / counts.clamp(min=1).sqrt()
        out[spec.name] = w * (spec.num_classes / w.sum())  # mean weight ~ 1.0
    return out


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
            wandb.init(project="libre-eomt", name=run_name, dir=str(logdir))

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
    imgsz: int = 644,
    epochs: int = 50,
    batch: int = 4,
    nominal_batch: int = 16,
    accum: int = 0,
    lr0: float = 1e-4,
    weight_decay: float = 0.05,
    backbone_lr_mult: float = 0.1,
    llrd: float = 0.85,
    warmup_epochs: float = 1.0,
    warmup_lr_start: float = 1e-6,
    min_lr_ratio: float = 0.01,
    ema: bool = True,
    ema_decay: float = 0.9999,
    ema_tau: float = 2000.0,
    mask_anneal: bool = True,
    mask_anneal_start: float = 0.0,
    mask_anneal_end: float = 0.9,
    clip_norm: float = 0.01,
    workers: int = 8,
    prefetch: int = 4,
    device: str = "auto",
    amp: bool = True,
    tf32: bool = True,
    compile: bool = False,
    seed: int | None = None,
    pretrained: bool = True,
    flip_prob: float = 0.5,
    min_scale: float = 0.1,
    max_scale: float = 2.0,
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
    train_num_points: int = 12544,
    oversample_ratio: float = 3.0,
    importance_sample_ratio: float = 0.75,
    num_upscale_blocks: int | None = None,
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
    """
    if imgsz % 14:
        raise ValueError(f"imgsz={imgsz} must be divisible by 14 (DINOv2 grid).")

    if seed is not None:
        _seed_everything(seed)

    dev = _resolve_device(device)

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
        if resume_ckpt.get("imgsz"):
            imgsz = int(resume_ckpt["imgsz"])
        print(f"[resume] checkpoint {resume_path} (size={size}, imgsz={imgsz})")

    # Warm start (fine-tune): recover the architecture from the checkpoint here so
    # the model is built to match, but the trainer state (epoch/optimizer/EMA/best)
    # is NOT restored below — training begins fresh at epoch 0.
    init_ckpt = None
    if init_weights:
        init_path = resolve_checkpoint(init_weights, prefer="best")
        init_ckpt = load_raw(init_path)
        size = init_ckpt.get("size", size)
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
    loss_weights = normalize_loss_weights(
        {
            "no_object_weight": no_object_weight,
            "class_weight": class_weight,
            "mask_weight": mask_weight,
            "dice_weight": dice_weight,
            "train_num_points": train_num_points,
            "oversample_ratio": oversample_ratio,
            "importance_sample_ratio": importance_sample_ratio,
        }
    )
    if resume_ckpt is not None:
        loss_weights = normalize_loss_weights(resume_ckpt.get("loss_weights"))
        num_upscale_blocks = resume_ckpt.get("num_upscale_blocks", num_upscale_blocks)
    elif init_ckpt is not None and init_ckpt.get("num_upscale_blocks") is not None:
        num_upscale_blocks = int(init_ckpt["num_upscale_blocks"])

    # --- data ---
    train_ds = CocoInstanceSeg(
        train_images,
        train_json,
        imgsz=imgsz,
        flip_prob=flip_prob,
        min_scale=min_scale,
        max_scale=max_scale,
    )
    nc, names = train_ds.num_classes, train_ds.names
    aux_specs = train_ds.aux_specs
    print(f"[data] train: {len(train_ds)} images, {nc} classes")
    if aux_specs:
        print(
            "[data] aux heads: "
            + ", ".join(f"{s.name}({s.num_classes})" for s in aux_specs)
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
    if val_images and val_json:
        val_ds = CocoValImages(val_images, val_json, imgsz=imgsz, letterbox=letterbox)
        print(f"[data] val:   {len(val_ds)} images")
        # Held-out aux accuracy needs GT masks/attrs in the train id space: a
        # deterministic (no-crop) seg view of the val split, sharing train's maps.
        if aux_specs:
            val_seg_ds = CocoInstanceSeg(
                val_images,
                val_json,
                imgsz=imgsz,
                transform=build_val_transform(imgsz, letterbox=letterbox),
                shared_aux=(aux_specs, train_ds._attr_id_maps),
            )

    # Optional per-class CE weights for imbalanced attributes (computed once).
    aux_class_weight = _compute_aux_class_weights(train_ds) if (aux_specs and aux_class_weights) else None

    # Record how this run was trained (model size + key hyper-parameters).
    _write_run_config(
        run_dir / "args.yaml",
        {
            "size": size,
            "imgsz": imgsz,
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
            "warmup_epochs": warmup_epochs,
            "min_lr_ratio": min_lr_ratio,
            "clip_norm": clip_norm,
            "ema": ema,
            "ema_decay": ema_decay,
            "ema_tau": ema_tau,
            "amp": amp,
            "tf32": tf32,
            "compile": compile,
            "seed": seed,
            "pretrained": pretrained,
            "mask_anneal": mask_anneal,
            "mask_anneal_start": mask_anneal_start,
            "mask_anneal_end": mask_anneal_end,
            "flip_prob": flip_prob,
            "min_scale": min_scale,
            "max_scale": max_scale,
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
        },
    )

    # --- model ---
    model = build_model(
        size,
        nc=nc,
        imgsz=imgsz,
        names=names,
        aux_heads=aux_specs,
        aux_head_arch=aux_head_arch,
        loss_weights=loss_weights,
        num_upscale_blocks=num_upscale_blocks,
    ).to(dev)
    print(f"[model] loss weights: {loss_weights}; upscale blocks: {model.num_upscale_blocks}")
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
        missing, unexpected = model.load_state_dict(init_ckpt["model"], strict=False)
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

    # ``torch.compile`` wraps the module; ``core`` is the un-wrapped model used for
    # buffer access (``attn_mask_probs``) and state-dict export so checkpoints keep
    # clean (un-prefixed) keys. Compile is experimental here — the HF matcher/loss
    # use dynamic shapes that can trigger graph breaks/recompiles.
    if compile:
        model = torch.compile(model)
    core = _unwrap(model)

    scaler = torch.amp.GradScaler("cuda", enabled=amp and dev.type == "cuda")
    iters_per_epoch = max(1, len(train_loader))
    total_iters = epochs * iters_per_epoch
    warmup_iters = int(warmup_epochs * iters_per_epoch)
    warmup_start_factor = warmup_lr_start / lr0 if lr0 > 0 else 0.0

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
            task="instance",
            aux_heads=aux_specs,
            aux_head_arch=aux_head_arch,
            letterbox=letterbox,
            loss_weights=loss_weights,
            num_upscale_blocks=core.num_upscale_blocks,
            **extra,
        )
        save_checkpoint(ckpt, path)

    # Per-epoch metrics CSV (segmentation always; one column per aux head if present).
    csv_fields = ["epoch", "train/loss"]
    csv_fields += [f"train/aux_acc/{s.name}" for s in aux_specs]
    if val_ds is not None:
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
        for step, (pixel_values, mask_labels, class_labels, aux_labels) in enumerate(pbar):
            it = epoch * iters_per_epoch + step
            factor = _lr_factor(it, total_iters, warmup_iters, warmup_start_factor, min_lr_ratio)
            for g in optimizer.param_groups:
                g["lr"] = g["initial_lr"] * factor

            if mask_anneal:
                p = _attn_mask_prob(it, total_iters, mask_anneal_start, mask_anneal_end)
                core.eomt.attn_mask_probs.fill_(p)

            pixel_values = pixel_values.to(dev)
            mask_labels = [m.to(dev) for m in mask_labels]
            class_labels = [c.to(dev) for c in class_labels]
            aux_labels = {k: [t.to(dev) for t in v] for k, v in aux_labels.items()}

            # Gradient accumulation: zero grads at the start of each window, divide
            # the loss by ``accum_steps`` (so summed micro-batch grads average), and
            # only step/clip/update on the window boundary (or the epoch's last batch).
            if step % accum_steps == 0:
                optimizer.zero_grad(set_to_none=True)
            aux_gated = None
            with torch.amp.autocast("cuda", enabled=amp and dev.type == "cuda"):
                out = model(pixel_values, mask_labels=mask_labels, class_labels=class_labels)
                loss = out["loss"]
                if aux_specs:
                    # Match once per step, then gate to well-localized (IoU) and
                    # correctly-classified queries so the attribute trains only on
                    # instances the detector actually got right. Reuse for accuracy.
                    aux_indices = match_queries(model, out, mask_labels, class_labels)
                    aux_gated = gate_indices(
                        out, aux_indices, mask_labels, class_labels,
                        iou_thr=aux_iou_gate, require_class=aux_class_gate,
                    )
                    a_loss, _ = aux_loss(
                        model, out, mask_labels, class_labels, aux_labels,
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
                    model, out, mask_labels, class_labels, aux_labels, indices=aux_gated
                ).items():
                    aux_hits[name] += hit
                    aux_tot[name] += tot
                if track_pc_train:  # per-primary diagnostic when there is no val set
                    iou_only = gate_indices(
                        out, aux_indices, mask_labels, class_labels,
                        iou_thr=aux_iou_gate, require_class=False,
                    )
                    for name, buckets in aux_accuracy_by_primary(
                        model, out, mask_labels, class_labels, aux_labels, indices=iou_only
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
                "lr": f"{optimizer.param_groups[-1]['lr']:.2e}",
            }
            if aux_specs:  # running matched-query accuracy per head
                postfix["aux_acc"] = " ".join(
                    f"{n}={_safe_acc(aux_hits[n], aux_tot[n]):.2f}" for n in aux_hits
                )
            pbar.set_postfix(postfix)
            if tb is not None and step % 20 == 0:
                scalars = {"train/loss": loss_item, "lr/head": optimizer.param_groups[-1]["lr"]}
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
            metrics = evaluate(
                eval_model,
                val_ds,
                device=dev,
                batch_size=batch,
                num_workers=workers,
                conf_thres=conf_thres,
                max_det=max_det,
                amp=amp,
                verbose=True,
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
        _save(weights_dir / "last.pt", state_dict=core.state_dict(), with_trainer_state=True)
        cur = metrics.get("segm/mAP", None)
        if cur is not None and cur > best_metric:
            best_metric = cur
            _save(weights_dir / "best.pt", state_dict=eval_model.state_dict(), with_trainer_state=False)
            print(f"[epoch {epoch}] new best segm mAP {best_metric:.4f} -> best.pt")

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

    if tb is not None:
        tb.close()

    return {
        "best_metric": best_metric,
        "last": str(weights_dir / "last.pt"),
        "best": str(weights_dir / "best.pt") if (weights_dir / "best.pt").exists() else None,
        "weights_dir": str(weights_dir),
    }

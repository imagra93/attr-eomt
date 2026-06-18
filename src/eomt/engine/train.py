"""Training loop for EoMT instance segmentation.

Initializes the encoder from DINOv2, fine-tunes with AdamW (a low LR multiplier
on the pretrained encoder, full LR on the mask/class head), a cosine schedule
with warmup, AMP and gradient clipping. Validates every ``val_interval`` epochs
with COCO segm mAP and keeps both ``last.pt`` and the ``best.pt`` (highest
``segm/mAP``). The EoMT segmentation loss is computed inside the HF model.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data import CocoInstanceSeg, CocoValImages, collate_train
from ..model import build_model, load_dinov2_backbone
from ..serialization import load_raw, save_checkpoint, wrap_checkpoint
from .validate import evaluate

_ENCODER_PREFIXES = ("eomt.embeddings", "eomt.layers", "eomt.layernorm")


def _resolve_device(device: str) -> torch.device:
    if device in ("", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def build_optimizer(model, lr: float, weight_decay: float, backbone_lr_mult: float):
    """AdamW with a low LR multiplier on the pretrained DINOv2 encoder."""
    backbone, head = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone if name.startswith(_ENCODER_PREFIXES) else head).append(p)
    groups = [
        {"params": backbone, "lr": lr * backbone_lr_mult},
        {"params": head, "lr": lr},
    ]
    groups = [g for g in groups if g["params"]]
    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)
    for g in opt.param_groups:
        g["initial_lr"] = g["lr"]
    return opt


def _lr_factor(it, total_iters, warmup_iters, warmup_start_factor, min_ratio):
    if it < warmup_iters:
        return warmup_start_factor + (1.0 - warmup_start_factor) * it / max(1, warmup_iters)
    progress = (it - warmup_iters) / max(1, total_iters - warmup_iters)
    cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return min_ratio + (1.0 - min_ratio) * cos


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
    lr0: float = 1e-4,
    weight_decay: float = 0.05,
    backbone_lr_mult: float = 0.1,
    warmup_epochs: float = 1.0,
    warmup_lr_start: float = 1e-6,
    min_lr_ratio: float = 0.01,
    clip_norm: float = 0.01,
    workers: int = 4,
    device: str = "auto",
    amp: bool = True,
    pretrained: bool = True,
    flip_prob: float = 0.5,
    min_scale: float = 0.5,
    max_scale: float = 1.0,
    project: str = "runs/train",
    name: str = "eomt",
    val_interval: int = 1,
    conf_thres: float = 0.0,
    max_det: int = 100,
    logger: str = "none",
    resume: str | None = None,
) -> dict:
    """Fine-tune an EoMT instance-segmentation model on a COCO-format dataset."""
    if imgsz % 14:
        raise ValueError(f"imgsz={imgsz} must be divisible by 14 (DINOv2 grid).")

    dev = _resolve_device(device)
    run_dir = Path(project) / name
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

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
    print(f"[data] train: {len(train_ds)} images, {nc} classes")
    train_loader = DataLoader(
        train_ds,
        batch_size=batch,
        shuffle=True,
        num_workers=workers,
        collate_fn=collate_train,
        pin_memory=True,
        drop_last=True,
    )

    val_ds = None
    if val_images and val_json:
        val_ds = CocoValImages(val_images, val_json, imgsz=imgsz)
        print(f"[data] val:   {len(val_ds)} images")

    # --- model ---
    model = build_model(size, nc=nc, imgsz=imgsz, names=names).to(dev)
    start_epoch, best_metric = 0, -1.0
    optimizer = build_optimizer(model, lr0, weight_decay, backbone_lr_mult)

    if resume:
        ckpt = load_raw(resume)
        model.load_state_dict(ckpt["model"], strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_metric = float(ckpt.get("best_metric", -1.0))
        print(f"[resume] from {resume} at epoch {start_epoch}")
    elif pretrained:
        load_dinov2_backbone(model)
        model.to(dev)

    scaler = torch.amp.GradScaler("cuda", enabled=amp and dev.type == "cuda")
    iters_per_epoch = max(1, len(train_loader))
    total_iters = epochs * iters_per_epoch
    warmup_iters = int(warmup_epochs * iters_per_epoch)
    warmup_start_factor = warmup_lr_start / lr0 if lr0 > 0 else 0.0

    tb = _Logger(logger, run_dir, name) if logger != "none" else None

    def _save(path: Path, *, with_trainer_state: bool):
        extra = {}
        if with_trainer_state:
            extra = {
                "epoch": epoch,
                "best_metric": best_metric,
                "optimizer": optimizer.state_dict(),
            }
        ckpt = wrap_checkpoint(
            model.state_dict(),
            size=size,
            nc=nc,
            imgsz=imgsz,
            names=names,
            task="instance",
            **extra,
        )
        save_checkpoint(ckpt, path)

    # --- epochs ---
    for epoch in range(start_epoch, epochs):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs - 1}", unit="batch")
        for step, (pixel_values, mask_labels, class_labels) in enumerate(pbar):
            it = epoch * iters_per_epoch + step
            factor = _lr_factor(it, total_iters, warmup_iters, warmup_start_factor, min_lr_ratio)
            for g in optimizer.param_groups:
                g["lr"] = g["initial_lr"] * factor

            pixel_values = pixel_values.to(dev)
            mask_labels = [m.to(dev) for m in mask_labels]
            class_labels = [c.to(dev) for c in class_labels]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp and dev.type == "cuda"):
                loss = model(pixel_values, mask_labels=mask_labels, class_labels=class_labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()

            running += float(loss.detach())
            avg = running / (step + 1)
            pbar.set_postfix(loss=f"{float(loss):.3f}", avg=f"{avg:.3f}", lr=f"{optimizer.param_groups[-1]['lr']:.2e}")
            if tb is not None and step % 20 == 0:
                tb.log({"train/loss": float(loss), "lr/head": optimizer.param_groups[-1]["lr"]}, it)

        epoch_loss = running / iters_per_epoch
        print(f"[epoch {epoch}] mean loss {epoch_loss:.4f}")

        # --- validation ---
        metrics = {}
        if val_ds is not None and (epoch + 1) % val_interval == 0:
            metrics = evaluate(
                model,
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
            if tb is not None:
                tb.log({f"val/{k}": v for k, v in metrics.items()}, epoch)

        # --- checkpoints ---
        _save(weights_dir / "last.pt", with_trainer_state=True)
        cur = metrics.get("segm/mAP", None)
        if cur is not None and cur > best_metric:
            best_metric = cur
            _save(weights_dir / "best.pt", with_trainer_state=False)
            print(f"[epoch {epoch}] new best segm mAP {best_metric:.4f} -> best.pt")

    if tb is not None:
        tb.close()

    return {
        "best_metric": best_metric,
        "last": str(weights_dir / "last.pt"),
        "best": str(weights_dir / "best.pt") if (weights_dir / "best.pt").exists() else None,
        "weights_dir": str(weights_dir),
    }

#!/usr/bin/env python
"""Train an EoMT model on a dataset.

Defaults to COCO 2017 instance segmentation, which is auto-downloaded on first run.

    python scripts/train.py                          # COCO, large model
    python scripts/train.py --size s --batch 16      # 12 epochs, the official COCO recipe
    python scripts/train.py --data sample_data/data.yaml --epochs 1

Augmentation defaults are a strong general recipe (see ``eomt.data.transforms.AugConfig``). Override any knob:

    python scripts/train.py --data my/data.yaml --aug min_scale=0.6 --aug max_scale=1.6 --aug instance_crop_prob=0.8
    python scripts/train.py --data my/data.yaml --aug-preset legacy      # the original flip + LSJ + crop + colour jitter

To train on a dataset with secondary per-instance attributes (the auxiliary-class
feature), just point --data at it; the heads are discovered from the COCO JSON.

Fine-tune or resume (use one, not both):

    python scripts/train.py --data my/data.yaml --resume runs/train/eomt-l                  # continue a run: epoch, optimizer, EMA
    python scripts/train.py --data my/data.yaml --weights runs/train/eomt-l/weights/best.pt \
        --name eomt-l-ft --epochs 3 --lr 2e-5 --warmup 100,200 --ema-decay 0.999 --mask-free  # a short, gentle warm start

A flag left out keeps ``EoMT.train()``'s default (see the README, "Optimisation recipe").
"""

from __future__ import annotations

import argparse

import yaml

from eomt import EoMT
from eomt.engine.train import EOMT_MASK_ANNEAL_START


def _parse_aug(items: list[str], preset: str | None) -> dict | None:
    """``["rotate_prob=0.5", "blur_sigma=[0.3,1]"]`` -> ``{"rotate_prob": 0.5, "blur_sigma": [0.3, 1]}`` (YAML values)."""
    aug: dict = {"preset": preset} if preset else {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise SystemExit(f"--aug expects KEY=VALUE, got {item!r}")
        aug[key.strip()] = yaml.safe_load(value)
    return aug or None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="coco", help="Dataset YAML path or alias ('coco' auto-downloads).")
    p.add_argument("--size", default="l", choices=["s", "b", "l"], help="Model size.")
    p.add_argument(
        "--task", default="instance", choices=["instance", "detect"],
        help="Head family: 'instance' (mask segmentation) or 'detect' (boxes only).",
    )
    p.add_argument("--epochs", type=int, default=12, help="Total epochs (official COCO recipe: 12); the LR schedule and mask anneal span them.")
    p.add_argument("--batch", type=int, default=4, help="Micro-batch size (per optimizer micro-step).")
    p.add_argument("--imgsz", type=int, default=644, help="Square input size (divisible by 14).")
    p.add_argument("--device", default="auto", help="auto (GPU 0), cpu, cuda:N / N, or GPU indices such as 0,1 to train on several.")
    p.add_argument("--name", default=None, help="Run name (default: eomt-{size}).")
    p.add_argument("--weights", default=None, help="Init from a checkpoint/run to fine-tune (warm start).")
    p.add_argument("--resume", default=None, help="Resume a run from a checkpoint/run folder.")
    p.add_argument(
        "--fpn-scales", default="2,1,0.5",
        help="B1 multi-scale SimpleFPN scales relative to the native grid (default "
             "'2,1,0.5', on by default). Pass 'none'/'off' for the single-scale model.",
    )
    p.add_argument("--box-head", action="store_true", help="Auxiliary box head: L1 + GIoU loss and a box term in the matcher.")
    p.add_argument("--no-deep-supervision", action="store_true", help="Stop supervising a query block once its masked attention is annealed away (it stays supervised while it masks).")
    p.add_argument("--upscale-blocks", type=int, default=None, help="Mask-head upscale blocks (2 = 184x184 logits at 644 px, 3 = 368x368).")
    p.add_argument("--iou-aware-cls", action="store_true", help="Class target of a matched query = (1 + mask IoU) / 2 (scores rank masks by quality).")
    p.add_argument("--quality-weight", type=float, default=0.0, help="Mask-quality head loss weight (predicted IoU scores the masks); 0 = off.")
    p.add_argument("--val-imgsz", type=int, default=None, help="Validation input size (default: --imgsz); e.g. imgsz x the mean training scale.")
    p.add_argument(
        "--aug", action="append", default=[], metavar="KEY=VALUE",
        help="Augmentation override (repeatable), e.g. --aug rotate_prob=0.5 --aug 'blur_sigma=[0.3,1.0]'. "
             "Any eomt.data.transforms.AugConfig field; beats the data.yaml `train_aug` block.",
    )
    p.add_argument(
        "--aug-preset", choices=["default", "legacy"], default=None,
        help="Start from the strong default recipe or from the original flip + LSJ + crop + colour-jitter recipe.",
    )
    g = p.add_argument_group("optimisation (default: EoMT.train()'s, the reference EoMT recipe)")
    g.add_argument("--lr", type=float, default=None, help="Peak head LR (lr0, default 1e-4); the encoder follows with layer-wise decay.")
    g.add_argument("--warmup", default=None, metavar="HEAD,VIT", help="Two-stage warmup in optimizer steps, e.g. 500,1000 (the default).")
    g.add_argument("--nominal-batch", type=int, default=None, help="Global effective batch (16); --batch is per GPU, the rest is accumulated.")
    g.add_argument("--ema-decay", type=float, default=None, help="EMA decay (0.9999). Lower it for a short run, which would barely leave its start.")
    g.add_argument("--ema-tau", type=float, default=None, help="EMA warmup time constant in steps (2000).")
    g.add_argument("--mask-free", action="store_true",
                   help="No masked attention from the first step (annealing windows 0 -> 0): for fine-tuning a model that is deployed mask-free.")
    g.add_argument("--aux-dropout", type=float, default=None, help="Dropout in the attribute heads (0).")
    g.add_argument("--val-interval", type=int, default=None, help="Validate every N epochs and after the last (2).")
    g.add_argument("--workers", type=int, default=None, help="Data-loader workers per GPU (8).")
    g.add_argument("--seed", type=int, default=None, help="Seed (ignored when resuming, so the epoch's shuffle is not replayed).")
    args = p.parse_args()

    if args.weights and args.resume:
        raise SystemExit("pass --resume (continue a run) or --weights (warm-start a fresh run), not both.")
    overrides = {
        "lr0": args.lr, "nominal_batch": args.nominal_batch, "ema_decay": args.ema_decay, "ema_tau": args.ema_tau,
        "aux_head_dropout": args.aux_dropout, "val_interval": args.val_interval, "workers": args.workers,
        "seed": None if args.resume else args.seed,
    }
    if args.warmup:
        overrides["warmup_steps"] = tuple(int(x) for x in args.warmup.split(","))
    if args.mask_free:
        zeros = (0.0,) * len(EOMT_MASK_ANNEAL_START)
        overrides.update(mask_anneal_start=zeros, mask_anneal_end=zeros)
    overrides = {k: v for k, v in overrides.items() if v is not None}

    if args.fpn_scales.strip().lower() in ("", "none", "off", "0"):
        fpn_scales = None
    else:
        fpn_scales = [float(s) for s in args.fpn_scales.split(",") if s.strip()]

    # Init from a checkpoint (fine-tune / resume) or from a size (fresh, DINOv2 backbone).
    model = EoMT(args.weights or args.resume or args.size, device=args.device)
    result = model.train(
        data=args.data,
        family=args.task,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        name=args.name,
        resume=bool(args.resume),
        fpn_scales=fpn_scales,
        box_head=args.box_head,
        deep_supervision=not args.no_deep_supervision,
        num_upscale_blocks=args.upscale_blocks,
        iou_aware_cls=args.iou_aware_cls,
        quality_weight=args.quality_weight,
        val_imgsz=args.val_imgsz,
        aug=_parse_aug(args.aug, args.aug_preset),
        **overrides,
    )
    if result["best_metric"] >= 0:
        metric = "bbox mAP" if args.task == "detect" else "segm mAP"
        print(f"[done] best {metric}={result['best_metric']:.4f}; weights in {result['weights_dir']}")
    else:
        print(f"[done] weights in {result['weights_dir']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Train an EoMT model on a dataset.

Defaults to COCO 2017 instance segmentation, which is auto-downloaded on first run.

    python scripts/train.py                          # COCO, large model
    python scripts/train.py --size s --epochs 50 --batch 16
    python scripts/train.py --data sample_data/data.yaml --epochs 1

To train on a dataset with secondary per-instance attributes (the auxiliary-class
feature), just point --data at it; the heads are discovered from the COCO JSON.
"""

from __future__ import annotations

import argparse

from eomt import EoMT


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="coco", help="Dataset YAML path or alias ('coco' auto-downloads).")
    p.add_argument("--size", default="l", choices=["s", "b", "l"], help="Model size.")
    p.add_argument(
        "--task", default="instance", choices=["instance", "detect"],
        help="Head family: 'instance' (mask segmentation) or 'detect' (boxes only).",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=4, help="Micro-batch size (per optimizer micro-step).")
    p.add_argument("--imgsz", type=int, default=644, help="Square input size (divisible by 14).")
    p.add_argument("--device", default="auto")
    p.add_argument("--name", default=None, help="Run name (default: eomt-{size}).")
    p.add_argument("--weights", default=None, help="Init from a checkpoint/run to fine-tune (warm start).")
    p.add_argument("--resume", default=None, help="Resume a run from a checkpoint/run folder.")
    p.add_argument(
        "--fpn-scales", default="2,1,0.5",
        help="B1 multi-scale SimpleFPN scales relative to the native grid (default "
             "'2,1,0.5', on by default). Pass 'none'/'off' for the single-scale model.",
    )
    p.add_argument("--optim-8bit", action=argparse.BooleanOptionalAction, default=True,
                   help="torchao 8-bit AdamW (~1.8 GB less VRAM for ViT-L; within-noise of fp32 "
                        "Adam). ON by default; --no-optim-8bit for plain fp32 AdamW. Resuming a "
                        "run must reuse the same setting (optimizer state differs).")
    p.add_argument("--ema-device", default="cpu", choices=["cuda", "cpu"],
                   help="Where to hold the EMA shadow copy (default 'cpu': frees ~1.2 GB VRAM, "
                        "moved to GPU only for validation; EMA math is identical either way).")
    args = p.parse_args()

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
        optim_8bit=args.optim_8bit,
        ema_device=args.ema_device,
    )
    if result["best_metric"] >= 0:
        metric = "bbox mAP" if args.task == "detect" else "segm mAP"
        print(f"[done] best {metric}={result['best_metric']:.4f}; weights in {result['weights_dir']}")
    else:
        print(f"[done] weights in {result['weights_dir']}")


if __name__ == "__main__":
    main()

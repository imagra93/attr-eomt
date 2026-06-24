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
    args = p.parse_args()

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
    )
    if result["best_metric"] >= 0:
        metric = "bbox mAP" if args.task == "detect" else "segm mAP"
        print(f"[done] best {metric}={result['best_metric']:.4f}; weights in {result['weights_dir']}")
    else:
        print(f"[done] weights in {result['weights_dir']}")


if __name__ == "__main__":
    main()

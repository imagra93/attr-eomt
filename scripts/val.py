#!/usr/bin/env python
"""Validate a trained EoMT model on a dataset (COCO segm + bbox mAP).

    python scripts/val.py runs/train/eomt-l                 # validate on COCO
    python scripts/val.py runs/train/eomt-l/weights/best.pt --data sample_data/data.yaml
    python scripts/val.py runs/train/eomt-l --augment                  # + the image at 1.5x imgsz (test-time augmentation)
    python scripts/val.py runs/train/eomt-l --tiles                    # native-resolution tiles + the whole image
    python scripts/val.py runs/train/eomt-l --tiles 1024 --augment     # 1024 px tiles + the image at 1x and 1.5x
"""

from __future__ import annotations

import argparse

from eomt import EoMT


def add_tta_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("test-time augmentation / tiled inference (instance family; see eomt/tta.py)")
    g.add_argument("--augment", action="store_true", help="Also run the image at 1.5x imgsz and merge.")
    g.add_argument("--tiles", type=int, nargs="?", const=True, default=False, metavar="SIZE",
                   help="Overlapping tiles (SIZE px, default the model's imgsz) plus the whole image, merged.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("weights", help="Checkpoint .pt or a run/weights folder.")
    p.add_argument("--data", default="coco", help="Dataset YAML path or alias ('coco').")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default="auto")
    add_tta_args(p)
    args = p.parse_args()

    model = EoMT(args.weights, device=args.device)
    metrics = model.val(data=args.data, batch=args.batch, augment=args.augment, tiles=args.tiles)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()

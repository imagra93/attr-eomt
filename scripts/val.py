#!/usr/bin/env python
"""Validate a trained EoMT model on a dataset (COCO segm + bbox mAP).

    python scripts/val.py runs/train/eomt-l                 # validate on COCO
    python scripts/val.py runs/train/eomt-l/weights/best.pt --data sample_data/data.yaml
"""

from __future__ import annotations

import argparse

from eomt import EoMT


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("weights", help="Checkpoint .pt or a run/weights folder.")
    p.add_argument("--data", default="coco", help="Dataset YAML path or alias ('coco').")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    model = EoMT(args.weights, device=args.device)
    metrics = model.val(data=args.data, batch=args.batch)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()

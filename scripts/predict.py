#!/usr/bin/env python
"""Predict and plot segmentation (+ any secondary attributes) on some images.

    python scripts/predict.py runs/train/eomt-l                       # uses scripts/sample_images
    python scripts/predict.py runs/train/eomt-l/weights/best.pt path/to/images --conf 0.3

Annotated images are written to runs/predict/; each detection is labelled with its
class and, for models with auxiliary heads, each attribute and its confidence.
"""

from __future__ import annotations

import argparse

from eomt import EoMT


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("weights", help="Checkpoint .pt or a run/weights folder.")
    p.add_argument("source", nargs="?", default="scripts/sample_images", help="Image file or directory.")
    p.add_argument("--out", default="runs/predict", help="Output directory for annotated images.")
    p.add_argument("--conf", type=float, default=0.3, help="Confidence threshold.")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    model = EoMT(args.weights, device=args.device)
    results = model.predict(args.source, plot=True, save=args.out, conf_thres=args.conf)
    print(f"[done] wrote {len(results)} annotated image(s) to {args.out}")


if __name__ == "__main__":
    main()

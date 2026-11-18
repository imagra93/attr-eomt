#!/usr/bin/env python
"""Predict images with a trained EoMT model and render them.

    python scripts/predict.py runs/train/eomt-l images/                  # every image of the folder, batched
    python scripts/predict.py runs/train/eomt-l photo.jpg --tiles        # native-resolution tiles + the whole image
    python scripts/predict.py runs/train/multiview photos/ --subject     # the photos of each subject together

The confidence threshold and the other inference settings default to the checkpoint's (its run config's ``inference``).
"""

from __future__ import annotations

import argparse

from eomt import EoMT


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("weights", help="Checkpoint .pt or a run/weights folder.")
    p.add_argument("source", help="Image file or folder (with --subject: one subject's photos, or a subfolder each).")
    p.add_argument("--out", default="runs/predict", help="Output folder of the rendered images.")
    p.add_argument("--conf", type=float, default=None, help="Confidence threshold (default: the checkpoint's).")
    p.add_argument("--imgsz", type=int, default=None, help="Input size (default: the trained size).")
    p.add_argument("--batch", type=int, default=8, help="Photos per forward pass.")
    p.add_argument("--device", default="auto")
    p.add_argument("--fp32", action="store_true", help="Run the network in fp32 (default: bf16 on a GPU).")
    p.add_argument("--augment", action="store_true", help="Also run the image at 1.5x imgsz and merge.")
    p.add_argument("--tiles", type=int, nargs="?", const=True, default=False, metavar="SIZE",
                   help="Overlapping tiles (SIZE px, default imgsz) plus the whole image, merged.")
    p.add_argument("--subject", action="store_true", help="Predict the photos of each subject together (cross-view).")
    args = p.parse_args()
    model = EoMT(args.weights, device=args.device)
    for _ in model.predict(args.source, plot=True, save=args.out, conf_thres=args.conf, imgsz=args.imgsz,
                           batch=args.batch, amp=not args.fp32, augment=args.augment, tiles=args.tiles,
                           subject=args.subject, stream=True):
        pass


if __name__ == "__main__":
    main()

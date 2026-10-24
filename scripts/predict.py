#!/usr/bin/env python
"""Predict and plot segmentation (+ any secondary attributes) on some images.

    python scripts/predict.py runs/train/eomt-l                       # uses scripts/sample_images
    python scripts/predict.py runs/train/eomt-l/weights/best.pt path/to/images --conf 0.3
    python scripts/predict.py runs/train/eomt-l big_images/ --tiles --augment   # tiles + the image at 1x and 1.5x
    python scripts/predict.py runs/train/multiview photos/ --subject            # the photos of a subject together

Annotated images are written to runs/predict/; each detection is labelled with its
class and, for models with auxiliary heads, each attribute and its confidence.

--subject is for a model from scripts/train_multiview.py: the photos in the folder are one
subject and each subfolder is another, so a folder of subfolders predicts many subjects.
"""

from __future__ import annotations

import argparse

from eomt import EoMT
from val import add_tta_args


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("weights", help="Checkpoint .pt or a run/weights folder.")
    p.add_argument("source", nargs="?", default="scripts/sample_images", help="Image file or directory.")
    p.add_argument("--out", default="runs/predict", help="Output directory for annotated images.")
    p.add_argument("--conf", type=float, default=None, help="Confidence threshold (default: the checkpoint's inference setting, else 0.3).")
    p.add_argument("--device", default="auto")
    add_tta_args(p)
    p.add_argument("--subject", nargs="?", const="order", choices=["order", "partition", "anchor"],
                   help="Predict the photos of each subject together (see above). A subject with more photos than the "
                        "model takes at once is split in order (default), into similar photos (partition), or each "
                        "photo with its most similar ones (anchor).")
    args = p.parse_args()

    model = EoMT(args.weights, device=args.device)
    results = model.predict(args.source, plot=True, save=args.out, conf_thres=args.conf,
                            augment=args.augment, tiles=args.tiles, subject=args.subject or False)
    print(f"[done] wrote {len(results)} annotated image(s) to {args.out}")


if __name__ == "__main__":
    main()

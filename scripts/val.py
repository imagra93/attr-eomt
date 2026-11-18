#!/usr/bin/env python
"""Validate a trained EoMT model: COCO segm + bbox mAP and, for a model with attribute heads, their metrics (accuracy per
categorical head, mean absolute error in real units per regression head).

    python scripts/val.py runs/train/eomt-l                            # on the validation data of its own run
    python scripts/val.py runs/train/eomt-l --data other/run.yaml      # on another dataset's validation split
    python scripts/val.py runs/train/eomt-l --augment                  # + the image at 1.5x imgsz (test-time augmentation)
    python scripts/val.py runs/train/eomt-l --tiles 1024 --flip        # 1024 px tiles + the whole image, and mirrored
    python scripts/val.py runs/train/multiview --multiview             # a multi-view post-training, as its trainer did
    python scripts/val.py other/best.pt --multiview --run runs/train/multiview --sets-file runs/train/multiview/val_sets.json

``--multiview`` scores the photos one by one and the sets of photos of each subject, with the keys of the run's
metrics.csv; ``--run`` validates as another run did and ``--sets-file`` fixes the sets, so that two models (or two
versions of a dataset) are scored on the same ones. The network runs in bf16 on a GPU, as in training and inference.
"""

from __future__ import annotations

import argparse

from eomt import EoMT


def add_tta_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("test-time augmentation / tiled inference (instance family; see eomt/tta.py)")
    g.add_argument("--augment", action="store_true", help="Also run the image at 1.5x imgsz and merge.")
    g.add_argument("--tiles", type=int, nargs="?", const=True, default=False, metavar="SIZE",
                   help="Overlapping tiles (SIZE px, default the model's imgsz) plus the whole image, merged.")
    g.add_argument("--flip", action="store_true", help="Also run every view mirrored (flip-aware heads are swapped back).")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("weights", help="Checkpoint .pt or a run/weights folder (best.pt preferred).")
    p.add_argument("--data", default=None, help="Run config YAML or 'coco' (default: the checkpoint's own run).")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--workers", type=int, default=4, help="Data-loader workers.")
    p.add_argument("--imgsz", type=int, default=None, help="Evaluation input size (default: the trained size).")
    p.add_argument("--device", default="auto")
    p.add_argument("--no-attributes", action="store_true", help="Skip the attribute heads' metrics (mAP only).")
    p.add_argument("--group-by", default=None, help="Image field(s), comma-separated: the main metrics per value too.")
    add_tta_args(p)
    g = p.add_argument_group("several photos of one subject (a model post-trained with multiview)")
    g.add_argument("--multiview", action="store_true", help="Validate as the trainer of the post-training did.")
    g.add_argument("--run", default=None, help="Validate as this run did (a run folder, its args.yaml or a checkpoint).")
    g.add_argument("--sets-file", default=None, help="Fixed validation sets (a run's val_sets.json).")
    args = p.parse_args()

    model = EoMT(args.weights, device=args.device)
    group_by = args.group_by.split(",") if args.group_by else None
    if args.multiview:
        if args.augment or args.tiles or args.flip:
            raise SystemExit("--multiview validates as the trainer does: no --augment / --tiles / --flip")
        metrics = model.val(data=args.data, multiview=True, run=args.run, sets_file=args.sets_file, batch=args.batch,
                            workers=args.workers, imgsz=args.imgsz, group_by=group_by)
    else:
        augment = {"flip": True} if args.flip else args.augment  # flip alone: the 1x / 1.5x whole image, mirrored too
        metrics = model.val(data=args.data, run=args.run, batch=args.batch, workers=args.workers, imgsz=args.imgsz,
                            augment=augment, tiles=args.tiles, attributes=not args.no_attributes, group_by=group_by)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Post-train a model so that the photos of one subject are predicted together, up to 4 at a time (cross-view).

The model's weights stay frozen: only the new cross-view modules and a matching head learn, on sets of views made from
the dataset's single annotated photos (no multi-photo annotations needed), so a photo predicted alone gives exactly the
model's old result. Point --weights at the trained model and --data at its dataset:

    python scripts/train_multiview.py --data my/data.yaml --weights runs/train/eomt-l/weights/best.pt
    python scripts/train_multiview.py --data my/data.yaml --weights best.pt --device 0,1    # two GPUs

The result is runs/train/multiview/weights/best.pt. Predict the photos of each subject together with

    python scripts/predict.py runs/train/multiview photos/ --subject    # photos/ = one subject, or a subfolder each

This is scripts/train.py with the recipe --multiview configs/xview.yaml --batch 1 --epochs 2 --val-interval 1
--name multiview: any of its flags can be added and wins (e.g. --epochs 4, --name multiview-2, --workers 4; see
python scripts/train.py --help). What trains and how the sets are made is in configs/xview.yaml.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import train

RECIPE = [
    "--multiview", str(Path(__file__).resolve().parent.parent / "configs" / "xview.yaml"),
    "--batch", "1",          # sets of photos per GPU (a set of 4 at 644 px takes ~15 GB), accumulated to 16
    "--epochs", "2",
    "--val-interval", "1",   # best.pt is the best epoch on the validation sets
    "--name", "multiview",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                allow_abbrev=False)
    p.add_argument("--data", required=True, help="Dataset YAML of single annotated photos.")
    p.add_argument("--weights", required=True, help="The trained model to post-train: checkpoint .pt or run folder.")
    args, rest = p.parse_known_args()
    # argparse keeps the last value of a repeated flag, so the user's own flags beat the recipe.
    train.main(["--data", args.data, "--weights", args.weights, *RECIPE, *rest])


if __name__ == "__main__":
    main()

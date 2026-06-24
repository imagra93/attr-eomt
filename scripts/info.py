#!/usr/bin/env python
"""Inspect an EoMT checkpoint: model size/task, classes, preprocessing, train state.

    python scripts/info.py runs/train/eomt-l                 # resolves best.pt
    python scripts/info.py runs/train/eomt-l/weights/last.pt
    python scripts/info.py hf://imagra93/eomt-l-coco

Prints everything needed to load and run the checkpoint — no model is built.
"""

from __future__ import annotations

import argparse

from eomt import format_summary, summarize_checkpoint


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("weights", help="Checkpoint .pt, a run/weights folder, or an hf:// ref.")
    args = p.parse_args()
    print(format_summary(summarize_checkpoint(args.weights)))


if __name__ == "__main__":
    main()

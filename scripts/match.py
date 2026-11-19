#!/usr/bin/env python
"""Link the instances of the photos of one subject: which detections are the same physical instance.

    python scripts/match.py runs/train/multiview photos/subject_42/      # one subject
    python scripts/match.py runs/train/multiview photos/ --each-subdir   # one subject per subfolder

The links use the model's matching head when it has one (a ``multiview`` post-training with ``matching``), else the
per-query embedding (re-identification), at its calibrated threshold and gate; ``--sim-thres`` and ``--group-by``
override them. Writes a grid of the photos colored by identity and a JSON summary per subject.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from eomt import EoMT


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("weights", help="Checkpoint .pt or a run/weights folder.")
    p.add_argument("source", help="Folder of photos of one subject (or a parent, with --each-subdir).")
    p.add_argument("--each-subdir", action="store_true", help="Every subfolder of source is one subject.")
    p.add_argument("--out", default="runs/match", help="Output folder.")
    p.add_argument("--conf", type=float, default=None, help="Confidence threshold (default: the checkpoint's).")
    p.add_argument("--sim-thres", type=float, default=None, help="Cosine floor of a link (default: the calibrated one).")
    p.add_argument("--group-by", default="auto",
                   help="Gate keys, comma-separated ('class' and attribute heads; 'none': no gate; default: calibrated).")
    p.add_argument("--links", default="chain", choices=("chain", "all", "none"), help="Connectors drawn in the grid.")
    p.add_argument("--no-plot", action="store_true", help="Only the JSON summary.")
    p.add_argument("--fill", action="store_true",
                   help="Top the last group up with the photos before it (5 photos at 4 a time: 4 and 4, not 4 + 1).")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    gate = args.group_by
    gate = "auto" if gate == "auto" else None if gate.lower() in ("", "none", "off") else tuple(gate.split(","))
    model = EoMT(args.weights, device=args.device)
    src = Path(args.source)
    subjects = sorted(d for d in src.iterdir() if d.is_dir()) if args.each_subdir else [src]
    for folder in subjects:
        model.match(folder, plot=not args.no_plot, save=args.out, conf_thres=args.conf, sim_thres=args.sim_thres,
                    group_by=gate, link_mode=args.links, fill=args.fill)


if __name__ == "__main__":
    main()

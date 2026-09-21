#!/usr/bin/env python
"""Re-identify instances across a set of photos of the same subject.

Each source folder is ONE subject photographed from several viewpoints; the script
decides which detections across those photos are the same physical instance and writes
an annotated grid plus a JSON summary.

    python scripts/match.py weights/best.pt photos/subject_42/
    python scripts/match.py weights/best.pt photos/ --each-subdir
    python scripts/match.py weights/best.pt photos/subject_42/ --group-by class,position
    python scripts/match.py weights/best.pt photos/subject_42/ --no-group-by --sim-thres 0.9
"""

from __future__ import annotations

import argparse
from pathlib import Path

from eomt import EoMT


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("weights", help="Checkpoint .pt or a run/weights folder.")
    p.add_argument("source", help="Folder of photos of one subject (or a parent, with --each-subdir).")
    p.add_argument("--out", default="runs/match", help="Output directory.")
    p.add_argument("--conf", type=float, default=0.3, help="Detection confidence threshold.")
    p.add_argument("--max-det", type=int, default=30, help="Max detections per photo.")
    p.add_argument("--sim-thres", type=float, default=0.6,
                   help="Cosine-similarity floor for 'same instance'.")
    p.add_argument("--group-by", default="class",
                   help="Comma-separated gate keys, e.g. class,position.")
    p.add_argument("--no-group-by", action="store_true",
                   help="Disable gating entirely (use a higher --sim-thres).")
    p.add_argument("--guard", default="mean", choices=("mean", "min", "off"),
                   help="Merge guard against transitive chaining.")
    p.add_argument("--guard-factor", type=float, default=1.0)
    p.add_argument("--center", action="store_true",
                   help="Mean-center embeddings before normalizing (transductive).")
    p.add_argument("--arrows", default="chain", choices=("chain", "all", "none"))
    p.add_argument("--legend-attr", default=None,
                   help="Attribute head shown beside the class in the legend "
                        "(default: each identity's first attribute).")
    p.add_argument("--panel-size", type=int, default=480)
    p.add_argument("--each-subdir", action="store_true",
                   help="Treat every immediate subfolder of SOURCE as its own subject.")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    group_by = None if args.no_group_by else tuple(
        k.strip() for k in args.group_by.split(",") if k.strip()
    )
    root = Path(args.source)
    if not root.exists():
        raise SystemExit(f"[match] no such source: {root}")
    if args.each_subdir and not root.is_dir():
        raise SystemExit(f"[match] --each-subdir needs a parent folder, got a file: {root}")
    sources = sorted(d for d in root.iterdir() if d.is_dir()) if args.each_subdir else [root]
    if not sources:
        raise SystemExit(f"[match] no subdirectories under {root}")

    model = EoMT(args.weights, device=args.device)
    total = 0
    for src in sources:
        result = model.infer_match(
            src, plot=not args.no_plot, save=args.out, conf_thres=args.conf,
            max_det=args.max_det, group_by=group_by, sim_thres=args.sim_thres,
            guard=args.guard, guard_factor=args.guard_factor, center=args.center,
            arrow_mode=args.arrows, panel_size=args.panel_size,
            legend_attr=args.legend_attr,
        )
        total += result["num_identities"]
    print(f"[done] {total} identities across {len(sources)} subject(s) -> {args.out}")


if __name__ == "__main__":
    main()

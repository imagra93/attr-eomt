#!/usr/bin/env python
"""Train an EoMT model. The run is written out in its run config YAML (``--data``, see sample_data/run.yaml): the data,
the attribute heads and every settings section; a flag beats the file.

    python scripts/train.py --data sample_data/run.yaml --size s                 # the run as its YAML says
    python scripts/train.py --data sample_data/run.yaml --epochs 1 --device 0,1  # a flag beats the file
    python scripts/train.py --data coco --size l --batch 4                       # COCO 2017 (downloaded if missing)

Any setting of a section (model, optim, aux, aug, val, multiview, inference) or train() keyword, as ``--set``
(repeatable, the value read as YAML):

    python scripts/train.py --data run.yaml --set optim.lr0=5e-5 --set aug.rotate_prob=0 --set val.interval=1

Warm start (weights only, a new run) or resume (continue a run from its last.pt, exactly):

    python scripts/train.py --data run.yaml --weights runs/train/eomt-l/weights/best.pt --name eomt-l-ft \\
        --set optim.lr0=2e-5 --set 'optim.warmup_steps=[100,200]' --set optim.ema_decay=0.999 --set optim.mask_anneal=false
    python scripts/train.py --data run.yaml --resume runs/train/eomt-l

A multi-view post-training (sets of photos of one subject) is a warm start with a ``multiview`` section in the YAML.
"""

from __future__ import annotations

import argparse

import yaml

from eomt import EoMT

_SECTIONS = ("model", "optim", "aux", "aug", "val", "multiview", "inference", "attributes")


def parse_sets(items: list[str]) -> dict:
    """``["optim.lr0=5e-5", "epochs=3"]`` -> ``{"optim": {"lr0": 5e-05}, "epochs": 3}`` (values read as YAML)."""
    out: dict = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise SystemExit(f"--set expects KEY=VALUE or SECTION.KEY=VALUE, got {item!r}")
        section, dot, field = key.strip().partition(".")
        if dot:
            if section not in _SECTIONS:
                raise SystemExit(f"--set {key}: unknown section {section!r} (one of {', '.join(_SECTIONS)})")
            out.setdefault(section, {})[field] = yaml.safe_load(value)
        else:
            out[section] = yaml.safe_load(value)
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="coco", help="Run config YAML, or 'coco' (downloaded if missing).")
    p.add_argument("--size", default="l", choices=["s", "b", "l"], help="Model size of a new model.")
    p.add_argument("--task", default=None, choices=["instance", "detect"], help="Masks (default) or boxes only.")
    p.add_argument("--weights", default=None, help="Warm-start from a checkpoint or run (weights only, a new run).")
    p.add_argument("--resume", default=None, help="Continue a run (its folder or last.pt).")
    p.add_argument("--device", default="auto", help="auto (GPU 0), cpu, N, or GPU indices such as 0,1 (one process each).")
    for flag, kind, what in (("--epochs", int, "Epochs the schedule spans (12)."), ("--batch", int, "Micro-batch per GPU (4)."),
                             ("--imgsz", int, "Input size, a multiple of 14 (644)."), ("--workers", int, "Data-loader workers per GPU (8)."),
                             ("--seed", int, "Seed (ignored when resuming)."), ("--name", str, "Run name (eomt-<size>).")):
        p.add_argument(flag, type=kind, default=None, help=what)
    p.add_argument("--set", action="append", default=[], metavar="[SECTION.]KEY=VALUE",
                   help="Any setting, e.g. --set optim.lr0=5e-5 --set multiview.trainable=query_blocks (repeatable).")
    args = p.parse_args(argv)
    if args.weights and args.resume:
        raise SystemExit("pass --resume (continue a run) or --weights (warm-start a new run), not both")
    kw = parse_sets(args.set)
    kw.update({k: v for k, v in {"family": args.task, "epochs": args.epochs, "batch": args.batch, "imgsz": args.imgsz,
                                 "workers": args.workers, "seed": None if args.resume else args.seed,
                                 "name": args.name}.items() if v is not None})
    model = EoMT(args.weights or args.resume or args.size, device=args.device)
    result = model.train(data=args.data, resume=bool(args.resume), **kw)
    print(f"[done] best {result['best_key']}={result['best_metric']:.4f}; weights in {result['weights_dir']}"
          if result.get("best") else f"[done] weights in {result['weights_dir']}")


if __name__ == "__main__":
    main()

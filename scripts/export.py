#!/usr/bin/env python
"""Write a copy of a checkpoint for deployment with its Linear and Conv weights in bf16 (or fp16): about half the file.
Norms, embeddings and the other parameters stay fp32, and everything the checkpoint records is kept (classes, heads,
inference settings, link calibration, the run that made it, its epoch and metrics).

    python scripts/export.py runs/train/multiview/weights/best.pt deploy/best_bf16.pt
    python scripts/export.py runs/train/multiview deploy/best_fp16.pt --dtype fp16

EoMT(path) loads it back into an fp32 model, with the rounded weights.
"""

from __future__ import annotations

import argparse

from eomt import EoMT


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("weights", help="Checkpoint .pt or a run/weights folder (best.pt preferred).")
    p.add_argument("out", help="The checkpoint to write.")
    p.add_argument("--dtype", default="bf16", choices=("bf16", "fp16"), help="Precision of the Linear / Conv weights.")
    args = p.parse_args()
    EoMT(args.weights, device="cpu").save(args.out, dtype=args.dtype)
    print(f"[export] {args.out}: {args.dtype} Linear / Conv weights")


if __name__ == "__main__":
    main()

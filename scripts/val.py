#!/usr/bin/env python
"""Validate a trained EoMT model on a dataset: COCO segm + bbox mAP and, for a model with secondary heads, their metrics
(accuracy per categorical head, mean absolute error in real units per regression head).

    python scripts/val.py runs/train/eomt-l                 # validate on COCO
    python scripts/val.py runs/train/eomt-l/weights/best.pt --data sample_data/data.yaml
    python scripts/val.py runs/train/eomt-l --augment                  # + the image at 1.5x imgsz (test-time augmentation)
    python scripts/val.py runs/train/eomt-l --tiles                    # native-resolution tiles + the whole image
    python scripts/val.py runs/train/eomt-l --tiles 1024 --augment     # 1024 px tiles + the image at 1x and 1.5x
    python scripts/val.py runs/train/eomt-l --flip                     # + every view mirrored (flip-aware heads are swapped back)
    python scripts/val.py runs/train/eomt-l/weights/last.pt --ema      # the EMA weights of a last.pt, as the trainer validates

``best.pt`` is already the EMA, exported mask-free. ``--ema`` reads the EMA copy kept in a ``last.pt`` and runs it in the
masked-attention regime of the checkpoint's epoch (``--mask annealed``, from the run's args.yaml: the numbers of
metrics.csv) or mask-free (``--mask free``: as deployed). Without ``--ema`` the weights are those saved (mask-free).
The network runs under autocast on a GPU, as in the trainer's validation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from eomt import EoMT
from eomt.engine.train import _attn_mask_prob
from eomt.serialization import resolve_checkpoint


def add_tta_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("test-time augmentation / tiled inference (instance family; see eomt/tta.py)")
    g.add_argument("--augment", action="store_true", help="Also run the image at 1.5x imgsz and merge.")
    g.add_argument("--tiles", type=int, nargs="?", const=True, default=False, metavar="SIZE",
                   help="Overlapping tiles (SIZE px, default the model's imgsz) plus the whole image, merged.")
    g.add_argument("--flip", action="store_true",
                   help="Also run every view mirrored (alone: --augment with mirrored views). Heads the checkpoint marks "
                        "flip-aware are swapped back; an older checkpoint takes the mapping from the dataset's JSON, with a warning.")


def use_ema(model: EoMT, ckpt: Path, mask: str) -> str:
    """Load the EMA weights of a ``last.pt`` into ``model`` and set its masked-attention regime; returns what was done."""
    state = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
    if "ema" not in state:
        raise SystemExit(f"{ckpt} has no EMA state (best.pt is already the EMA); drop --ema.")
    core = model.model
    core.load_state_dict(state["ema"]["module"])
    probs = [0.0] * core.eomt.attn_mask_probs.numel()
    if mask == "annealed":
        args = ckpt.parent.parent / "args.yaml"
        if not args.is_file():
            raise SystemExit(f"--mask annealed needs the run's {args}; use --mask free.")
        run = yaml.safe_load(args.read_text())
        frac = (state["epoch"] + 1) / run["epochs"]
        probs = [_attn_mask_prob(frac * 1e6, 1e6, a, b, run["poly_power"])  # x1e6: the schedule works in iterations
                 for a, b in zip(run["mask_anneal_start"], run["mask_anneal_end"])]
    core.eomt.set_attn_mask_probs(probs)
    return f"EMA of epoch {state['epoch']}, masked-attention probabilities {[round(x, 3) for x in probs]}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("weights", help="Checkpoint .pt or a run/weights folder (best.pt preferred).")
    p.add_argument("--data", default="coco", help="Dataset YAML path or alias ('coco').")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--workers", type=int, default=4, help="Data-loader workers.")
    p.add_argument("--imgsz", type=int, default=None, help="Evaluation input size (default: the trained size).")
    p.add_argument("--device", default="auto")
    p.add_argument("--no-attributes", action="store_true", help="Skip the secondary heads' metrics (mAP only).")
    p.add_argument("--ema", action="store_true", help="Validate the EMA weights of a last.pt (the trainer's regime).")
    p.add_argument("--mask", choices=("annealed", "free"), default="annealed",
                   help="With --ema: masked attention at the checkpoint's annealing state (matches metrics.csv) or off (deployed).")
    add_tta_args(p)
    args = p.parse_args()
    model = EoMT(args.weights, device=args.device)
    if args.ema:  # a run folder resolves to its last.pt, the only file that holds the EMA copy
        ckpt = resolve_checkpoint(args.weights, prefer="last")
        print(f"[val] {ckpt}: {use_ema(model, ckpt, args.mask)}")
    augment = {"flip": True} if args.flip else args.augment  # flip alone: the 1x / 1.5x whole image, mirrored too
    metrics = model.val(data=args.data, batch=args.batch, workers=args.workers, imgsz=args.imgsz, augment=augment,
                        tiles=args.tiles, attributes=not args.no_attributes, amp=True)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()

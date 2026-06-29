#!/usr/bin/env python
"""Upload an EoMT checkpoint to the Hugging Face Hub.

    python scripts/push_to_hub.py runs/train/segment-coco-s imagra93/eomt-segment-coco-s
    python scripts/push_to_hub.py runs/train/detect-coco-s imagra93/eomt-detect-coco-s --public
    python scripts/push_to_hub.py runs/train/detect-coco-s imagra93/eomt-detect-coco-s \
        --filename model-best.pt -m "detect coco-s checkpoint"

``weights`` points at anything ``EoMT(...)`` accepts: a ``.pt`` file, a run folder
(``best.pt`` is preferred), or a ``run/weights`` dir. It is loaded and re-saved as a
self-describing checkpoint (size/task/classes/preprocessing all embedded), so the
uploaded file reloads cleanly with ``EoMT.from_pretrained(repo_id)``.

Repos are **private by default**; pass ``--public`` to publish openly. The Hub token
is read from ``HUGGINGFACE_TOKEN`` / ``HF_TOKEN`` (env or a local ``.env``), falling
back to a cached ``huggingface-cli login``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _load_token() -> None:
    """Make a Hub token visible to huggingface_hub as ``HF_TOKEN``.

    Accepts the repo's ``HUGGINGFACE_TOKEN`` name (and a local ``.env``) and mirrors
    it into ``HF_TOKEN``, which is what huggingface_hub actually reads. A token
    already exported, or a cached ``huggingface-cli login``, is left untouched.
    """
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"):
        token = os.environ.get("HF_TOKEN") or os.environ["HUGGINGFACE_TOKEN"]
    else:
        token = None
        env_file = Path(__file__).resolve().parent.parent / ".env"
        if env_file.is_file():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith(("HUGGINGFACE_TOKEN", "HF_TOKEN")) and "=" in line:
                    token = line.split("=", 1)[1].strip().strip("'\"")
                    break
    if token:
        os.environ.setdefault("HF_TOKEN", token)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("weights", help="Checkpoint .pt, a run folder, or a run/weights dir.")
    p.add_argument("repo_id", help="Target Hub repo, e.g. 'imagra93/eomt-segment-coco-s'.")
    p.add_argument("--filename", default="model.pt", help="Name for the file in the repo.")
    vis = p.add_mutually_exclusive_group()
    vis.add_argument("--private", dest="private", action="store_true", default=True,
                     help="Create a private repo (default).")
    vis.add_argument("--public", dest="private", action="store_false",
                     help="Create a public repo.")
    p.add_argument("-m", "--message", default="Upload EoMT checkpoint", help="Commit message.")
    args = p.parse_args()

    _load_token()

    from eomt import EoMT

    model = EoMT(args.weights, device="cpu")
    url = model.push_to_hub(
        args.repo_id, filename=args.filename, private=args.private,
        commit_message=args.message,
    )
    visibility = "private" if args.private else "public"
    print(f"uploaded {args.filename} -> {url} ({visibility})")
    print(f"reload with: EoMT.from_pretrained({args.repo_id!r}"
          + (f", filename={args.filename!r}" if args.filename != "model.pt" else "") + ")")


if __name__ == "__main__":
    main()

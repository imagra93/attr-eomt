#!/usr/bin/env bash
# Validate EoMT checkpoint(s) on COCO (segm + bbox mAP via pycocotools).
#
# Usage:
#   scripts/val_coco.sh path/to/best.pt          # validate one checkpoint
#   scripts/val_coco.sh                           # validate all trained sizes
#                                                 # (runs/train/eomt_<size>/weights/best.pt)
#   scripts/val_coco.sh best.pt --conf-thres 0.0  # extra args forwarded to `eomt val`
#
# Env overrides: DATA, DEVICE, BATCH, PROJECT, SIZES.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA="${DATA:-configs/coco.yaml}"
DEVICE="${DEVICE:-auto}"
BATCH="${BATCH:-4}"
PROJECT="${PROJECT:-runs/train}"
SIZES="${SIZES:-s b l}"

run_val() { # checkpoint [extra args...]
  local ckpt="$1"; shift || true
  echo "=== validating ${ckpt} ==="
  eomt val --weights "$ckpt" --data "$DATA" --device "$DEVICE" --batch "$BATCH" "$@"
}

if [[ $# -gt 0 && -f "$1" ]]; then
  ckpt="$1"; shift
  run_val "$ckpt" "$@"
else
  # No explicit checkpoint: validate every trained size that exists.
  found=0
  for size in $SIZES; do
    ckpt="${PROJECT}/eomt_${size}/weights/best.pt"
    if [[ -f "$ckpt" ]]; then
      found=1
      run_val "$ckpt" "$@"
    else
      echo "skip eomt_${size}: ${ckpt} not found"
    fi
  done
  [[ "$found" -eq 1 ]] || { echo "No checkpoints found. Pass one explicitly: scripts/val_coco.sh <ckpt>"; exit 1; }
fi

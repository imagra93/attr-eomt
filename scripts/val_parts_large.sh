#!/usr/bin/env bash
# Validate an EoMT checkpoint on the vehicle-parts val set (segm + bbox mAP via
# pycocotools). Validation has no augmentation, so the flip setting does not apply
# here — this is just the eval counterpart of scripts/train_parts_large.sh.
#
# Usage:
#   scripts/val_parts_large.sh                                    # auto: PROJECT/NAME/weights/best.pt
#   scripts/val_parts_large.sh runs/train/eomt-l/weights/best.pt  # explicit checkpoint
#   DEVICE=cuda:1 BATCH=4 scripts/val_parts_large.sh
#   scripts/val_parts_large.sh best.pt --conf-thres 0.0          # extra args -> eomt val
#
# Env overrides: DATA, SIZE, DEVICE, BATCH, PROJECT, NAME.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA="${DATA:-data/parts/main_coco/data.yaml}"
SIZE="${SIZE:-l}"
DEVICE="${DEVICE:-auto}"
BATCH="${BATCH:-4}"
PROJECT="${PROJECT:-runs/train}"
NAME="${NAME:-eomt-${SIZE}-parts}"

if [[ $# -gt 0 && -f "$1" ]]; then
  CKPT="$1"; shift
else
  CKPT="${PROJECT}/${NAME}/weights/best.pt"
fi
[[ -f "$CKPT" ]] || { echo "checkpoint not found: $CKPT (pass one explicitly)"; exit 1; }

echo "=== validating ${CKPT} on ${DATA} ==="
eomt val --weights "$CKPT" --data "$DATA" --device "$DEVICE" --batch "$BATCH" "$@"

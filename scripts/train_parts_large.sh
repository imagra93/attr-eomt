#!/usr/bin/env bash
# Train EoMT on the vehicle-parts dataset (data/parts/main_coco) with horizontal
# flips DISABLED.
#
# Why flips are off here: this dataset carries a per-instance left/right
# "laterality" attribute (a secondary head). A plain horizontal flip mirrors the
# image WITHOUT swapping left<->right, which corrupts that supervision — so this
# dataset REQUIRES --flip-prob 0. (Plain COCO has no such attribute; see
# scripts/train_all_coco.sh, which keeps flips on.)
#
# Usage:
#   scripts/train_parts_large.sh                      # large, defaults
#   DEVICE=cuda:1 scripts/train_parts_large.sh        # pick a GPU
#   SIZE=b EPOCHS=50 BATCH=4 scripts/train_parts_large.sh
#   scripts/train_parts_large.sh --lr0 5e-5           # extra args -> eomt train
#
# Env overrides: DATA, SIZE, DEVICE, EPOCHS, BATCH, PROJECT, NAME.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA="${DATA:-data/parts/main_coco/data.yaml}"
SIZE="${SIZE:-l}"
DEVICE="${DEVICE:-auto}"
EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-2}"
PROJECT="${PROJECT:-runs/train}"
NAME="${NAME:-eomt-${SIZE}-parts}"

echo "Training EoMT-${SIZE} on ${DATA} with flips OFF -> ${PROJECT}/${NAME}"
eomt train \
  --data "$DATA" \
  --size "$SIZE" \
  --flip-prob 0 \
  --epochs "$EPOCHS" \
  --batch "$BATCH" \
  --device "$DEVICE" \
  --project "$PROJECT" \
  --name "$NAME" \
  "$@"

echo "Done. Checkpoints under ${PROJECT}/${NAME}/weights/."

#!/usr/bin/env bash
# Train all three EoMT sizes (s, b, l) on COCO instance segmentation.
#
# COCO is auto-downloaded by configs/coco.yaml if missing. Each run lands in
# runs/train/eomt_<size>/ with weights/best.pt + weights/last.pt.
#
# Usage:
#   scripts/train_all_coco.sh                 # train s, b, l with defaults
#   SIZES="s l" scripts/train_all_coco.sh     # subset of sizes
#   EPOCHS=12 DEVICE=cuda:0 scripts/train_all_coco.sh
#   scripts/train_all_coco.sh --lr0 5e-5      # extra args forwarded to `eomt train`
#
# Env overrides: DATA, EPOCHS, DEVICE, PROJECT, SIZES,
#                BATCH_S, BATCH_B, BATCH_L (per-size batch sizes).
set -euo pipefail
cd "$(dirname "$0")/.."

DATA="${DATA:-configs/coco.yaml}"
EPOCHS="${EPOCHS:-5}"
DEVICE="${DEVICE:-auto}"
PROJECT="${PROJECT:-runs/train}"
SIZES="${SIZES:-s b l}"

# Larger backbones need a smaller batch; tune for your GPU memory.
declare -A BATCH=( [s]="${BATCH_S:-16}" [b]="${BATCH_B:-8}" [l]="${BATCH_L:-4}" )

for size in $SIZES; do
  echo "================================================================"
  echo "  Training EoMT-${size}  (data=${DATA}, epochs=${EPOCHS}, batch=${BATCH[$size]})"
  echo "================================================================"
  eomt train \
    --data "$DATA" \
    --size "$size" \
    --epochs "$EPOCHS" \
    --batch "${BATCH[$size]}" \
    --device "$DEVICE" \
    --project "$PROJECT" \
    --name "eomt_${size}" \
    "$@"
done

echo "Done. Checkpoints under ${PROJECT}/eomt_<size>/weights/."

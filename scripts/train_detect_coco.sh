#!/usr/bin/env bash
# Sanity-check the EoMT *detect* head on COCO 2017.
#
# This trains the large detect model (box head, no masks) on the standard COCO
# instance-detection benchmark. The point is to confirm the `family="detect"`
# pipeline is correct on a known-good dataset: if bbox mAP climbs into the
# expected range here, the detect code is fine and any trouble on a custom set
# (e.g. fuel-project) is a data/setup issue, not a bug in the head.
#
# COCO 2017 auto-downloads to ./datasets/coco on first run (~20 GB) via
# configs/coco.yaml (download: true).
#
# Device defaults to 'auto', which picks the CUDA GPU with the most free memory
# (handy when GPU 0 is busy). Pin one with DEVICE=cuda:1 if you prefer.
# Logs/metrics/weights land in runs/train/detect-coco-l/.
#
#   bash scripts/train_detect_coco.sh                 # defaults (auto GPU)
#   DEVICE=cuda:0 EPOCHS=12 bash scripts/train_detect_coco.sh
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-auto}"
SIZE="${SIZE:-b}"
EPOCHS="${EPOCHS:-300}"
BATCH="${BATCH:-8}"          # large model @ 644px; engine auto-accumulates to eff. batch 16
IMGSZ="${IMGSZ:-644}"
NAME="${NAME:-detect-coco-${SIZE}}"

echo "[train_detect_coco] size=${SIZE} device=${DEVICE} epochs=${EPOCHS} batch=${BATCH} imgsz=${IMGSZ} name=${NAME}"

python scripts/train.py \
    --data coco \
    --task detect \
    --size "${SIZE}" \
    --epochs "${EPOCHS}" \
    --batch "${BATCH}" \
    --imgsz "${IMGSZ}" \
    --device "${DEVICE}" \
    --name "${NAME}"

echo "[train_detect_coco] done -> runs/train/${NAME}/weights/best.pt"
echo "[train_detect_coco] validate with: python scripts/val.py runs/train/${NAME} --data coco"

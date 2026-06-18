#!/usr/bin/env bash
# Load a checkpoint and render instance-segmentation predictions on two sample
# images. The images are downloaded on first run (defaults: COCO val2017 images;
# the first is the canonical image used in HuggingFace's EoMT examples).
#
# Usage:
#   scripts/infer_sample.sh path/to/best.pt
#   CONF=0.4 DEVICE=cuda:0 scripts/infer_sample.sh best.pt
#   IMG1_URL=... IMG2_URL=... scripts/infer_sample.sh best.pt   # custom images
#   scripts/infer_sample.sh best.pt --no-draw-boxes             # forwarded to predict
#
# Env overrides: IMG_DIR, OUT_DIR, CONF, DEVICE, IMG1_URL, IMG2_URL.
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT="${1:?usage: scripts/infer_sample.sh <checkpoint.pt> [extra eomt predict args]}"
shift || true

IMG_DIR="${IMG_DIR:-scripts/sample_images}"
OUT_DIR="${OUT_DIR:-runs/predict}"
CONF="${CONF:-0.3}"
DEVICE="${DEVICE:-auto}"

# Two COCO-domain demo images (override with IMG1_URL / IMG2_URL).
IMG1_URL="${IMG1_URL:-http://images.cocodataset.org/val2017/000000039769.jpg}"  # cats on a couch
IMG2_URL="${IMG2_URL:-http://images.cocodataset.org/val2017/000000000139.jpg}"  # indoor scene

mkdir -p "$IMG_DIR"

download() { # url dest
  local url="$1" dest="$2"
  if [[ -f "$dest" ]]; then echo "  have ${dest}"; return; fi
  echo "  downloading ${url} -> ${dest}"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$url" -o "$dest"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "$dest" "$url"
  else
    echo "Need curl or wget to download sample images." >&2; exit 1
  fi
}

download "$IMG1_URL" "$IMG_DIR/sample1.jpg"
download "$IMG2_URL" "$IMG_DIR/sample2.jpg"

eomt predict \
  --weights "$CKPT" \
  --source "$IMG_DIR" \
  --out "$OUT_DIR" \
  --conf-thres "$CONF" \
  --device "$DEVICE" \
  "$@"

echo "Annotated images written to ${OUT_DIR}/."

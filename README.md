# libre-eomt

Standalone **EoMT** (Encoder-only Mask Transformer) for **instance segmentation**,
extracted into a small installable package. It builds the architecture in three
sizes, initializes the ViT encoder from **DINOv2-with-registers**, and provides
training, per-epoch COCO-mAP validation, and inference/rendering.

EoMT is a DINOv2 ViT whose last few transformer blocks are augmented with
learnable queries producing mask-classification output (Mask2Former-style).
This package wraps the Apache-2.0 HuggingFace `EomtForUniversalSegmentation`;
weights trained from a DINOv2 initialization are yours to release.

## Sizes

| size | backbone        | hidden | layers | heads | queries |
|------|-----------------|--------|--------|-------|---------|
| `s`  | DINOv2-small    | 384    | 12     | 6     | 100     |
| `b`  | DINOv2-base     | 768    | 12     | 12    | 200     |
| `l`  | DINOv2-large    | 1024   | 24     | 16    | 200     |

Default input is a patch-14-aligned square (`644 = 14 × 46`) so DINOv2 weights
load 1:1.

## Install

```bash
pip install -e .              # add [logging] for tensorboard/wandb, [dev] for pytest
```

## Usage

### Python

```python
from eomt import build_model, load_dinov2_backbone, load_model

model = build_model("s", nc=80)     # build architecture
load_dinov2_backbone(model)         # init encoder from DINOv2 (downloads weights)

seg = load_model("runs/train/eomt/weights/best.pt")   # reload a trained checkpoint
```

### CLI

```bash
# Train (auto-downloads COCO 2017 if configs/coco.yaml points at a missing dir)
eomt train --data configs/coco.yaml --size s --epochs 50 --batch 4 --device cuda

# Validate a checkpoint (COCO segm + bbox mAP)
eomt val --weights runs/train/eomt/weights/best.pt --data configs/coco.yaml

# Render predictions on an image or a folder
eomt predict --weights best.pt --source path/to/imgs --out runs/predict

# Just fetch COCO 2017
eomt download --root datasets/coco
```

You can point training/validation at **any COCO-format dataset** by passing
`--train-images/--train-json/--val-images/--val-json` directly, or by editing
`configs/coco.yaml`.

## What's included

- Architecture (`s`/`b`/`l`) + DINOv2 init — `eomt.model`, `eomt.config`
- COCO-format datasets, augmentations (torchvision v2), autodownload — `eomt.data`
- Training loop (AdamW + backbone LR mult, cosine warmup, AMP, ckpt) — `eomt.engine.train`
- COCO-mAP validation (`pycocotools`) — `eomt.engine.validate`
- Inference + rendering — `eomt.engine.predict`, `eomt.visualize`

## Not included

Deployment-format export (ONNX / TensorRT / etc.) is intentionally out of scope.
A detect (true box head) family and a semantic-segmentation family are planned;
the code carries a `family` parameter so they can be added without API churn, but
only `instance` is implemented today.

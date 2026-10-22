# Sample dataset

A tiny, illustrative COCO-format dataset showing how attr-eomt reads data —
including the **secondary per-instance classification** (attributes) extension.

```
sample_data/
├── data.yaml                         # dataset config: the paths only
├── run.yaml                          # run config: the paths + heads, augmentation, training, inference
├── annotations/
│   ├── instances_train.json          # 2 images, 3 instances, 2 attribute heads
│   └── instances_val.json            # 1 image, 2 instances
└── images/{train,val}/               # YOUR .jpg files — not shipped here
```

> The `images/` folders are intentionally empty: this is a **schema example**, not
> a runnable dataset. Drop the referenced `file_name`s in to train for real.

## What it demonstrates

Clothing instances (the same example shown in the top-level README): one model
segments each garment and reads off two independent attributes per detection.

- **Primary classes** come from `categories` (`short_sleeve_top`,
  `long_sleeve_dress`, `vest_dress`, `skirt`, `trousers`) — `nc` is inferred.
- **Two secondary heads** (`scale`, `viewpoint`) are declared once at the top
  level under `attributes`, then assigned per instance via each annotation's
  `attributes: {head: id}` map.
- Raw attribute ids are remapped to a contiguous `0..n-1` per head, so the
  non-contiguous `scale` ids (`1`, `2`, `3`) become `0`, `1`, `2` automatically.
- Remove the `attributes` blocks entirely and it's an ordinary instance-seg
  dataset — the secondary heads simply don't get built.

The head names (`scale`, `viewpoint`) are just examples; define whatever
attributes your task needs.

The JSONs hold only data. How a model learns them is the **run config**,
[`run.yaml`](run.yaml): which attributes get a head and how each one behaves
(`applies_to`, `flip_aware`, a regression's `log` / `range`, `crop`), the
augmentation, the `train()` settings and the inference settings saved in every
checkpoint, in one commented file. Copy it for your own dataset.

## Try it

```bash
python scripts/train.py --data sample_data/run.yaml --size s              # everything from run.yaml
python scripts/train.py --data sample_data/run.yaml --size s --epochs 1   # a flag beats the file
```

```python
from eomt import EoMT

EoMT("s").train(data="sample_data/run.yaml", epochs=1, batch=1)   # keywords beat the file too
```

Per-epoch train/val metrics (segmentation mAP and, when present, per-head
attribute accuracy) are written to `runs/train/sample/metrics.csv` (run.yaml's `project` / `name`).

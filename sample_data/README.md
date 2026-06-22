# Sample dataset

A tiny, illustrative COCO-format dataset showing how libre-eomt reads data —
including the **secondary per-instance classification** (attributes) extension.

```
sample_data/
├── data.yaml                         # dataset config (point --data here)
├── annotations/
│   ├── instances_train.json          # 2 images, 3 instances, 2 attribute heads
│   └── instances_val.json            # 1 image, 2 instances
└── images/{train,val}/               # YOUR .jpg files — not shipped here
```

> The `images/` folders are intentionally empty: this is a **schema example**, not
> a runnable dataset. Drop the referenced `file_name`s in to train for real.

## What it demonstrates

A bowl of fruit (the same example shown in the top-level README): one model
segments each fruit and reads off two independent attributes per detection.

- **Primary classes** come from `categories` (`apple`, `banana`, `orange`, `pear`)
  — `nc` is inferred.
- **Two secondary heads** (`ripeness`, `grade`) are declared once at the top
  level under `attributes`, then assigned per instance via each annotation's
  `attributes: {head: id}` map.
- Raw attribute ids are remapped to a contiguous `0..n-1` per head, so the
  non-contiguous `grade` ids (`10`, `20`) become `0`, `1` automatically.
- Remove the `attributes` blocks entirely and it's an ordinary instance-seg
  dataset — the secondary heads simply don't get built.

The head names (`ripeness`, `grade`) are just examples; define whatever
attributes your task needs.

## Try it

```python
from eomt import EoMT

EoMT("s").train(data="sample_data/data.yaml", epochs=1, batch=1)
```

Per-epoch train/val metrics (segmentation mAP and, when present, per-head
attribute accuracy) are written to `runs/train/eomt-s/metrics.csv`.

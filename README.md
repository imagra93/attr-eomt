# attr-eomt

Standalone **EoMT** (Encoder-only Mask Transformer) for **instance segmentation**
and **object detection**, extended with one feature that sets it apart:
**secondary per-instance classification heads** ("auxiliary classes"). Alongside the
usual mask/box + class output, the model predicts **one or several independent
attributes for every detected instance** — and they train and infer for free on top
of the detector, without inflating the primary class space.

EoMT itself is a DINOv2-with-registers ViT whose last few transformer blocks are
augmented with learnable queries (Mask2Former-style). This package builds it in three
sizes, initializes the encoder from DINOv2, and provides training, per-epoch COCO-mAP
validation, and inference/rendering — all behind a small `EoMT` class. It is a
clean-room, Apache-2.0-compatible reimplementation; weights you train are yours to
release.

```python
from eomt import EoMT

model = EoMT("l")                            # fresh large model (DINOv2 backbone)
model.train(data="coco", epochs=50)          # COCO 2017 auto-downloads if missing

model = EoMT("runs/train/eomt-l")            # reload a run — size/classes/heads auto-detected
model.predict("images/", plot=True)          # render masks/boxes + per-instance attributes
```

---

## Two model families: segmentation & detection

Both families share the same DINOv2 encoder, query mechanism, NMS-free matching and
auxiliary heads — they differ only in the head on top and what they output:

| family | `--task` | output | metric driving `best.pt` |
|--------|----------|--------|--------------------------|
| **instance** (default) | `instance` | per-instance **masks** + boxes + class | `segm/mAP` |
| **detect** | `detect`  | per-instance **boxes** + class (DETR-style box head, no masks) | `bbox/mAP` |

```python
EoMT("l").train(data="coco", family="instance")   # masks (default)
EoMT("l").train(data="coco", family="detect")     # boxes only
```

The family is recorded in the checkpoint, so `val` / `predict` pick the right
post-processing automatically. Everything below (sizes, auxiliary heads, recipe)
applies to both.

---

## ⭐ Auxiliary per-instance attribute heads

The primary task is **unchanged**: instance segmentation (or detection) over `nc`
classes. On top of it you can attach **one or several independent secondary
classifiers** — a per-instance *attribute* predicted for each detected object, read
straight from that query's embedding. You can have as many heads as your data
defines.

### Why it matters

Conventional detectors fold every distinction into a single flat label space. When an
object has both a *type* and several *attributes* — a car's make **and** its colour
**and** whether a door is open — the only way to express that is the Cartesian product
`type × colour × state × …`. That space explodes combinatorially, starves each leaf
class of training examples, and breaks Hungarian matching by multiplying the query
targets.

**attr-eomt factorizes the label space instead.** The primary head stays small and
*general*; orthogonal attributes are predicted by independent secondary heads, each
reading the same per-instance query embedding the detector already computes. Because
the heads are independent:

- **Categories stay collapsed and general.** Keep a compact, well-populated primary
  taxonomy (`car`, `person`, `fruit`) and push fine-grained or orthogonal distinctions
  into attributes — every primary class keeps its full sample count instead of being
  shattered into rare leaves.
- **Unseen combinations generalize.** A `colour` head trained across many object types
  predicts `red` on a class it *never co-occurred with in red*, because colour is
  learned independently of type. The model composes `attribute × class` combinations
  that **never appear in the training data** — combinations a flat label space cannot
  even represent.
- **Instances stay separate.** EoMT is **NMS-free**, so two overlapping same-class
  objects remain two distinct queries; each carries its own attribute predictions
  rather than being merged.
- **It rides along for free.** Attributes reuse the backbone and the detector's own
  matched queries — they add only a thin linear/MLP head and a cross-entropy term, not
  a second model or a second pass.

### Example: a bowl of fruit

One model segments each fruit (primary classes `apple` / `banana` / `orange` /
`pear`) and, for **every** detection, reads off two **independent** attribute heads —
`ripeness` (`unripe` / `turning` / `ripe`) and a quality `grade` (`A` / `B`). The
renderer prints the primary class + score on the first row and each attribute + its
confidence on the row beneath it.

![A bowl of fruit; each box labelled with its fruit class plus a ripeness and a grade](docs/examples/example_1.png)

`ripeness` and `grade` are *orthogonal* — they vary independently — which is exactly
the case that's awkward to fold into the primary class space. The same pattern fits
any "class **plus** per-instance sub-labels" task: **retail shelves → product +
facing**, **vehicles → type + colour + damage state**, **cells / leaves → type +
health**, **apparel → garment + pattern**.

> The image above is a **simulation**: a stock photo with hand-placed detections fed
> through the package's own renderer ([`eomt.visualize.draw_instances`](eomt/visualize.py))
> to show the output format — not a trained model's predictions.

### How it works

- **Embedding source.** Each head reads the per-query embedding — the input to EoMT's
  `class_predictor`, captured with a forward hook (`[B, Q, hidden]`).
- **Matching.** Supervision reuses EoMT's *own* Hungarian matcher
  (`model.eomt.criterion.matcher`), so every attribute is trained on the **same**
  query→GT assignment the detection loss used; the attribute is read *after* matching.
- **Loss.** Cross-entropy per head over matched queries, summed across heads and scaled
  by `aux_w` (default `1.0`), added to the detector loss. Empty-match batches contribute
  a graph-preserving zero.
- **Checkpoint selection is unchanged.** The attribute "rides along": its per-head
  matched-query accuracy is shown live and written to `metrics.csv`, but never drives
  `best.pt` (still `segm/mAP` or `bbox/mAP`).
- **Inference.** Each result attaches `aux = {head: {"ids", "probs"}}` for the kept
  detections, and `predict(plot=True)` renders each attribute next to the class label
  using names stored in the checkpoint.

### Data format (auto-discovered from the COCO JSON)

Attributes live **inside the COCO annotations** — each annotation is already a
per-instance object, so alignment is automatic and `pycocotools` still parses it. Just
two additions to a standard COCO file.

**1. A top-level `attributes` list** — one entry per head, defining its vocabulary:

```jsonc
"attributes": [
  {
    "name": "ripeness",
    "categories": [
      {"id": 0, "name": "unripe"},
      {"id": 1, "name": "turning"},
      {"id": 2, "name": "ripe"}
    ]
  },
  {
    "name": "grade",
    "categories": [
      {"id": 10, "name": "A"},
      {"id": 20, "name": "B"}
    ]
  }
]
```

**2. A per-annotation `attributes` map** — `{head: raw_id}` on each instance:

```jsonc
{
  "id": 1, "image_id": 42, "category_id": 1,
  "segmentation": [...], "bbox": [...], "area": 1234, "iscrowd": 0,
  "attributes": {"ripeness": 1, "grade": 20}
}
```

Notes:

- Raw ids are remapped to a contiguous `0..n-1` per head (so `grade`'s `10`/`20` become
  `0`/`1`); `categories` may be omitted, in which case the id set is inferred.
- A missing per-annotation value defaults to `0`; a JSON with **no** `attributes` ⇒
  detection-only, exactly as before.
- No YAML changes needed — heads (count, classes, names) are discovered from the JSON,
  the same as `nc`.

A tiny, self-contained example (two heads, including a non-contiguous id set) lives in
[sample_data/](sample_data/).

---

## Models & sizes

| size | backbone        | hidden | layers | heads | queries |
|------|-----------------|--------|--------|-------|---------|
| `s`  | DINOv2-small    | 384    | 12     | 6     | 100     |
| `b`  | DINOv2-base     | 768    | 12     | 12    | 200     |
| `l`  | DINOv2-large    | 1024   | 24     | 16    | 200     |

Default input is a patch-14-aligned square (`644 = 14 × 46`) so DINOv2 weights load 1:1.

### Install

```bash
pip install attr-eomt                  # from PyPI
pip install "attr-eomt[logging]"       # + tensorboard/wandb
pip install -e ".[dev]"                # from source (editable; [dev] adds pytest/build/twine)
```

---

## Usage

Everything goes through one class. Initialize from a **size** (fresh model, pretrained
DINOv2 backbone) or from a **checkpoint / run folder** (family, size, classes, image
size, normalization and any auxiliary heads are auto-detected from the `.pt`):

```python
from eomt import EoMT

# Train on COCO 2017 (auto-downloaded on first run):
EoMT("l").train(data="coco", epochs=50, batch=4)

# ...or any COCO-format dataset (point at its data.yaml):
EoMT("s").train(data="sample_data/data.yaml", epochs=1, batch=1)

# Validate and predict from a trained run:
EoMT("runs/train/eomt-l").val(data="coco")
EoMT("runs/train/eomt-l").predict("images/", plot=True)   # writes annotated images
```

### Useful `train()` options

Passed as keyword arguments to `model.train(...)`:

| arg | default | effect |
|------|---------|--------|
| `family` | `"instance"` | `"instance"` (masks) or `"detect"` (boxes only) |
| `nominal_batch` / `accum` | `16` / `0` | gradient accumulation to an **effective batch** of `nominal_batch`; `accum=N` sets the step count explicitly |
| `ema` / `ema_decay` / `ema_tau` | `True` / `0.9999` / `2000` | validate & export `best.pt` from an **EMA** of the weights |
| `llrd` | `0.85` | **layer-wise LR decay** on the DINOv2 backbone (`1.0` = flat `backbone_lr_mult`) |
| `min_scale` / `max_scale` | `0.1` / `2.0` | **Large-Scale Jitter** range |
| `letterbox` | `True` | aspect-preserving **letterbox** eval (vs legacy square stretch); recorded in the checkpoint |
| `flip_prob` | `0.5` | horizontal-flip probability. **Set `0`** for datasets with a left/right attribute — hflip mirrors pixels without swapping the laterality label |
| `mask_anneal` | `True` | anneal masked attention `1→0` over training (EoMT recipe) |
| `aux_w` | `1.0` | weight on the summed secondary-head loss |

### Training recipe (defaults)

Defaults follow the EoMT/Mask2Former fine-tuning recipe; each piece is a keyword
argument, so legacy behaviour is one override away.

- **Effective batch via gradient accumulation** — LR/WD/clip are tuned for an effective
  batch of 16; EoMT is a ViT (LayerNorm, no BatchNorm) so accumulation ≈ a true large
  batch at a fraction of the memory.
- **EMA weights** validated and saved as `best.pt`; `last.pt` holds live weights +
  optimizer + EMA state for exact resume.
- **Large-Scale Jitter** for training, **letterbox** for eval/inference (mode stored
  per-checkpoint so `val`/`predict` match training automatically).
- **AdamW** with no weight decay on norms/biases/embeddings and layer-wise LR decay on
  the backbone.
- **Masked-attention annealing** `1 → 0` so the final stretch trains mask-free and
  matches efficient (mask-less) inference.

---

## Roadmap / future work

- **Model export.** ONNX / TensorRT (and friends) for deployment — currently out of
  scope; the inference path is being kept export-friendly.
- **Keypoints.** A keypoint/pose head family alongside `instance` and `detect` (the code
  already carries a `family` parameter so new heads slot in without API churn).
- **Pretrained COCO checkpoints.** None are published yet. COCO-trained `s`/`b`/`l`
  weights will be released on the Hugging Face Hub (the `from_pretrained` / `hf://`
  loading plumbing is already in place and waiting for them).
- **Multi-image re-ID via contrastive learning.** Train the auxiliary head with a
  contrastive objective so each instance's query embedding becomes a **re-identification
  vector** — matching the same object across images, frames and cameras for tracking and
  retrieval. The aux head already produces a per-instance embedding from the detector's
  own matched queries; re-ID reuses that signal instead of bolting on a separate model.
```

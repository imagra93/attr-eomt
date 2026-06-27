<div align="center">

<img src="https://raw.githubusercontent.com/imagra93/attr-eomt/main/docs/assets/00-hero-banner.png" alt="attr-eomt — Encoder-only Mask Transformer with per-instance attribute heads" width="100%">

<p>
  <a href="https://pypi.org/project/attr-eomt/"><img src="https://img.shields.io/pypi/v/attr-eomt.svg?color=4ec9b0" alt="PyPI version"></a>
  <a href="https://pypi.org/project/attr-eomt/"><img src="https://img.shields.io/pypi/pyversions/attr-eomt.svg" alt="Python versions"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License"></a>
  <a href="https://imagra93.github.io/attr-eomt"><img src="https://img.shields.io/badge/docs-annotated%20explainer-e2b341.svg" alt="Annotated explainer"></a>
</p>

**One query embedding, many independent labels.**

📖 **[Read the annotated explainer →](https://imagra93.github.io/attr-eomt)**

</div>

---

## Abstract

**attr-eomt** is a standalone **EoMT** (Encoder-only Mask Transformer) for **instance
segmentation** and **object detection**, extended with one feature that sets it apart:
**independent per-instance attribute heads**. Alongside the usual mask/box + class
output, the model predicts **one or several orthogonal attributes for every detected
instance** — colour, state, grade, laterality, anything — read straight off the *same*
per-query embedding the detector already computes. Instead of folding every distinction
into one combinatorial class list, attr-eomt **factorizes the label space**: a small,
well-populated primary taxonomy plus thin attribute heads that *add* rather than
*multiply*. Each head reuses the detector's own Hungarian match, so attributes train
and infer **for near-zero extra compute** — no second model, no second pass, and the
primary detection metric is left exactly as it was. The result is composable,
generalizing per-instance labels that can even predict `attribute × class` combinations
never seen during training.

It is a clean-room, Apache-2.0-compatible reimplementation: weights you train are yours
to release.

```python
from eomt import EoMT

model = EoMT("l")                            # fresh large model (DINOv2 backbone)
model.train(data="coco", epochs=50)          # COCO 2017 auto-downloads if missing

model = EoMT("runs/train/eomt-l")            # reload a run — size/classes/heads auto-detected
model.predict("images/", plot=True)          # render masks/boxes + per-instance attributes
```

---

## Architecture

EoMT is a **DINOv2-with-registers ViT** whose last few transformer blocks are augmented
with a fixed set of **learnable queries** (the Mask2Former idea) — each query is one
"slot" that latches onto one object instance. After the encoder runs, every query emits
a single vector, the **per-query embedding** of shape `[B, Q, hidden]`. The whole model
is then just "turn that embedding into predictions": a **class head** for the primary
label and a **mask/box head** for geometry. It is **NMS-free**, so two overlapping
garments stay two distinct queries instead of being merged — the property that lets
attributes stay attached to the right instance.

The attribute heads add nothing to this picture except themselves: they tap the **exact
same embedding** (captured non-invasively with a forward hook), each a small classifier
on top.

<div align="center">
<img src="https://raw.githubusercontent.com/imagra93/attr-eomt/main/docs/assets/01-architecture.png" alt="Architecture: image → DINOv2 ViT with learnable queries → per-query embedding, fanning out to the class head, mask/box head and several auxiliary attribute heads" width="100%">
</div>

### Two model families: segmentation & detection

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
post-processing automatically. Everything below applies identically to both.

### Models & sizes

| size | backbone        | hidden | layers | heads | queries |
|------|-----------------|--------|--------|-------|---------|
| `s`  | DINOv2-small    | 384    | 12     | 6     | 100     |
| `b`  | DINOv2-base     | 768    | 12     | 12    | 200     |
| `l`  | DINOv2-large    | 1024   | 24     | 16    | 200     |

Default input is a patch-14-aligned square (`644 = 14 × 46`) so DINOv2 weights load 1:1.

### Compute & inference speed

Measured on a single **NVIDIA GeForce RTX 5090**, `644 × 644` input, batch size 1.
GFLOPs are multiply-accumulates at that resolution (attention included); latency /
throughput are the median over 50 runs after warm-up, under `torch.amp.autocast`
(fp16) — the package's own inference path.

**`instance` family** (masks + boxes + class):

| size | params | GFLOPs | latency (fp16) | throughput (fp16) | throughput (fp32) |
|------|--------|--------|----------------|-------------------|-------------------|
| `s`  | 24.0 M | 128    | 8.4 ms         | 119 img/s         | 70 img/s          |
| `b`  | 93.9 M | 430    | 17.4 ms        | 58 img/s          | 32 img/s          |
| `l`  | 317 M  | 1144   | 30.2 ms        | 33 img/s          | 15 img/s          |

**`detect` family** (boxes + class, no mask head):

| size | params | GFLOPs | latency (fp16) | throughput (fp16) | throughput (fp32) |
|------|--------|--------|----------------|-------------------|-------------------|
| `s`  | 22.7 M | 89     | 2.9 ms         | 348 img/s         | 120 img/s         |
| `b`  | 88.6 M | 276    | 5.3 ms         | 190 img/s         | 60 img/s          |
| `l`  | 308 M  | 881    | 13.6 ms        | 74 img/s          | 21 img/s          |

Dropping the mask-upsampling head makes `detect` substantially lighter and ~1.3–3×
faster. Figures are for the detector itself (backbone + queries + heads); the
attribute heads add a thin linear/MLP per head and are negligible by design.

---

## ⭐ Method — factorizing the label space

This is the contribution. Conventional detectors fold every distinction into a single
flat label space. When an object has both a *type* and several *attributes* — a garment's
class **and** its viewpoint **and** whether it's occluded — the only way to express that is
the Cartesian product `type × viewpoint × occlusion × …`. That space explodes combinatorially,
starves each leaf class of training examples, and breaks Hungarian matching by
multiplying the query targets.

**attr-eomt factorizes instead.** The primary head stays small and *general*; orthogonal
attributes are predicted by independent secondary heads, each reading the same
per-instance query embedding the detector already computes — they **add, not multiply**.

<div align="center">
<img src="https://raw.githubusercontent.com/imagra93/attr-eomt/main/docs/assets/02-factorized-labels.png" alt="Flat label space (type × colour × state leaf classes) versus factorized independent heads (type + colour + state outputs)" width="100%">
</div>

Because the heads are independent:

- **Categories stay collapsed and general.** Keep a compact, well-populated primary
  taxonomy (`short_sleeve_top`, `dress`, `trousers`) and push fine-grained or orthogonal
  distinctions into attributes — every primary class keeps its full sample count instead
  of being shattered into rare leaves.
- **Unseen combinations generalize.** A `viewpoint` head trained across many garment types
  predicts `side` on a class it *never co-occurred with sideways*, because viewpoint is
  learned independently of type. The model composes `attribute × class` combinations
  that **never appear in the training data** — combinations a flat label space cannot
  even represent.
- **Instances stay separate.** EoMT is **NMS-free**, so two overlapping same-class
  objects remain two distinct queries; each carries its own attribute predictions
  rather than being merged.
- **It rides along for free.** Attributes reuse the backbone and the detector's own
  matched queries — they add only a thin linear/MLP head and a cross-entropy term, not
  a second model or a second pass.

### Example: clothing with per-instance attributes

One model segments each garment (primary classes like `vest_dress` / `short_sleeve_top`
/ `long_sleeve_dress` / `skirt` / `trousers` …) and, for **every** detection, reads off
four **independent** attribute heads — `scale` (`small` / `modest` / `large`),
`occlusion` (`no` / `slight` / `medium`), `zoom_in` (`no` / `medium` / `large`) and
`viewpoint` (`frontal` / `side` / `back`). The renderer prints the primary class + score
on the first row and each attribute + its confidence on the rows beneath it.

![Two people in dresses; each instance labelled with its garment class plus scale, occlusion, zoom and viewpoint attributes](https://raw.githubusercontent.com/imagra93/attr-eomt/main/docs/examples/sample.jpg)

The four attributes are *orthogonal* to the garment class — they vary independently —
which is exactly the case that's awkward to fold into the primary class space. The same
pattern fits any "class **plus** per-instance sub-labels" task: **retail shelves →
product + facing**, **documents → element + role**, **cells → type + health**.

> Trained on the public **[DeepFashion2](https://github.com/switchablenorms/DeepFashion2)**
> dataset (13 garment classes + 4 attribute heads) and rendered with the package's own
> renderer ([`eomt.visualize.draw_instances`](eomt/visualize.py)).

---

## Training — it rides on the detector's own match

Attributes never run their own matcher. Detection already solves "which query is
responsible for which ground-truth object" via the **Hungarian matcher**; attributes
simply reuse that same query→GT assignment and read the answer off the matched queries.

<div align="center">
<img src="https://raw.githubusercontent.com/imagra93/attr-eomt/main/docs/assets/03-training-flow.png" alt="Training flow: query_embed → Hungarian matcher (reused) → IoU gate → matched queries [N, hidden] → per-head cross-entropy → × aux_w → added to detector loss" width="100%">
</div>

- **Embedding source.** Each head reads the per-query embedding — the input to EoMT's
  `class_predictor`, captured with a forward hook (`[B, Q, hidden]`).
- **Matching.** Supervision reuses EoMT's *own* Hungarian matcher
  (`model.eomt.criterion.matcher`), so every attribute is trained on the **same**
  query→GT assignment the detection loss used; the attribute is read *after* matching.
- **Gate.** An optional IoU gate drops barely-overlapping matched pairs (common early in
  training) so attributes only learn from queries that actually localize the object.
- **Loss.** Cross-entropy per head over matched queries, summed across heads and scaled
  by `aux_w` (default `1.0`), added to the detector loss. Empty-match batches contribute
  a graph-preserving zero, and missing labels use `ignore_index` and contribute nothing.
- **Checkpoint selection is unchanged.** The attribute "rides along": its per-head
  matched-query accuracy is shown live and written to `metrics.csv`, but never drives
  `best.pt` (still `segm/mAP` or `bbox/mAP`).
- **Inference.** Each result attaches `aux = {head: {"ids", "probs"}}` for the kept
  detections, and `predict(plot=True)` renders each attribute next to the class label
  using names stored in the checkpoint.

---

## Data format (auto-discovered from the COCO JSON)

Attributes live **inside the COCO annotations** — each annotation is already a
per-instance object, so alignment is automatic and `pycocotools` still parses it. Just
two additions to a standard COCO file; **no YAML changes** — heads (count, classes,
names) are discovered from the JSON, the same as `nc`.

**1. A top-level `attributes` list** — one entry per head, defining its vocabulary:

```jsonc
"attributes": [
  {
    "name": "scale",
    "categories": [
      {"id": 1, "name": "small"},
      {"id": 2, "name": "modest"},
      {"id": 3, "name": "large"}
    ]
  },
  {
    "name": "viewpoint",
    "categories": [
      {"id": 0, "name": "frontal"},
      {"id": 1, "name": "side"},
      {"id": 2, "name": "back"}
    ]
  }
]
```

**2. A per-annotation `attributes` map** — `{head: raw_id}` on each instance:

```jsonc
{
  "id": 1, "image_id": 42, "category_id": 1,
  "segmentation": [...], "bbox": [...], "area": 1234, "iscrowd": 0,
  "attributes": {"scale": 3, "viewpoint": 0}
}
```

Notes:

- Raw ids are remapped to a contiguous `0..n-1` per head (so `scale`'s `1`/`2`/`3` become
  `0`/`1`/`2`); `categories` may be omitted, in which case the id set is inferred.
- A missing per-annotation value defaults to `0`; a JSON with **no** `attributes` ⇒
  detection-only, exactly as before.

A tiny, self-contained example (two heads, including a non-contiguous id set) lives in
[sample_data/](sample_data/).

---

## Install

```bash
pip install attr-eomt                  # from PyPI
pip install "attr-eomt[logging]"       # + tensorboard/wandb
pip install -e ".[dev]"                # from source (editable; [dev] adds pytest/build/twine)
```

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

<div align="center">

<img src="https://raw.githubusercontent.com/imagra93/attr-eomt/main/docs/assets/00-hero-banner.png" alt="attr-eomt — one DINOv2 encoder predicts instances plus independent per-instance attribute heads in a single pass, contrasted with flat combinatorial labels and a detector-plus-second-model pipeline" width="100%">

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

## What it is

**attr-eomt** is a standalone **EoMT** (Encoder-only Mask Transformer) for **instance
segmentation** and **object detection**, with one feature that sets it apart: **independent
per-instance attribute heads**. Alongside the mask/box + class output, it predicts one or
several orthogonal attributes for *every* detected instance — read straight off the **same**
per-query embedding the detector already computes. No second model, no second pass, and the
primary detection metric is untouched (the figure above tells the whole story).

It's a clean-room, Apache-2.0 reimplementation: the weights you train are yours to release.

```python
from eomt import EoMT

model = EoMT("l")                            # fresh large model (DINOv2 backbone)
model.train(data="coco")                     # COCO 2017 auto-downloads if missing; 12 epochs, the official recipe

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

This collapses what is classically a *two-stage* pipeline — detect, crop each box, run a
second classifier per crop — into a single pass. Attributes therefore cost only a thin head
each, see **full-image context** (not just a cropped box), and never inherit a second
model's cropping errors — the modern, single-stage formulation of the DETR / Mask2Former
lineage (see the figure at the top).

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

| size | backbone        | hidden | layers | heads | queries | params |
|------|-----------------|--------|--------|-------|---------|--------|
| `s`  | DINOv2-small    | 384    | 12     | 6     | 100     | 24 M   |
| `b`  | DINOv2-base     | 768    | 12     | 12    | 200     | 94 M   |
| `l`  | DINOv2-large    | 1024   | 24     | 16    | 200     | 317 M  |

Default input is a patch-14-aligned square (`644 = 14 × 46`) so DINOv2 weights load 1:1. The `detect` family drops the
mask-upsampling head and is lighter; each attribute head adds only a thin linear / MLP layer.

---

## Factorizing the label space

This is the contribution. Conventional detectors fold every distinction into one flat
label space: an object's `type × viewpoint × occlusion × …` becomes a Cartesian product of
leaf classes that explodes combinatorially, starves each leaf of examples, and multiplies
the Hungarian matcher's targets. **attr-eomt factorizes instead** — a small, general primary
head plus independent attribute heads that **add, not multiply**.

Because the heads are independent, the primary taxonomy stays compact and every class keeps
its full sample count; attributes ride along for near-zero compute; and the model composes
`attribute × class` combinations that **never appear in the training data** — combinations a
flat label space cannot even represent.

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

- **Embedding source.** Each head reads the per-query embedding — the input to EoMT's
  `class_predictor`, captured with a forward hook (`[B, Q, hidden]`).
- **Matching.** Supervision reuses EoMT's *own* Hungarian matcher
  (`model.eomt.criterion.matcher`), so every attribute is trained on the **same**
  query→GT assignment the detection loss used; the attribute is read *after* matching.
- **Gate.** An optional IoU gate drops barely-overlapping matched pairs (common early in
  training) so attributes only learn from queries that actually localize the object.
- **Loss.** Cross-entropy per head over matched queries (a regression head: Huber on its
  standardized value, see below), summed across heads and scaled
  by `aux_w` (default `1.0`), added to the detector loss. Empty-match batches contribute
  a graph-preserving zero, and missing labels use `ignore_index` and contribute nothing.
- **Balancing.** Both losses are O(1), so equal weights are a sane start. `aux_w_per_head`
  weights each head, `aux_class_weights=True` re-weights rare classes (rare values of a
  regression head, by smoothed density), and `aux_balance="uncertainty"` learns one weight
  per head (Kendall et al. 2018) with the detector loss as the fixed anchor. Each epoch
  prints every head's pull on the shared queries (its gradient norm over the detector's,
  `train/aux_grad/<head>`): well above 1 a head steers the encoder, far below it only reads it.
- **Checkpoint selection is unchanged.** The attribute "rides along": its per-head
  matched-query accuracy is shown live and written to `metrics.csv`, but never drives
  `best.pt` (still `segm/mAP` or `bbox/mAP`).
- **Inference.** Each result attaches `aux = {head: {"ids", "probs"}}` for the kept
  detections (and `aux_values = {head: values}` in real units for the regression heads), and
  `predict(plot=True)` renders each attribute next to the class label
  using names stored in the checkpoint.

---

## Data format (auto-discovered from the COCO JSON)

Attributes live **inside the COCO annotations** — each annotation is already a
per-instance object, so alignment is automatic and `pycocotools` still parses it. Just
two additions to a standard COCO file — heads (count, classes, names) are discovered from the
JSON, the same as `nc`. The dataset holds **only data** (each attribute's name, type and
categories); which attributes get a head and how each one learns (`applies_to`, `flip_aware`,
`log`, `range`, `crop`) is the **run config**'s `attributes` section — see *Run config* below.

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
- A **missing or out-of-vocab** per-annotation value is *ignored* (`-100`), not trained as
  class `0` — so a **partially tagged** dataset is valid: each head learns only from the
  instances that actually carry its value. A JSON with **no** `attributes` ⇒ detection-only,
  exactly as before.

**Class-conditional heads.** Give an attribute an optional `applies_to` list of primary-class
names or ids in the run config, and that head is only trained on — and only emitted for —
instances of those classes (hard routing on the primary class). Omit it and the head applies
to every class. So different attributes can attach to different classes, each with its own
label set, in one model:

```yaml
# run config
attributes:
  posture: {applies_to: [cat, dog]}
```

At inference a scoped head reports `ids = -1` ("not applicable") for detections whose class it
does not cover. The scope is stored in the checkpoint, so it survives reload.

**Flip-aware attributes.** A horizontal flip mirrors the photo but not the labels, so an attribute
that encodes a side (a left shoe shows up as a right one) is corrupted by `flip_prob > 0`. Give
it `flip_aware` in the run config, a map of category names to the ones they become on a horizontal
flip, and the training augmentation swaps them for every instance it flips (each pair swaps both
ways, so `{left: right}` is enough; unlisted categories stay as they are):

```jsonc
// data: the attribute's categories
{"name": "side", "categories": [{"id": 0, "name": "left"}, {"id": 1, "name": "center"}, {"id": 2, "name": "right"}]}
```

```yaml
# run config: how it learns
attributes:
  side: {flip_aware: {left: right}}
```

Mosaic, mixup and copy-paste flip each source image independently, and the swap follows each
instance. Only the horizontal flip is handled: keep `vflip_prob` and `rot90_prob` off for such an
attribute. It is categorical only. The mapping is stored in the checkpoint, and test-time augmentation with
`flip` swaps it back in each mirrored view's prediction before the views are merged (a checkpoint saved before the
mapping was stored takes it from the val JSON, with a warning). The training banner lists each head's swapped pairs. A
regression head needs no mapping: its value must not change under a flip (see below).

**Regression attributes.** `"type": "regression"` in the data makes a head regress one real number
per instance instead of choosing a category; categorical and regression heads mix freely in one
dataset. In the run config, `log` (regress `log1p` of the value, for heavy-tailed positives such as
costs) and `range` (valid values; predictions are clipped to it) are optional, and `applies_to` works as above:

```jsonc
// data
"attributes": [
  {"name": "grade", "categories": [...]},
  {"name": "price", "type": "regression", "unit": "EUR"}
]
// per annotation:
"attributes": {"grade": 3, "price": 812.5}
```

```yaml
# run config
attributes:
  grade:
  price: {log: true, range: [0, 20000], applies_to: [sofa]}
```

A missing, non-numeric or out-of-range value is ignored. The head is trained on the value
standardized by the train split's mean and std (stored in the checkpoint), validated as the
mean absolute error in real units (`val/aux_mae/<head>`; `metrics.png` gives each regression
head its own MAE panel, `aux_per_class.png` its MAE per primary class), and predicted in real units
(`aux_values`). The value must not change under the augmentations (scale, crop, flip,
rotation): a cost or a score is fine, a pixel area or an angle is not. Tracking votes on the
categorical heads only, and re-identification cannot gate on a regression head.

A value that depends on how much of the instance is in view can say so with `crop`: the
training augmentation measures, per instance, the fraction it keeps in view (mask mass, or box
area, inside the crop window, after copy-paste occlusion), and the value follows it.
`"crop": "scale"` multiplies the value by that fraction — for an extensive quantity such as the
object's surface in cm² (half the object cropped away ⇒ half the surface). `"crop": "ignore"` drops the
value when less than `min_visible` (default `0.9`) is kept — for a property of the whole object
that a cut view no longer shows. Validation and inference see whole images, so the head predicts
the full value there:

```yaml
# run config
attributes:
  surface_cm2: {log: true, crop: scale}
  length_cm: {crop: ignore, min_visible: 0.95}
```

The fraction is the share of the instance's mask mass kept inside the window (box overlap for the detect family),
exposed as `TrainAugment.last_info["visible"]`. `crop` applies to regression heads only, is training-only (not stored in
the checkpoint), and is not applied to a custom `transform=`.

**Sidecar format (optional).** You can keep the COCO JSON as plain, standard COCO and put the
attributes beside it instead of inside it: an `attributes.yaml` schema (same keys, regression heads included) in the dataset root plus
`attributes/<split>.json` values keyed by annotation id (`{ann_id: {head: value}}`). If present
(and the JSON has no embedded `attributes`), it is merged in memory at load — so a plain COCO
dataset always works and the sidecar is picked up automatically when you add it. Embedded
`attributes` in the JSON take precedence.

A tiny, self-contained example (two heads, including a non-contiguous id set) lives in
[sample_data/](sample_data/), with its run config [sample_data/run.yaml](sample_data/run.yaml).

Datasets built before the run config may still carry `applies_to` / `flip_aware` / `log` / `range` / `crop` in their
attribute definitions: they are still read, and the run config's `attributes` win over them.

---

## Run config (one YAML per run)

**The dataset holds the data; one YAML holds the run.** The COCO JSONs carry the images, the instances and each
attribute's name, type and categories — nothing about training. Everything else is written out in the YAML that
`--data` points at, the **run config**: the data paths, which attributes get a head and how each learns, the
augmentation, the `train()` settings and the inference settings saved with the model. One file says how a model was
made, so a run can be read, repeated and compared from it. Start from the commented template
[`sample_data/run.yaml`](sample_data/run.yaml):

```yaml
# --- data
path: .
train_images: images/train
train_json: annotations/instances_train.json
val_images: images/val
val_json: annotations/instances_val.json

# --- the heads, and how each learns: flip_aware, crop, min_visible, log, range, applies_to
#     (an attribute of the data left out gets no head; no section: every attribute as the data declares it)
attributes:
  scale:
  viewpoint:
  # side: {flip_aware: {left: right}}
  # price: {log: true, range: [0, 20000], applies_to: [sofa]}
  # surface_cm2: {log: true, crop: scale}

# --- augmentation: AugConfig fields over its defaults (`train_aug`, the older name, is the same section)
aug:
  flip_prob: 0.5
  min_scale: 0.4
  max_scale: 1.6

# --- train() keywords: anything train() takes but the data paths (the size: --size, the device: --device)
train:
  project: ../runs/train
  name: sample
  epochs: 12
  imgsz: 644
  batch: 4
  # init_weights: ../runs/train/eomt-l/weights/best.pt   # warm start from a checkpoint

# --- saved in every checkpoint: the defaults of predict() / track() with that model
inference:
  conf_thres: 0.3
  max_det: 100
```

```bash
python scripts/train.py --data sample_data/run.yaml --size s --device 0,1     # the whole run from the file
python scripts/train.py --data sample_data/run.yaml --size s --epochs 1       # a flag beats the file
python scripts/predict.py runs/train/sample/weights/best.pt images/           # with the conf_thres it was saved with
```

- **Precedence:** explicit arguments (command line, `train()` / `EoMT.train()` keywords) beat the YAML, which beats the
  library's defaults. Relative paths (`path`, `train.init_weights`, `train.project`) are relative to the YAML.
- **Recorded:** `args.yaml` in the run folder and every checkpoint (`run_config`) hold the resolved run — every
  hyper-parameter, the augmentation and how each head learned (with its regression statistics); the checkpoint also
  keeps `inference`, which `predict()` / `track()` (and `scripts/predict.py`) use for any setting the call leaves out.
- **Errors early:** an attribute the YAML configures but the data does not define, an unknown behavior or inference key,
  or a data path / size / device inside `train` stops the run before it starts.

## Cross-photo re-identification

Given several photos of one subject, `infer_match()` decides which detections are the same physical instance, with no
second model and nothing retrained: the per-query embedding that feeds the class head is the fingerprint. Photos are
matched pairwise with the Hungarian matcher and linked into identities with union-find (at most one instance per photo,
plus a merge guard against transitive chaining). `group_by` limits which pairs may compete (same class by default; add
attribute heads to tighten it) and `sim_thres` (cosine, default `0.7`) is the floor for "same instance". Each run
writes its candidate pairs and similarity diagnostics to JSON, so the threshold can be retuned without re-running
inference.

```python
result = EoMT("runs/train/eomt-l/weights/best.pt").infer_match("photos/subject_42/", group_by=("class",))
```

```bash
python scripts/match.py weights/best.pt photos/ --each-subdir   # one folder per subject
```

Every option is documented in [`eomt/engine/match.py`](eomt/engine/match.py) and `python scripts/match.py --help`.

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
EoMT("l").train(data="coco", batch=4)   # 12 epochs, validated every 2: the official recipe

# ...or any COCO-format dataset, the run written out in its run config (see "Run config"):
EoMT("s").train(data="sample_data/run.yaml")
EoMT("s").train(data="sample_data/run.yaml", epochs=1)   # a keyword beats the file

# Validate and predict from a trained run:
EoMT("runs/train/eomt-l").val(data="coco")
EoMT("runs/train/eomt-l").predict("images/", plot=True)   # writes annotated images
```

**Several GPUs.** `device="0,1"` trains on both GPUs, one process each (DistributedDataParallel); `"auto"` is GPU 0.
`batch` is per GPU and `nominal_batch` (16) stays the global effective batch, so the optimizer steps, LR schedule and
masked-attention annealing are exactly those of one GPU. Validation splits the val images between the GPUs; the main process scores them and alone logs and writes checkpoints. The processes are spawned, so every `train()`
argument must be picklable.

```python
EoMT("l", device="0,1").train(data="coco", batch=2)   # 2 per GPU x accum 4 x 2 GPUs = effective 16
```

**Test-time augmentation and tiled inference** (instance family). `predict()` and `val()` take `augment=True` (the
image at `imgsz` and zoomed to 1.5x `imgsz`) and `tiles=True` (overlapping tiles of `imgsz` px of the original image, i.e. native resolution,
25 % overlap, plus the whole image: small objects in large images). The views' instances are merged: same class and
mask IoU >= 0.5 measured where both views can see, so an object cut by a tile border is put back together; the mask
logits are averaged and the score is the mean over the views that could see the object (one that missed it votes 0).
Each takes a dict of overrides; see
[`eomt/tta.py`](eomt/tta.py). `augment={"flip": True}` also merges mirrored views. Attributes marked `flip_aware` are swapped
back in a mirrored view; left / right *primary classes* are not (a mirrored view predicts them swapped), so keep `flip` off
for those, and for attributes that encode a side without being flip-aware.

```python
m = EoMT("runs/train/eomt-l")
m.predict("images/", augment=True)                                  # + the image at 1.5x imgsz
m.val(data="data.yaml", tiles=True)                                 # native tiles + the whole image
m.predict("images/", augment={"flip": True}, tiles=1024)            # 1x / 1.5x and mirrored; 1024 px tiles
```

`scripts/val.py` and `scripts/predict.py` take `--augment` and `--tiles [SIZE]` (`scripts/val.py` also `--flip`; its arguments are
in [Command-line scripts](#command-line-scripts-arguments-and-things-to-take-into-account)). Each view is one more forward pass.

For the full training recipe, every `train()` knob, and int8 compression, see the
**[annotated explainer →](https://imagra93.github.io/attr-eomt)** — it's the deep dive.

### Optimisation recipe

`train()` follows the reference EoMT recipe (checked numerically against the official code: every parameter group's
LR, every step of the LR schedule, and the masked-attention annealing agree exactly):

| | default |
|---|---|
| optimizer | AdamW, `lr0=1e-4`, `weight_decay=0.05` (not applied to norms, biases and embeddings) |
| head LR | `lr0` |
| encoder LR | layer-wise decay `llrd=0.8`, `backbone_lr_mult=1.0`: the **top ViT block is at the full `lr0`** and each block below is 0.8× the one above; embeddings follow the first block, the final norm is at `lr0` |
| schedule | polynomial decay (`poly_power=0.9`) to 0 after a two-stage warmup in optimizer steps `warmup_steps=(500, 1000)`: the head warms up while the ViT LR is held at 0, then the ViT ramps up |
| masked attention | annealed block by block (`mask_anneal_start/end`, fractions of training: 1/6–2/6, 2/6–3/6, 3/6–4/6, 4/6–5/6, polynomial); the last sixth of training is mask-free |
| length | `epochs=12`, validated every `val_interval=2` epochs and after the last (official COCO instance recipe, about 89k optimizer steps at batch 16) |

In EoMT the last four transformer blocks are the ones that process the queries and predict the masks, so they must learn at
close to the head's speed: scaling the whole encoder by 0.1 (as detector-style recipes do for a backbone) starves them. The
LR decay and the annealing span `epochs`.

**Validation follows the training regime**, as in the official EoMT: masked attention at the annealing probabilities of
the epoch's end. A model that still masks is far worse without its masks, so a mask-free curve before the last sixth follows the
annealing and jumps as each block finishes it, not the model. `best.pt` is deployed mask-free, so it is only chosen once every block's annealing is
over (a run stopped earlier, e.g. with `stop_after_epochs`, has `last.pt` alone), and every checkpoint is saved mask-free.
Training prints the windows in epochs when it starts.

### Command-line scripts: arguments and things to take into account

**`scripts/train.py`.** Every flag is a keyword of `train()`; a flag left out keeps `train()`'s default.

| flag | `train()` keyword | what it does |
|---|---|---|
| `--data` | `data` | dataset YAML, or `coco` (auto-downloads) |
| `--size` / `--task` | | model size `s` `b` `l`; head family `instance` (masks) or `detect` (boxes) |
| `--epochs` | `epochs` | total epochs: the LR schedule and the masked-attention annealing span them |
| `--batch` / `--nominal-batch` | `batch` / `nominal_batch` | micro-batch **per GPU** / global effective batch (16); the difference is gradient accumulation |
| `--device` | `device` | `auto`, `cpu`, `cuda:N`, or `0,1` for several GPUs (one process each) |
| `--weights` / `--resume` | `init_weights` / `resume` | warm-start a fresh run from a checkpoint / continue a run. **One of the two, not both** |
| `--lr`, `--warmup HEAD,VIT` | `lr0`, `warmup_steps` | peak head LR (the encoder follows with layer-wise decay) and the two-stage warmup in optimizer steps |
| `--ema-decay`, `--ema-tau` | `ema_decay`, `ema_tau` | EMA of the weights (what is validated and exported as `best.pt`) |
| `--mask-free` | `mask_anneal_start/end` | no masked attention from step 0 (annealing windows 0 -> 0) |
| `--aux-dropout` | `aux_head_dropout` | dropout in the attribute heads |
| `--val-interval`, `--workers`, `--seed` | same | validate every N epochs (and after the last); data-loader workers per GPU; seed |
| `--aug KEY=VALUE`, `--aug-preset` | `aug` | augmentation overrides (see [Augmentation](#augmentation)) |
| `--fpn-scales`, `--box-head`, `--no-deep-supervision`, `--upscale-blocks`, `--iou-aware-cls`, `--quality-weight`, `--val-imgsz` | same | [architecture options](#architecture-options) |

Things to take into account when training:

- **Resume or warm start.** `--resume` continues a run exactly (epoch, optimizer, schedule, EMA, best metric) and re-reads the
  architecture from the checkpoint: keep the other flags as they were, and the seed is ignored so the shuffle is not replayed.
  `--weights` loads the weights only and starts a new run at epoch 0. Start from a `best.pt` (the EMA, saved mask-free); a
  `last.pt` holds the live weights.
- **A short fine-tune needs its own recipe.** Lower `--lr`, shorten `--warmup`, and lower `--ema-decay` (at 0.9999 the EMA
  averages over ~10k steps and barely leaves its starting point in a few epochs). If the model is deployed mask-free, pass
  `--mask-free`: with the default annealing a warm-started model is trained *masked* for the first sixth of the run, a regime it
  no longer runs in.
- **`best.pt` is only chosen mask-free.** With the default annealing that means from 5/6 of the schedule on; with `--mask-free`
  every validation is eligible.
- **Validated epochs only.** With `--val-interval N` the other epochs have empty validation cells in `metrics.csv`; the curves in
  `metrics.png` join the validated points.
- **Attribute heads follow the run config.** Which attributes train and their `applies_to`, `flip_aware`, `crop`, `log` and
  `range` come from the YAML's `attributes` section (see *Run config*), else from the dataset's definitions; the training
  banner lists them. The flip mapping, scope and regression statistics are stored in the checkpoint, and the whole run
  config in `args.yaml` and the checkpoint's `run_config`.

**`scripts/val.py`.**

| flag | what it does |
|---|---|
| `weights` | checkpoint `.pt` or a run / weights folder (`best.pt` preferred) |
| `--data`, `--batch`, `--workers`, `--device` | dataset YAML (`coco`), batch, data-loader workers, device |
| `--imgsz` | evaluation input size (default: the trained one; the positional embeddings are interpolated) |
| `--augment` | add the image at 1.5x `imgsz` |
| `--tiles [SIZE]` | overlapping tiles (default `imgsz` px, native resolution) plus the whole image |
| `--flip` | also run every view mirrored (alone: `--augment` with mirrored views) |
| `--no-attributes` | mAP only |
| `--ema`, `--mask annealed\|free` | validate the EMA weights of a `last.pt` (see below) |

Things to take into account when validating:

- **What is reported.** `segm/*` and `bbox/*` mAP and, for a model with secondary heads, `aux_acc/<head>` (accuracy) and
  `aux_mae/<head>` (regression heads, mean absolute error in real units). They are scored on the same predictions as the mAP, so
  they also work with `--augment`, `--tiles` and `--flip`.
- **The attribute population is the deployed one.** Each GT instance is matched to the kept detection of its class with the
  highest mask IoU (one to one, IoU >= 0.5). The trainer's `val/aux_*` columns score matched *queries* instead, so the two differ
  slightly: compare TTA against plain within the script's numbers, not against `metrics.csv`. A regression head's MAE is
  dominated by its large values, so also look at a relative error before concluding.
- **Regime.** A model that is still annealing is only meaningful with its masks: `--ema --mask annealed` reproduces the trainer's
  validation of a `last.pt` (EMA weights, the masked-attention probabilities of that epoch, read from the run's `args.yaml`),
  `--ema --mask free` is the deployed regime, and `best.pt` is already the EMA, mask-free.
- **TTA is not free and not uniform.** It costs one forward pass per view and mostly helps small objects and strict-IoU mAP; the
  classification heads often do not move. A regression head's prediction is *averaged over views at different zoom*, so a value
  that depends on the object's apparent size can drift under TTA: check it against the plain run before relying on it.
- **`--flip` and the heads.** Heads the checkpoint stores as flip-aware are swapped back in mirrored views; an older checkpoint
  takes the mapping from the dataset's JSON, with a warning. Left / right *primary classes* are not handled: keep `--flip` off for
  them. The val categories of every head must match the model's, or the run stops instead of scoring against the wrong ids.
- **One process.** `scripts/val.py` runs on one device; the trainer's own validation uses all the GPUs.

### Architecture options

All are keyword arguments of `train()` (and flags of `scripts/train.py`); checkpoints record them, so `EoMT(path)` rebuilds
the model with the same modules, and checkpoints without them load as before.

| option | what it does |
|---|---|
| `fpn_scales=None` | drop the multi-scale FPN (default `(2, 1, 0.5)`) |
| `box_head=True` | auxiliary box head: L1 + GIoU loss on the matched queries, and the same terms in the matching cost (a localisation signal that does not depend on the mask grid). Predicted boxes are returned as `aux_boxes` / `head_boxes` |
| `deep_supervision=False` | stop supervising a query block's prediction once that block's masked attention is annealed away (while it masks, the prediction builds its attention mask and stays supervised). Blocks drop out at 2/6, 3/6, 4/6, 5/6 of training |
| `num_upscale_blocks=3` | finer mask logits (368² at 644 px) for small objects; cannot warm-start a 2-block checkpoint |
| `iou_aware_cls=True` | train a matched query's class probability towards `(1 + mask IoU) / 2` instead of 1 (IoU-aware targets, as in VarifocalNet / Stable-DINO), so the score ranks loose masks below tight ones; above 0.5, so matched queries keep their class for the attribute gate. Hard targets reduce exactly to the weighted CE |
| `quality_weight=2` | mask-quality head: a small MLP on the query embedding predicts each matched query's mask IoU (BCE against the measured IoU, final prediction only; Mask Scoring R-CNN), and postprocess scores class probability x predicted IoU instead of the mean mask probability |
| `val_imgsz=N` | validate (and pick `best.pt`) at another input size than `imgsz`. Scale jitter shows the objects larger, on average, than a val image resized to `imgsz`: validate at `imgsz` × the mean training scale |
| `stop_after_epochs=N` | stop after N epochs while the LR schedule and mask annealing still span `epochs` (truncated A/B runs) |

Always on (training-side; inference of existing checkpoints is unchanged):

- **Masked attention keeps small objects in view.** A query attends to every patch its predicted mask touches (max-pool of
  the mask logits onto the patch grid), and a query whose mask touches no patch attends to all of them, as in Mask2Former. The
  reference EoMT samples the mask bilinearly at patch centres and leaves an empty-mask query blind.
- **Mask terms that see small objects.** Matching scores every query on every cell of the mask-logit grid instead of on
  12,544 random points, and dice is computed on every GT pixel, so every instance is supervised however small. BCE keeps the
  reference's 12,544 points concentrated on the uncertain boundaries.
- **Attribute IoU gate at full resolution.** The gate that decides which matched queries train the attribute heads compares the
  upsampled prediction with the GT, so objects of a few pixels are not erased by a "nearest" downscale.

Inference: `predict_image(..., amp=True)` (and `predict(..., amp=True)`) runs the network under bf16 autocast. Off by default.

The accuracy effect of these options depends on the data. Measure it with truncated runs (`stop_after_epochs`), which
follow the schedule of a full run, against a reference run at equal epochs.

---

## Augmentation

Training augmentation is one config, [`AugConfig`](eomt/data/transforms.py), run by one pipeline
(`TrainAugment`) for both model families. The defaults are a **strong general recipe**: the original
EoMT/Mask2Former one (horizontal flip, Large-Scale Jitter 0.1–2.0, random crop, colour jitter) **plus** small
rotation / shear / perspective, gamma, grayscale, Gaussian blur, sensor noise, JPEG recompression, glare and
"safe" random erasing (a rectangle that never overlaps an instance). Masks are resized by area averaging (soft,
mass-preserving), so thin objects are not shredded into dots by nearest-neighbour resampling.

| group | knobs (default probability) |
|---|---|
| geometry | `flip_prob` 0.5 · `vflip_prob` 0 · `rot90_prob` 0 · LSJ `min_scale`/`max_scale` 0.1–2.0 · `rotate_prob` 0.3 (±10°) · `shear_prob` 0.2 (±5°) · `perspective_prob` 0.15 |
| optics / sensor | colour jitter 1.0 · `gamma_prob` 0.3 · `grayscale_prob` 0.05 · `blur_prob` 0.2 · `noise_prob` 0.2 · `jpeg_prob` 0.3 · `glare_prob` 0.15 · `erasing_prob` 0.2 |
| multi-image (instance family) | `mosaic_prob` 0 · `mixup_prob` 0 · `copy_paste_prob` 0 |
| crop | `instance_crop_prob` 0 (instance-aware crop) · `instance_crop_empty_prob` 0 · `zoom_prob` 0 (close-ups, `zoom_scale` 4–5x) |
| masks | `mask_resize="area"` (or `"nearest"`) |

Override per run, per dataset, or from the CLI — precedence is explicit keywords (`flip_prob`, `min_scale`,
`max_scale`) **>** `aug=` **>** the YAML's `aug` (or `train_aug`) block **>** defaults:

```python
model.train(data="coco", aug={"rotate_prob": 0.5, "blur_prob": 0})   # in code
model.train(data="coco", aug={"preset": "legacy"})                    # the original recipe (hard masks, no extras)
model.train(data="coco", train_transform=my_callable)                 # bring your own (image, masks) -> (image, masks)
```

```yaml
# data.yaml — settings for THIS dataset
train_aug:
  min_scale: 0.6                  # thin / tiny objects: keep the effective scale r = imgsz/long_side * s in ~[0.4, 1.0]
  max_scale: 1.6
  instance_crop_prob: 0.8         # optional instance-aware crop: window placed around a (rarity-weighted) instance
  instance_crop_rarity_attr: material
```

```bash
python scripts/train.py --data my/data.yaml --aug min_scale=0.6 --aug instance_crop_prob=0.8 --aug-preset default
```

Things worth knowing:

- **Orientation.** Flips, 90° turns and rotations change the pixels but not the label. If an attribute encodes
  orientation (e.g. a `viewpoint` head) set `flip_prob=0` and `rotate_prob=0`, or, for a left / right attribute,
  mark it `flip_aware` (see *Flip-aware attributes*) and keep the horizontal flip; `vflip_prob` / `rot90_prob` stay
  off unless your scenes have no canonical "up".
- **Thin or tiny objects.** Large-Scale Jitter at 0.1 shrinks a 5 px wide line to under a pixel; keep the effective
  scale within ~[0.4, 1.0] (`min_scale = 0.4·long_side/imgsz`, `max_scale = 1.0·long_side/imgsz`). The instance-aware
  crop pays off when the crop is much smaller than the image (high zoom, small `imgsz`), and
  `instance_crop_rarity_attr` makes rare attribute values the crop's focus more often.
- **Close-ups.** `zoom_prob` makes an occasional sample a close-up: scaled by `s ~ U(zoom_scale)` (default 4–5x
  the whole-image view, past the photo's own resolution) instead of the LSJ range, with the window on one instance
  (whole when it fits, else a window inside it). `zoom_attr` restricts it to instances with a value of that
  attribute and to the samples that have one, e.g. `{"zoom_prob": 0.1, "zoom_attr": "surface_cm2"}` so a crop-scaled
  head also learns that an object filling the frame keeps its real surface, and that a cut object keeps the share in view.
  It resizes the whole photo first, so keep it occasional.
- **Multi-image ops** (`mosaic`, `mixup`, `copy_paste`) add instances from other samples and are *off*: they splice or
  blend long thin structures, so try them deliberately and look at the result.
- **Negatives.** Images with no annotations are dropped from the train set by default. `train(keep_empty=True)` keeps
  them as zero-target samples (masks `(0, imgsz, imgsz)`, every query is trained as "no object"); validation already
  includes them. Instance family only.
- `args.yaml` of a run records the resolved `aug` dict; the log prints the active ops at start.
- `build_train_transform(imgsz, aug=...)` returns the same pipeline for use in your own loaders
  (`tf(image_uint8, masks) -> (image, masks)`).

---

## Roadmap / future work

- **Model export.** ONNX / TensorRT (and friends) for deployment — currently out of
  scope; the inference path is being kept export-friendly.
- **Keypoints.** A keypoint/pose head family alongside `instance` and `detect` (the code
  already carries a `family` parameter so new heads slot in without API churn).
- **Pretrained COCO checkpoints.** None are published yet. COCO-trained `s`/`b`/`l`
  weights will be released on the Hugging Face Hub (the `from_pretrained` / `hf://`
  loading plumbing is already in place and waiting for them).
- **Contrastive re-ID training.** Cross-photo re-identification already works at
  inference time (see [above](#cross-photo-re-identification)) on the embedding the
  detector computes anyway. What remains is *training* that embedding for the job: a
  contrastive objective on the matched queries would make each one a purpose-built
  re-identification vector rather than a by-product of the class head, which should
  widen the margin between matches and non-matches and make `sim_thres` transferable
  across datasets. Feeding those embeddings into the video tracker to re-associate
  objects across occlusions is the same lever.

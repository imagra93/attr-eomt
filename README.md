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
- A **missing or out-of-vocab** per-annotation value is *ignored* (`-100`), not trained as
  class `0` — so a **partially tagged** dataset is valid: each head learns only from the
  instances that actually carry its value. A JSON with **no** `attributes` ⇒ detection-only,
  exactly as before.

**Class-conditional heads.** Give an attribute definition an optional `applies_to` list of
primary-class names or ids, and that head is only trained on — and only emitted for —
instances of those classes (hard routing on the primary class). Omit it and the head applies
to every class. So different attributes can attach to different classes, each with its own
label set, in one model:

```jsonc
"attributes": [
  {"name": "posture", "categories": [...], "applies_to": ["cat", "dog"]}
]
```

At inference a scoped head reports `ids = -1` ("not applicable") for detections whose class it
does not cover. The scope is stored in the checkpoint, so it survives reload.

**Sidecar format (optional).** You can keep the COCO JSON as plain, standard COCO and put the
attributes beside it instead of inside it: an `attributes.yaml` schema in the dataset root plus
`attributes/<split>.json` values keyed by annotation id (`{ann_id: {head: value}}`). If present
(and the JSON has no embedded `attributes`), it is merged in memory at load — so a plain COCO
dataset always works and the sidecar is picked up automatically when you add it. Embedded
`attributes` in the JSON take precedence.

A tiny, self-contained example (two heads, including a non-contiguous id set) lives in
[sample_data/](sample_data/).

---

## Cross-photo re-identification

Several photos of the same scene from different viewpoints, and some instance appears
in three of them — is that one instance seen three times, or three separate ones?
`infer_match()` answers that at **inference time, with no second model and nothing
retrained**: every detected instance already carries a fingerprint, the per-query
embedding that feeds the class head, and two detections are the same physical instance
when those embeddings agree.

```python
from eomt import EoMT

model = EoMT("runs/train/eomt-l/weights/best.pt")
result = model.infer_match("photos/subject_42/", group_by=("class",))

for rec in result["identities"]:
    print(rec["identity_id"], rec["class_name"], rec["attributes"], rec["num_photos"])
```

One call handles one folder = one subject. Each detection is stamped with an
`identity_ids` entry, `identities` summarizes each identity (dominant class, smoothed
attributes, how many photos it appears in), and `matches` records every candidate pair
— accepted or not, with the reason — so a threshold can be retuned from the JSON
without re-running inference. With `plot` you get a grid image: every photo, instances
colored by identity, plain lines linking the matched pairs (thickness = similarity, on
an absolute scale that does not vary with panel size), and a legend.

```bash
python scripts/match.py weights/best.pt photos/subject_42/
python scripts/match.py weights/best.pt photos/ --each-subdir   # a folder per subject
```

### How it works

It is the DeepSORT recipe — detect, describe, associate — applied *between photos*
instead of between video frames, which is also what [`track()`](eomt/engine/track.py)
does for video. Three EoMT properties make the fingerprint free:

1. the model is **NMS-free**, so two overlapping instances stay two distinct queries;
2. each query owns one instance, so its embedding describes that instance alone;
3. the embedding is already computed on the way to the class head.

Matching runs the **Hungarian** matcher per pair of photos, then links the accepted
pairs into identities with union-find. Two constraints keep that honest: one physical
instance appears **at most once per photo** (union-find would otherwise chain two
instances of the same photo together through a third), and a **merge guard** rejects a
merge whose cross-cut similarity falls below the threshold — without it, A↔B and B↔C
chain into one identity even when A and C are nothing alike.

### Gating and thresholds

`group_by` decides which pairs may match at all. The default `("class",)` means only
same-class instances compete. Adding attribute heads tightens it, which matters
whenever the primary class is coarser than the distinction you care about: if one class
covers instances that sit at different places on the subject, an attribute head that
separates them — `("class", "position")`, say — stops an instance in one location from
ever matching one in another, however alike they look. Any head in the checkpoint can
be named; unknown names raise before the first forward pass. `group_by=None` disables
gating entirely and needs a much higher `sim_thres`, since far more pairs then compete
with no structural prior ruling any of them out.

`sim_thres` (default `0.7`) is the cosine floor for "same instance". Hungarian always
returns a full assignment, so the threshold is what turns *best available partner* into
*no partner — this is a new instance*. Measured on one checkpoint across four real
photo sets, with the gate on:

| | non-candidate pairs | true-match pairs |
|---|---|---|
| mean | 0.11 – 0.16 | 0.85 – 0.98 (the true-match mode) |
| p99 / max | 0.40 – 0.67 / 0.49 – 0.79 | — |

The default was raised from `0.6` to `0.7` after a 30-case run showed non-candidate
pairs reaching `0.73` and the two distributions overlapping in over half the cases —
`0.6` was admitting matches with no margin at all. Every run
writes its own `diagnostics` (within-identity vs across-identity percentiles) into
`<subject>_identities.json`; if those two distributions overlap, no threshold will
save the run.

### Limitations

- **A gate group with one instance per photo gets no benefit from the embedding.**
  Hungarian has no choice to make, and only `sim_thres` can veto the pairing. The
  fingerprint earns its keep where several instances of one group compete in the same
  photo.
- **Small-instance recall caps what can be matched.** An instance that is never
  detected in a view cannot be linked to it.
- **One folder must be one subject.** A folder holding photos of more than one subject
  will happily link generic-looking instances across them; that is a data problem, not
  a matching one.
- **Letterboxing shifts scale with orientation**, so a landscape and a portrait shot of
  the same subject are not on quite equal footing.

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

**Several GPUs.** `device="0,1"` trains on both GPUs, one process each (DistributedDataParallel); `"auto"` is GPU 0.
`batch` is per GPU and `nominal_batch` (16) stays the global effective batch, so the optimizer steps, LR schedule and
masked-attention annealing are exactly those of one GPU, with an epoch 1.9× faster on two RTX 5090s (1.8× end to end with
validation and checkpoints). Validation splits the val images between the
GPUs; the main process scores them and alone logs and writes checkpoints. The processes are spawned, so every `train()`
argument must be picklable.

```python
EoMT("l", device="0,1").train(data="coco", epochs=50, batch=2)   # 2 per GPU x accum 4 x 2 GPUs = effective 16
```

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

In EoMT the last four transformer blocks are the ones that process the queries and predict the masks, so they must learn at
close to the head's speed: scaling the whole encoder by 0.1 (as detector-style recipes do for a backbone) starves them. The
LR decay and the annealing span `epochs`, and validation runs mask-free, so metrics logged early in a long run understate
the model. The official COCO recipe trains for 12 epochs (about 89k optimizer steps at batch 16).

### Architecture options

All are keyword arguments of `train()` (and flags of `scripts/train.py`); checkpoints record them, so `EoMT(path)` rebuilds
the model with the same modules, and checkpoints without them load as before.

| option | what it does | cost, ViT-L at 644 px, batch 2, bf16 (measured, relative to the default) |
|---|---|---|
| `fpn_scales=None` | drop the multi-scale FPN (default `(2, 1, 0.5)`) | +12 % throughput, −0.5 GB |
| `box_head=True` | auxiliary box head: L1 + GIoU loss on the matched queries, and the same terms in the matching cost (a localisation signal that does not depend on the mask grid). Predicted boxes are returned as `aux_boxes` / `head_boxes` | −7 % |
| `deep_supervision=False` | stop supervising a query block's prediction once that block's masked attention is annealed away (while it masks, the prediction builds its attention mask and stays supervised: unsupervised, it made noise masks). Blocks drop out at 2/6, 3/6, 4/6, 5/6 of training | none while blocks mask; +26 % per step once all are annealed (last sixth) |
| `num_upscale_blocks=3` | 368² mask logits at 644 px (small masks); the block predictions stay at 184² (they only build 46² attention masks), only the output is 368²; cannot warm-start a 2-block checkpoint | −17 % (10.1 vs 12.1 img/s), 22.2 GB |
| `iou_aware_cls=True` | train a matched query's class probability towards `(1 + mask IoU) / 2` instead of 1 (IoU-aware targets, as in VarifocalNet / Stable-DINO), so the score ranks loose masks below tight ones; above 0.5, so matched queries keep their class for the attribute gate. Hard targets reduce exactly to the weighted CE. Windscreen damage, 12 epochs at 644 px, against the same recipe: segm AP 0.0296 -> 0.0758, AP small 0.0033 -> 0.0106, AP large 0.128 -> 0.210, bbox AP 0.177 -> 0.217, attribute accuracy unchanged | one upsample of the matched masks |
| `quality_weight=2` | mask-quality head: a small MLP on the query embedding predicts each matched query's mask IoU (BCE against the measured IoU, final prediction only; Mask Scoring R-CNN), and postprocess scores class probability x predicted IoU instead of the mean mask probability. Windscreen damage, with `iou_aware_cls`, two seeds against the same recipe without the head: segm AP +7 / +13 %, AP small +19 % / 3.6x, AP medium +21 / +22 %, AR100 +2 / +5 %, bbox AP and attribute accuracy unchanged | one MLP on the queries |
| `val_imgsz=N` | validate (and pick `best.pt`) at another input size than `imgsz`. Scale jitter shows the objects larger, on average, than a val image resized to `imgsz`: validate at `imgsz` × the mean training scale. Windscreens (training scale 0.4-1.0 of the photo, val 0.5), same checkpoint: segm AP 0.0113 at 644, 0.0173 at 896 | val time grows with the pixels |
| `stop_after_epochs=N` | stop after N epochs while the LR schedule and mask annealing still span `epochs` (truncated A/B runs) | |

Always on (training-side; inference of existing checkpoints is unchanged):

- **Masked attention keeps small objects in view.** A query may attend to every patch its predicted mask touches (max-pool of
  the mask logits onto the patch grid), and a query whose mask touches no patch attends to all of them, as in Mask2Former. The
  reference EoMT downsamples bilinearly (only the centre of each patch counts) and leaves an empty-mask query blind, which in
  practice blinded the queries of most objects smaller than a patch during the masked phase. The mask is a boolean
  `[B, 1, N, N]` and is not built in a block whose annealing probability has reached 0.
- **Mask terms that see small objects.** Matching scores every query on every cell of the mask-logit grid against the GT
  area-averaged onto it, instead of on 12,544 random points (one per ~33 px² at 644 px, so an object of a few dozen pixels often
  got no point and its assigned query changed from step to step). The mask loss keeps the reference's BCE on 12,544 points
  concentrated where the prediction is uncertain (the boundaries), and computes dice on every GT pixel against the upsampled
  logits, so every instance is supervised however small. A fully dense loss gave coarser masks (AP75 down) on thin objects;
  dropping the points altogether lost the boundaries. The `train_num_points` / `oversample_ratio` / `importance_sample_ratio`
  settings are gone (fixed at the reference values; checkpoints that carry them still load).
- **Attribute IoU gate at full resolution.** The gate that decides which matched queries train the attribute heads compares the
  upsampled prediction with the GT, instead of shrinking the GT to the logit grid with "nearest" (which erased many objects of a
  few pixels, so they never trained the attribute heads).

Inference: `predict_image(..., amp=True)` (and `predict(..., amp=True)`) runs the network under bf16 autocast, ~2.5x faster at 644 px
on an RTX 5090 with the same accuracy on a trained ViT-L model (AP within 0.002 in every size bucket). Off by default.

**What is not established:** the accuracy effect of `box_head`, `deep_supervision=False`,
`iou_aware_cls` and `quality_weight` (beyond the windscreen data), dropping the FPN and the always-on changes above depends on the data. Measure it with truncated runs (`stop_after_epochs`), which follow the schedule of a full run, against a reference run at equal
epochs.

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
| crop | `instance_crop_prob` 0 (instance-aware crop) · `instance_crop_empty_prob` 0 |
| masks | `mask_resize="area"` (or `"nearest"`) |

Override per run, per dataset, or from the CLI — precedence is explicit keywords (`flip_prob`, `min_scale`,
`max_scale`) **>** `aug=` **>** the dataset YAML's `train_aug` block **>** defaults:

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
  orientation (e.g. a `viewpoint` head) set `flip_prob=0` and `rotate_prob=0`; `vflip_prob` / `rot90_prob` stay off
  unless your scenes have no canonical "up".
- **Thin or tiny objects.** Large-Scale Jitter at 0.1 shrinks a 5 px crack to under a pixel; keep the effective
  scale within ~[0.4, 1.0] (`min_scale = 0.4·long_side/imgsz`, `max_scale = 1.0·long_side/imgsz`). The instance-aware
  crop pays off when the crop is much smaller than the image (high zoom, small `imgsz`), and
  `instance_crop_rarity_attr` makes rare attribute values the crop's focus more often.
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

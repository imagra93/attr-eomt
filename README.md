# libre-eomt

Standalone **EoMT** (Encoder-only Mask Transformer) for **instance segmentation**,
extracted into a small installable package. It builds the architecture in three
sizes, initializes the ViT encoder from **DINOv2-with-registers**, and provides
training, per-epoch COCO-mAP validation, and inference/rendering.

EoMT is a DINOv2 ViT whose last few transformer blocks are augmented with
learnable queries producing mask-classification output (Mask2Former-style).
This package wraps the Apache-2.0 HuggingFace `EomtForUniversalSegmentation`;
weights trained from a DINOv2 initialization are yours to release.

On top of the base model it adds **optional secondary per-instance
classification heads** ("attributes"): extra per-detection labels that train and
infer alongside segmentation without touching the primary class space — see
[Secondary per-instance classification](#secondary-per-instance-classification).

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
# --batch 4 with the default --nominal-batch 16 accumulates to an effective batch of 16
eomt train --data configs/coco.yaml --size s --epochs 50 --batch 4 --device cuda

# Validate a checkpoint (COCO segm + bbox mAP)
eomt val --weights runs/train/eomt/weights/best.pt --data configs/coco.yaml

# Sweep inference knobs (conf/mask threshold, max_det, min area) — no retraining
eomt sweep --weights runs/train/eomt/weights/best.pt --data configs/coco.yaml

# Render predictions on an image or a folder
eomt predict --weights best.pt --source path/to/imgs --out runs/predict

# Just fetch COCO 2017
eomt download --root datasets/coco
```

You can point training/validation at **any COCO-format dataset** by passing
`--train-images/--train-json/--val-images/--val-json` directly, or by editing
`configs/coco.yaml`.

Useful training knobs (full list via `eomt train --help`):

| flag | default | effect |
|------|---------|--------|
| `--nominal-batch` / `--accum` | `16` / `0` | gradient accumulation to an **effective batch** of `--nominal-batch` (EoMT recipe is 16); `--accum N` sets the step count explicitly. See [Training recipe](#training-recipe) |
| `--ema / --no-ema` | on | validate and export `best.pt` from an **EMA** (moving average) of the weights |
| `--ema-decay` / `--ema-tau` | `0.9999` / `2000` | EMA decay and its warmup ramp time-constant (in optimizer steps) |
| `--llrd` | `0.85` | **layer-wise LR decay** on the DINOv2 backbone (`1.0` = flat `--backbone-lr-mult`) |
| `--min-scale` / `--max-scale` | `0.1` / `2.0` | **Large-Scale Jitter** range (legacy stretch-style crop: `0.5` / `1.0`) |
| `--letterbox / --no-letterbox` | on | aspect-preserving **letterbox** eval (vs the legacy square stretch); recorded in the checkpoint |
| `--tf32 / --no-tf32` | on | TF32 matmul/conv on Ampere+ GPUs (free speedup) |
| `--compile` | off | `torch.compile` the model (experimental — the HF matcher/loss use dynamic shapes) |
| `--seed` | none | seed Python/NumPy/Torch RNGs for reproducible runs |
| `--mask-anneal / --no-mask-anneal` | on | anneal masked attention off over training (EoMT recipe, see below) |
| `--mask-anneal-start` / `--mask-anneal-end` | `0.0` / `0.9` | fractions of training over which masked attention is linearly annealed `1 → 0` |
| `--aux-w` | `1.0` | weight on the summed secondary-head loss (see below) |
| `--weights` | none | **warm-start / fine-tune** from a checkpoint: load its weights, then train fresh from epoch 0 with a new optimizer/schedule/EMA — vs `--resume`, which *continues* a run |

## Secondary per-instance classification

The primary task (instance segmentation over `nc` classes) is **unchanged**.
On top of it you can attach **one or several independent secondary classifiers**
— a per-instance *attribute* predicted for each detected mask, read from that
query's embedding. Each head is a single `Linear(hidden_size, num_classes)`; you
can have as many as your data defines (they live in an `nn.ModuleDict`).

This answers the "classes **and** subclasses per instance" need by encoding a
*subclass per instance* rather than flattening to a `class × subclass` product
space (which would wreck Hungarian matching and thin out per-class statistics).
Because EoMT is NMS-free, two overlapping same-class instances stay two distinct
queries; the attribute head separates them from their embeddings.

> The attribute names used in this README (`ripeness`, `grade`, …) are
> **illustrative only** — a head is just a named set of mutually exclusive
> labels. Define whatever attributes your dataset needs.

### Example: a bowl of fruit

One model segments each fruit (primary classes `apple` / `banana` / `orange` /
`pear`) and, for **every** detection, reads off two **independent** attribute
heads — `ripeness` (`unripe` / `turning` / `ripe`) and a quality `grade`
(`A` / `B`). The renderer prints the primary class + score on the first row and
each attribute + its confidence on the row beneath it.

![A bowl of fruit; each box labelled with its fruit class plus a ripeness and a grade](docs/examples/example_1.png)

`ripeness` and `grade` are *orthogonal* — they vary independently — which is
exactly the case that's awkward to fold into the primary class space: a
`fruit × ripeness × grade` product would wreck Hungarian matching and thin out
per-class statistics. And because EoMT is **NMS-free**, the two apples (one ripe,
one unripe) and the two oranges stay four distinct queries — each gets its own
`ripeness` and `grade` read from its query embedding.

```jsonc
"attributes": [
  {"name": "ripeness", "categories": [{"id": 0, "name": "unripe"},
                                      {"id": 1, "name": "turning"},
                                      {"id": 2, "name": "ripe"}]},
  {"name": "grade",    "categories": [{"id": 0, "name": "A"},
                                      {"id": 1, "name": "B"}]}
]
// per annotation, alongside the usual category_id / segmentation:
{"category_id": 1, "segmentation": [...], "attributes": {"ripeness": 2, "grade": 0}}
```

> The image above is a **simulation**: a stock photo with hand-placed detections
> fed through the package's own renderer
> ([`eomt.visualize.draw_instances`](src/eomt/visualize.py)) to show the output
> format — not the predictions of a trained model. The same pattern fits any
> "class **plus** per-instance sub-labels" task: **retail shelves → product +
> facing**, **cells / leaves → type + health**, **apparel → garment + pattern**.

### How it works

- **Embedding source.** Each head reads the per-query embedding — the input to
  EoMT's `class_predictor`, captured with a forward hook (`[B, Q, hidden]`).
- **Matching.** Supervision reuses EoMT's *own* Hungarian matcher
  (`model.eomt.criterion.matcher`), so every attribute is trained on the **same**
  query→GT assignment the detection loss used. The attribute is read *after*
  matching.
- **Loss.** Cross-entropy per head over matched queries, summed across heads and
  scaled by `--aux-w` (default `1.0`), added to the segmentation loss. Empty-match
  batches contribute a graph-preserving zero.
- **Checkpoint selection stays `segm/mAP`.** The attribute "rides along": its
  per-head matched-query train accuracy is shown live in the progress bar, printed
  each epoch (`[epoch N] aux train acc: …`) and written to `metrics.csv`, but never
  drives `best.pt`. An epoch (or head) with no matched queries logs `nan`.
- **Inference.** Postprocessing attaches `aux = {head: {"ids", "probs"}}` for the
  same kept detections, and `predict` renders each attribute next to the class
  label (`name 0.87 · value 0.93`) using names stored in the checkpoint.

### Data format (auto-discovered from the COCO JSON)

Attributes live **inside the COCO annotations** — each annotation is already a
per-instance object, so alignment is automatic and `pycocotools` still parses it.
Two additions to a standard COCO file:

```jsonc
{
  "categories": [ {"id": 1, "name": "apple"}, {"id": 2, "name": "banana"} ],

  "attributes": [                                       // NEW, top-level: per-head vocab(s)
    {"name": "ripeness", "categories": [{"id": 0, "name": "unripe"},
                                        {"id": 1, "name": "ripe"}]},
    {"name": "grade",    "categories": [{"id": 10, "name": "A"},
                                        {"id": 20, "name": "B"}]}
  ],

  "annotations": [
    { "id": 1, "image_id": 42, "category_id": 1,
      "segmentation": [...], "bbox": [...], "area": 1234, "iscrowd": 0,
      "attributes": {"ripeness": 1, "grade": 20} }       // NEW, per instance: {head: raw_id}
  ]
}
```

- The top-level `attributes` list defines each head's vocabulary; raw ids are
  remapped to a contiguous `0..n-1` per head. `categories` may be **omitted**, in
  which case the id set is inferred from the annotations.
- Per-annotation `attributes` is a `{head: raw_id}` map; a missing value defaults
  to contiguous `0`.
- Extra keys are valid COCO and ignored by existing tooling. A JSON with **no**
  `attributes` ⇒ detection-only, behaving exactly as before.

No YAML changes are needed — heads (count, classes, names) are discovered
straight from the JSON, the same as `nc`. The dataset YAML stays as in
[configs/coco.yaml](configs/coco.yaml). Programmatically,
`CocoInstanceSeg(..., attributes=[...] | False)` restricts or disables heads.

A tiny, self-contained example (two heads, including a non-contiguous id set)
lives in [data/sample_dataset/](data/sample_dataset/) — see its
[data.yaml](data/sample_dataset/data.yaml) and annotation JSONs for the exact
layout.

## Training recipe

Defaults follow the EoMT/Mask2Former fine-tuning recipe; each piece is a flag, so
the legacy behaviour is one override away.

- **Effective batch via gradient accumulation.** The LR / weight-decay / clip are
  tuned for an effective batch of 16, so by default training accumulates
  `round(--nominal-batch / --batch)` micro-batches per optimizer step (e.g.
  `--batch 4` → 4 steps → effective 16). EoMT is a ViT (LayerNorm, no BatchNorm),
  so this is ~equivalent to a true large batch at a fraction of the memory. Set
  `--nominal-batch 0` (or `--accum 1`) for the old step-every-batch behaviour.
- **EMA weights.** A moving average of the weights is kept during training and is
  what gets validated and saved as `best.pt`; `last.pt` holds the live weights plus
  the optimizer and EMA state for exact resume. Disable with `--no-ema`.
- **Large-Scale Jitter (LSJ).** Training resizes aspect-preserving over
  `[--min-scale, --max-scale]` (default `0.1–2.0`) then crops/pads to the square
  input — a strong scale augmentation, especially for small objects.
- **Letterbox eval.** Validation/inference resize the long side and pad to a square
  (no aspect distortion); the padding is cropped back out in postprocessing. The
  mode is stored per-checkpoint so `val`/`predict` match training automatically.
- **Optimizer.** AdamW with no weight decay on norms/biases/embeddings and
  layer-wise LR decay on the backbone (`--llrd`, deeper layers get a higher LR).
  `--llrd 1.0` reproduces the legacy flat `--backbone-lr-mult`.
- **Throughput.** TF32 + cudnn autotuner on by default; `--compile` and
  `--seed` are available.
- **Tunable objective.** The matcher/loss weights (`--class-weight`,
  `--mask-weight`, `--dice-weight`, `--no-object-weight`), PointRend sampling
  (`--train-num-points`) and mask-head depth (`--num-upscale-blocks`) are flags,
  persisted in the checkpoint so a tuned objective rebuilds on reload. After
  training, `eomt sweep` tunes the *inference* knobs (confidence/mask thresholds,
  `--max-det`, min mask area) over the val set — sharing one forward per batch —
  without retraining.

Resuming a run (`--resume <run|ckpt>`) restores the epoch, optimizer, EMA, and
`best_metric`, and recovers the dataset paths and architecture from the run's
`args.yaml` / checkpoint metadata.

## Masked-attention annealing

Training follows the EoMT recipe: the masked-attention probability is annealed
`1 → 0` linearly over the `[--mask-anneal-start, --mask-anneal-end]` fraction of
training, so the final stretch trains **mask-free** and matches efficient
(mask-less) inference. Validation and checkpointing run with masked attention
**off** (deterministic), so reported mAP and the saved buffer reflect how the
model is later run via `predict`/`val`. Disable with `--no-mask-anneal`.

## What's included

- Architecture (`s`/`b`/`l`) + DINOv2 init — `eomt.model`, `eomt.config`
- Optional secondary per-instance attribute heads — `eomt.aux_cls`, `eomt.config.AuxHeadSpec`
- COCO-format datasets (incl. per-instance attributes), Large-Scale Jitter +
  letterbox augmentations (torchvision v2), autodownload — `eomt.data`
- Training loop (AdamW + layer-wise LR decay + no-WD groups, cosine warmup,
  gradient accumulation, AMP, EMA weights, masked-attention annealing, aux-head
  loss, resume, per-epoch `metrics.csv`) — `eomt.engine.train`, `eomt.ema`
- COCO-mAP validation (`pycocotools`) with letterbox-aware mask remapping — `eomt.engine.validate`
- Mask2Former-style scoring (class confidence × mask objectness) — `eomt.postprocess`
- Inference + rendering (with attribute labels) — `eomt.engine.predict`, `eomt.visualize`

## Not included

Deployment-format export (ONNX / TensorRT / etc.) is intentionally out of scope.
A detect (true box head) family and a semantic-segmentation family are planned;
the code carries a `family` parameter so they can be added without API churn, but
only `instance` is implemented today.

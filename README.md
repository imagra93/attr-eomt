# attr-eomt

Standalone **EoMT** (Encoder-only Mask Transformer) for **instance segmentation**,
with one feature that sets it apart: **secondary per-instance classification heads**
("auxiliary classes"). Alongside the usual mask + class output, the model predicts
**one or several independent attributes for every detected instance** — and they
train and infer for free on top of segmentation, without inflating the primary
class space.

The name reflects exactly that: **attr-eomt** is EoMT extended with per-instance
**attr**ibute heads.

EoMT itself is a DINOv2-with-registers ViT whose last few transformer blocks are
augmented with learnable queries producing mask-classification output
(Mask2Former-style). This package builds it in three sizes, initializes the encoder
from DINOv2, and provides training, per-epoch COCO-mAP validation, and
inference/rendering — all behind a small `EoMT` class. It is a clean-room,
Apache-2.0-compatible reimplementation; weights you train are yours to release.

```python
from eomt import EoMT

model = EoMT("l")                            # fresh large model (DINOv2 backbone)
model.train(data="coco", epochs=50)          # COCO 2017 auto-downloads if missing

model = EoMT("runs/train/eomt-l")            # reload a run — size/classes/heads auto-detected
model.predict("images/", plot=True)          # render masks + per-instance attributes
```

---

## ⭐ Auxiliary classes (secondary per-instance classification)

The primary task (instance segmentation over `nc` classes) is **unchanged**. On top
of it you can attach **one or several independent secondary classifiers** — a
per-instance *attribute* predicted for each detected mask, read straight from that
query's embedding. You can have as many heads as your data defines.

This answers the "classes **and** subclasses per instance" need by encoding a
*subclass per instance* rather than flattening to a `class × subclass` product space
(which would wreck Hungarian matching and thin out per-class statistics). Because
EoMT is **NMS-free**, two overlapping same-class instances stay two distinct queries;
the attribute head separates them from their embeddings.

### Example: a bowl of fruit

One model segments each fruit (primary classes `apple` / `banana` / `orange` /
`pear`) and, for **every** detection, reads off two **independent** attribute heads —
`ripeness` (`unripe` / `turning` / `ripe`) and a quality `grade` (`A` / `B`). The
renderer prints the primary class + score on the first row and each attribute + its
confidence on the row beneath it.

![A bowl of fruit; each box labelled with its fruit class plus a ripeness and a grade](docs/examples/example_1.png)

`ripeness` and `grade` are *orthogonal* — they vary independently — which is exactly
the case that's awkward to fold into the primary class space. And because EoMT is
NMS-free, two apples (one ripe, one unripe) and two oranges stay four distinct
queries, each with its own `ripeness` and `grade`.

> The image above is a **simulation**: a stock photo with hand-placed detections fed
> through the package's own renderer ([`eomt.visualize.draw_instances`](eomt/visualize.py))
> to show the output format — not a trained model's predictions. The same pattern
> fits any "class **plus** per-instance sub-labels" task: **retail shelves → product +
> facing**, **cells / leaves → type + health**, **apparel → garment + pattern**.

### How it works

- **Embedding source.** Each head reads the per-query embedding — the input to EoMT's
  `class_predictor`, captured with a forward hook (`[B, Q, hidden]`).
- **Matching.** Supervision reuses EoMT's *own* Hungarian matcher
  (`model.eomt.criterion.matcher`), so every attribute is trained on the **same**
  query→GT assignment the detection loss used. The attribute is read *after* matching.
- **Loss.** Cross-entropy per head over matched queries, summed across heads and
  scaled by `aux_w` (default `1.0`), added to the segmentation loss. Empty-match
  batches contribute a graph-preserving zero.
- **Checkpoint selection stays `segm/mAP`.** The attribute "rides along": its per-head
  matched-query train accuracy is shown live and written to `metrics.csv`, but never
  drives `best.pt`.
- **Inference.** Each result attaches `aux = {head: {"ids", "probs"}}` for the kept
  detections, and `predict(plot=True)` renders each attribute next to the class label
  using names stored in the checkpoint.

### Data format (auto-discovered from the COCO JSON)

Attributes live **inside the COCO annotations** — each annotation is already a
per-instance object, so alignment is automatic and `pycocotools` still parses it. Two
additions to a standard COCO file:

```jsonc
{
  "categories": [ {"id": 1, "name": "apple"}, {"id": 2, "name": "banana"} ],

  "attributes": [                                       // NEW, top-level: per-head vocab(s)
    {"name": "ripeness", "categories": [{"id": 0, "name": "unripe"},
                                        {"id": 1, "name": "turning"},
                                        {"id": 2, "name": "ripe"}]},
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

- The top-level `attributes` list defines each head's vocabulary; raw ids are remapped
  to a contiguous `0..n-1` per head (so the non-contiguous `grade` ids `10`/`20` become
  `0`/`1`). `categories` may be omitted, in which case the id set is inferred.
- Per-annotation `attributes` is a `{head: raw_id}` map; a missing value defaults to `0`.
- A JSON with **no** `attributes` ⇒ detection-only, behaving exactly as before.

No YAML changes are needed — heads (count, classes, names) are discovered straight from
the JSON, the same as `nc`. A tiny, self-contained example (two heads, including a
non-contiguous id set) lives in [sample_data/](sample_data/).

---

## Sizes

| size | backbone        | hidden | layers | heads | queries |
|------|-----------------|--------|--------|-------|---------|
| `s`  | DINOv2-small    | 384    | 12     | 6     | 100     |
| `b`  | DINOv2-base     | 768    | 12     | 12    | 200     |
| `l`  | DINOv2-large    | 1024   | 24     | 16    | 200     |

Default input is a patch-14-aligned square (`644 = 14 × 46`) so DINOv2 weights load 1:1.

## Install

```bash
pip install attr-eomt                  # from PyPI
pip install "attr-eomt[logging]"       # + tensorboard/wandb
```

Or from source (editable, for development):

```bash
pip install -e ".[dev]"                # [dev] adds pytest/build/twine
```

## Pretrained weights

Checkpoints are hosted on the **Hugging Face Hub**, not bundled in the wheel.
Load one in a single line — it downloads once and is cached for later runs:

```python
from eomt import EoMT

model = EoMT.from_pretrained("imagra93/eomt-l-coco")   # downloads + caches
results = model.predict("images/", plot=True)
```

Equivalently, an `hf://` reference works anywhere a checkpoint path is accepted:

```python
EoMT("hf://imagra93/eomt-l-coco/model.pt").val(data="coco")
```

`from_pretrained` accepts `filename=` (which checkpoint in the repo, default
`model.pt`), `revision=` (branch/tag/commit) and `device=`.

To publish your own trained weights to the Hub (creates the repo if needed; needs
`huggingface-cli login` or `HF_TOKEN`):

```python
EoMT("runs/train/eomt-l").push_to_hub("your-username/eomt-l-coco")   # private by default
```

## Quickstart

Everything goes through one class. Initialize it from a **size** (a fresh model with a
pretrained DINOv2 backbone) or from a **checkpoint / run folder** (size, classes, image
size and any auxiliary heads are auto-detected):

```python
from eomt import EoMT

# Train on COCO 2017 (auto-downloaded on first run). batch 4 with the default
# nominal_batch 16 accumulates to an effective batch of 16.
EoMT("l").train(data="coco", epochs=50, batch=4)

# ...or any COCO-format dataset (point at its data.yaml):
EoMT("s").train(data="sample_data/data.yaml", epochs=1, batch=1)

# Validate a checkpoint (COCO segm + bbox mAP):
metrics = EoMT("runs/train/eomt-l").val(data="coco")

# Predict + render on an image or a folder; results carry boxes/scores/classes/masks
# (+ `aux` for models with auxiliary heads). plot=True writes annotated images:
results = EoMT("runs/train/eomt-l").predict("images/", plot=True)
```

Ready-to-run wrappers live in [scripts/](scripts/):

```bash
python scripts/train.py --size l --epochs 50 --batch 4     # COCO by default
python scripts/val.py     runs/train/eomt-l
python scripts/predict.py runs/train/eomt-l                 # uses scripts/sample_images
```

To resume or fine-tune, initialize from the checkpoint:

```python
EoMT("runs/train/eomt-l").train(data="coco", resume=True)   # continue the run
EoMT("runs/train/eomt-l").train(data="my.yaml", epochs=20)  # warm-start (fine-tune)
```

### Useful `train()` options

Passed as keyword arguments to `model.train(...)`:

| arg | default | effect |
|------|---------|--------|
| `nominal_batch` / `accum` | `16` / `0` | gradient accumulation to an **effective batch** of `nominal_batch` (EoMT recipe is 16); `accum=N` sets the step count explicitly |
| `ema` / `ema_decay` / `ema_tau` | `True` / `0.9999` / `2000` | validate & export `best.pt` from an **EMA** of the weights; decay + its warmup ramp |
| `llrd` | `0.85` | **layer-wise LR decay** on the DINOv2 backbone (`1.0` = flat `backbone_lr_mult`) |
| `min_scale` / `max_scale` | `0.1` / `2.0` | **Large-Scale Jitter** range (legacy stretch-style: `0.5` / `1.0`) |
| `letterbox` | `True` | aspect-preserving **letterbox** eval (vs legacy square stretch); recorded in the checkpoint |
| `flip_prob` | `0.5` | horizontal-flip probability. **Set `0`** for datasets with a left/right attribute — hflip mirrors pixels without swapping the laterality label |
| `mask_anneal` / `mask_anneal_start` / `mask_anneal_end` | `True` / `0.0` / `0.9` | anneal masked attention `1→0` over this fraction of training (EoMT recipe) |
| `aux_w` | `1.0` | weight on the summed secondary-head loss |

## Training recipe

Defaults follow the EoMT/Mask2Former fine-tuning recipe; each piece is a keyword
argument, so the legacy behaviour is one override away.

- **Effective batch via gradient accumulation.** LR / weight-decay / clip are tuned for
  an effective batch of 16, so training accumulates `round(nominal_batch / batch)`
  micro-batches per optimizer step. EoMT is a ViT (LayerNorm, no BatchNorm), so this is
  ~equivalent to a true large batch at a fraction of the memory.
- **EMA weights.** A moving average is validated and saved as `best.pt`; `last.pt` holds
  the live weights plus optimizer and EMA state for exact resume.
- **Large-Scale Jitter (LSJ).** Training resizes aspect-preserving over
  `[min_scale, max_scale]` then crops/pads to the square input — a strong scale aug.
- **Letterbox eval.** Validation/inference resize the long side and pad to a square; the
  padding is cropped back out in postprocessing. The mode is stored per-checkpoint so
  `val`/`predict` match training automatically.
- **Optimizer.** AdamW with no weight decay on norms/biases/embeddings and layer-wise LR
  decay on the backbone (`llrd`, deeper layers get a higher LR).
- **Tunable objective.** The matcher/loss weights (`class_weight`, `mask_weight`,
  `dice_weight`, `no_object_weight`), PointRend sampling (`train_num_points`) and
  mask-head depth (`num_upscale_blocks`) are arguments, persisted in the checkpoint so a
  tuned objective rebuilds on reload.
- **Masked-attention annealing.** The masked-attention probability is annealed `1 → 0`
  over `[mask_anneal_start, mask_anneal_end]` of training, so the final stretch trains
  **mask-free** and matches efficient (mask-less) inference. Validation and
  checkpointing run mask-free (deterministic).

## What's included

- Architecture (`s`/`b`/`l`) + DINOv2 init — `eomt.model`, `eomt.config`
- **Secondary per-instance attribute heads** — `eomt.aux_cls`, `eomt.config.AuxHeadSpec`
- COCO-format datasets (incl. per-instance attributes), Large-Scale Jitter + letterbox
  augmentations (torchvision v2), autodownload — `eomt.data`
- Training loop (AdamW + layer-wise LR decay + no-WD groups, cosine warmup, gradient
  accumulation, AMP, EMA weights, masked-attention annealing, aux-head loss, resume,
  per-epoch `metrics.csv`) — `eomt.engine.train`, `eomt.ema`
- COCO-mAP validation (`pycocotools`) with letterbox-aware mask remapping — `eomt.engine.validate`
- Mask2Former-style scoring (class confidence × mask objectness) — `eomt.postprocess`
- Inference + rendering (with attribute labels) — `eomt.engine.predict`, `eomt.visualize`

## Not included

Deployment-format export (ONNX / TensorRT / etc.) is intentionally out of scope. A
detect (true box head) family and a semantic-segmentation family are planned; the code
carries a `family` parameter so they can be added without API churn, but only
`instance` is implemented today.

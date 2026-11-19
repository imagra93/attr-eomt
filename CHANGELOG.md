# Changelog

## 3.1.0 (unreleased)

The multi-view post-training, rebuilt from what the 3.0 runs showed: the cross-view attention's null token never
learned (its logit stayed at its starting value), the matching head linked hardly better than plain re-identification
(link F1 0.68 vs 0.66), and the photos' own features never saw the other photos. **Multi-view checkpoints of 3.0 and
before are refused** (their single-photo starting checkpoints load as before, with identical predictions).

- **Cross-view attention, gated.** Each row's update is gated per head by `tanh` of a learned function of the row,
  starting at 0 (the sublayer starts as the identity), in place of the null token and the LayerScale. The patch tokens,
  pooled 2x2, attend to the other photos too and their update is spread back over the patches, so the mask features see
  the other photos (`xview: {patches, pool}`; ~+3 % per step). One attention call per set. The output projection has no
  bias, so a closed gate adds nothing.
- **Attention supervision** (`multiview.attn_w`): a loss pointing each linked query's cross-view attention at the same
  instance in the other photos (its mask and its query there).
- **The set matcher** (`eomt.matcher.SetMatcher`, replaces the matching head, `match_dim`, `match_residual` and the
  sigmoid matching loss): the detected instances of the photos, by their query embedding and mask box, attend to each
  other within and across photos; every two instances of two photos get a probability of being the same (a dual softmax
  with a "no partner in that photo" bin), trained on the `instance_id` links (`multiview.match_w`). Inference links with
  it (the photos predicted together as one set, any other two as a pair) at its calibrated probability threshold.
- **LoRA** (`multiview.lora: {rank, blocks, alpha}`, `eomt.lora`): the attention of the frozen ViT blocks below the query
  blocks adapts; the trainer's `best.pt` holds the merged weights, and any checkpoint with factors is merged on loading.
  Trainable group `lora`; the `match_head` group is now `matcher`.
- **Validation**: `sets/{match,reid}_oracle/*`, the links between the annotated instances themselves (linking with
  perfect detection); the blocks below the queries run once per set for the photos alone and together;
  `val.final_epochs` validates every epoch at the end of a run validated every `interval` epochs.
- **Training**: DDP no longer searches for unused parameters (every trainable parameter joins each step's graph) and
  keeps the gradients in its buckets.
- `EoMTModel.forward(trunk=..., keep_xview_attn=...)` and `EoMTEncoder.encode_trunk()`.
- **Groups of a subject's photos** (`eomt.engine.predict`): a photo may be in several groups (predicted in the first,
  context in the others, `group_owners`); `fill=True` (`subject_groups`, `predict_subject`, `predict(subject=True)`,
  `match`, `scripts/match.py --fill`) tops the last default group up with the photos before it, so 5 photos at 4 a
  time are two groups of 4 instead of 4 + 1; `similar_groups(features, max_views)` groups photos by
  `photo_similarity`: the cosine of one or several descriptions of each photo (e.g. a first model's class areas and the
  photo's appearance), averaged with `weights`, optionally `center`ed, filled the same way.
- **`photo_embed`**: every forward returns each photo's mean final patch token `(B, hidden)`; `predict_set` results
  carry it, and `predict_batch(photo_embed=True)`.
- `predict_batch` / `predict_set` / `predict_subject` take RGB arrays as well as PIL images, already preprocessed inputs
  (`prepared`: numpy or tensors, from `preprocess_numpy` or `preprocess_torch`) and the size each result is returned at
  (`sizes`).

## 3.0.0 (2026-11-19)

A cleanup for a public release: one way to do each thing, one config per concern, and the options no trained model
uses removed. **Checkpoints of 2.x load and predict exactly as before** (they are upgraded on load); run configs and
code written for 2.x need the changes below.

### Training: settings by concern

`train()` keeps the keywords of the run (data, `size`, `family`, `epochs`, `batch`, `nominal_batch`, `imgsz`, `device`,
`workers`, `seed`, `project` / `name`, `resume` / `init_weights`, `keep_empty`, `stop_after_epochs`, `logger`) and takes
the rest by section, each a dataclass or a mapping (and a section of the run config YAML):

| 2.x keyword | 3.0 |
|---|---|
| `lr0`, `weight_decay`, `llrd`, `backbone_lr_mult`, `poly_power`, `warmup_steps`, `clip_norm`, `ema`, `ema_decay`, `ema_tau` | `optim: {...}` |
| `mask_anneal`, `mask_anneal_start` / `_end`, `--mask-free` | `optim.mask_anneal` (true: the reference windows; false: mask-free) |
| `fpn_scales`, `num_upscale_blocks`, loss weights, `iou_aware_cls`, `quality_weight`, `aux_head_*` | `model: {...}` (IoU-aware targets and the quality head now on by default) |
| `aux_w`, `aux_w_per_head`, `aux_class_weights`, `aux_iou_gate`, `aux_class_gate` | `aux: {w, per_head, class_weights, iou_gate, class_gate}` |
| `val_interval`, `val_imgsz`, `conf_thres`, `max_det`, `val_group_by`, `best_key` | `val: {interval, imgsz, conf_thres, max_det, group_by, best_key}` |
| `flip_prob`, `min_scale`, `max_scale`, `train_aug` (YAML) | `aug: {...}` |
| `accum`, `amp_dtype`, `tf32`, `prefetch`, `letterbox`, `ddp` | removed (from `nominal_batch`; bf16 where supported; always on) |

Removed options: `box_head`, `deep_supervision=False`, `hires_*`, `aux_balance`, `aux_w_warmup`, the gradient probe.

### Several photos of one subject

`multiview` is a post-training on **real** sets (images with `subject_id`, annotations with `instance_id`):
`{xview, matching, match_w, gate, trainable, sets, val_sets}`. `xview` and `matching` are each optional: the matching
head trains without cross-view attention too. `trainable` takes the presets `new` and `query_blocks`, or a dict.
New: `sets.groups` (an image field) trains on the fixed groups of photos an application predicts together, each once
per epoch, with a `sets.p_random` share of random sets; validation scores those groups.

Removed: synthetic sets (`sets.cases`, distractors, mirrored copies, `p_subject`, `focus`), the `real` sub-section (the
run's own data is the sets' data), the refiner, view types, `xview.keys`, `xview_scale`, the InfoNCE and dustbin
matching losses (sigmoid only), `match_dim` / `match_residual` / `match_tau` (a residual head at the hidden size),
`match_neg_conf` (the inference threshold), `match_hard_*`, `linked_w`, `consistency_*`, `harm_w`, `val_synthetic`,
per-part `val_interval`. The metrics are `val/sets/...` (was `val/real/...`); the link sources are `match` and `reid`.

### Inference and API

- `EoMT.match()` is the one linking entry point, for any number of photos (`infer_match` is gone);
  `predict_subject(identities=True)` links in code. The grouping modes `partition` / `anchor` are gone (`groups=`).
- `predict()` batches (`batch=8`), decodes in the background, streams (`stream=True`) and runs in bf16 by default.
- `val()` defaults to the checkpoint's own validation data.
- Tracking moved to `examples/track.py`; int8 compression and the auxiliary box head were removed.
- Checkpoints load with `weights_only=True` and strictly; `eomt.migrate_checkpoint` upgrades 2.x ones.
- Messages go through the standard `eomt` logger (stdout, INFO) instead of `print`:
  `logging.getLogger("eomt").setLevel(logging.WARNING)` silences them.

### Fixes

- Unaugmented validation sets are placed as inference letterboxes the photos (they were placed at random): the stored
  link threshold is calibrated on the inputs inference sees, and fixed sets read back reproduce the trainer's numbers.
- Validation runs in bf16 like training and inference (it ran in fp16), in PyTorch's default numerics.
- Thin or tiny instances are no longer dropped from the set validation's ground truth.
- `last.pt` is written before validating; a resumed run reads the best so far from `best.pt` too.
- No GPU synchronisation per forward pass (masked-attention probabilities) or per training step; the final matching is
  solved once per step; `predict()` keeps no full-resolution float copies of the masks.

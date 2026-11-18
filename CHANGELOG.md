# Changelog

## 3.0.0

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

### Fixes

- Unaugmented validation sets are placed as inference letterboxes the photos (they were placed at random): the stored
  link threshold is calibrated on the inputs inference sees, and fixed sets read back reproduce the trainer's numbers.
- Validation runs in bf16 like training and inference (it ran in fp16), in PyTorch's default numerics.
- Thin or tiny instances are no longer dropped from the set validation's ground truth.
- `last.pt` is written before validating; a resumed run reads the best so far from `best.pt` too.
- No GPU synchronisation per forward pass (masked-attention probabilities) or per training step; the final matching is
  solved once per step; `predict()` keeps no full-resolution float copies of the masks.

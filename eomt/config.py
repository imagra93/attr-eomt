"""EoMT size presets and HuggingFace config construction.

EoMT (Encoder-only Mask Transformer, "Your ViT is Secretly an Image
Segmentation Model", CVPR 2025) wraps a DINOv2-with-registers ViT whose last
``num_blocks`` transformer layers are augmented with learnable queries that
produce mask-classification output. Both upstream pieces are permissive
(Apache-2.0 EoMT/transformers, Apache-2.0 DINOv2-with-registers), so weights
trained from a DINOv2 initialization are yours to release.

The three size codes map to the DINOv2 backbone widths:

    s -> DINOv2-small   (hidden 384, 12 layers, 6 heads)
    b -> DINOv2-base    (hidden 768, 12 layers, 12 heads)
    l -> DINOv2-large   (hidden 1024, 24 layers, 16 heads)

The default ``image_size``/``patch_size`` (644 / 14) are chosen so the
DINOv2-with-registers patch grid loads 1:1 with no kernel interpolation
(644 = 14 x 46).
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Default square input size (patch-14 aligned: 644 = 14 x 46).
DEFAULT_IMAGE_SIZE = 644
#: DINOv2 / EoMT patch size.
PATCH_SIZE = 14


@dataclass
class AuxHeadSpec:
    """One secondary per-instance classifier ("attribute") riding on EoMT.

    ``name`` keys the head everywhere (model ``ModuleDict``, COCO ``attributes``
    field, checkpoint metadata). ``names`` maps the contiguous ``0..num_classes-1``
    ids to human labels. Several specs ⇒ several independent heads.
    """

    name: str
    num_classes: int
    names: dict[int, str] = field(default_factory=dict)


def aux_specs_to_meta(specs: list[AuxHeadSpec] | None) -> list[dict]:
    """Serialize aux-head specs for a checkpoint."""
    return [
        {"name": s.name, "num_classes": int(s.num_classes), "names": dict(s.names)}
        for s in (specs or [])
    ]


def aux_specs_from_meta(meta: list[dict] | None) -> list[AuxHeadSpec]:
    """Rebuild aux-head specs from checkpoint metadata."""
    out: list[AuxHeadSpec] = []
    for d in meta or []:
        names = {int(k): str(v) for k, v in (d.get("names") or {}).items()}
        out.append(AuxHeadSpec(str(d["name"]), int(d["num_classes"]), names))
    return out


@dataclass(frozen=True)
class EoMTSizeConfig:
    """Per-size EoMT / DINOv2 hyper-parameters."""

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    dinov2_repo: str
    num_queries: int = 200
    num_blocks: int = 4
    num_register_tokens: int = 4
    num_upscale_blocks: int = 2


# DINOv2-with-registers backbones (Apache-2.0, facebook/*).
EOMT_CONFIGS: dict[str, EoMTSizeConfig] = {
    "s": EoMTSizeConfig(
        hidden_size=384,
        num_hidden_layers=12,
        num_attention_heads=6,
        dinov2_repo="facebook/dinov2-with-registers-small",
        num_queries=100,
    ),
    "b": EoMTSizeConfig(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        dinov2_repo="facebook/dinov2-with-registers-base",
        num_queries=200,
    ),
    "l": EoMTSizeConfig(
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        dinov2_repo="facebook/dinov2-with-registers-large",
        num_queries=200,
    ),
}

#: Available size codes.
SIZES = tuple(EOMT_CONFIGS.keys())

#: Encoder hidden_size -> size code (used to detect size from a checkpoint).
HIDDEN_TO_SIZE = {cfg.hidden_size: code for code, cfg in EOMT_CONFIGS.items()}

#: Segmentation-loss / criterion hyper-parameters exposed for tuning. Values are
#: the upstream (Mask2Former/EoMT) defaults — passing the defaults reproduces the
#: previous behaviour. ``build_eomt_config`` did NOT previously forward these, so
#: every build silently reset them to these defaults; they are now threaded and
#: persisted in the checkpoint so a tuned objective survives reload.
DEFAULT_LOSS_WEIGHTS: dict = {
    "no_object_weight": 0.1,
    "class_weight": 2.0,
    "mask_weight": 5.0,
    "dice_weight": 5.0,
    "train_num_points": 12544,
    "oversample_ratio": 3.0,
    "importance_sample_ratio": 0.75,
}

#: Detection (box-head) criterion weights. The shared ``no_object_weight`` /
#: ``class_weight`` keep the same meaning as for masks; the mask/dice terms are
#: replaced by DETR's box ``l1_weight`` and ``giou_weight`` (see :mod:`eomt.box_loss`).
DETECT_LOSS_WEIGHTS: dict = {
    "no_object_weight": 0.1,
    "class_weight": 2.0,
    "l1_weight": 5.0,
    "giou_weight": 2.0,
}


def _loss_weight_defaults(family: str) -> dict:
    return DETECT_LOSS_WEIGHTS if family == "detect" else DEFAULT_LOSS_WEIGHTS


def normalize_loss_weights(lw: dict | None, family: str = "instance") -> dict:
    """Fill a loss-weight dict with defaults for ``family``, rejecting unknown keys.

    ``None`` values fall back to the default (so partial dicts work). Raises on keys
    that are not real criterion params for the family to catch typos early. The
    ``"instance"`` family validates against the mask/dice keys; ``"detect"`` against
    the box L1/GIoU keys.
    """
    out = dict(_loss_weight_defaults(family))
    if lw:
        unknown = set(lw) - set(out)
        if unknown:
            raise ValueError(
                f"unknown loss_weights keys {sorted(unknown)} for family {family!r}; "
                f"valid keys are {sorted(out)}."
            )
        out.update({k: v for k, v in lw.items() if v is not None})
    if "train_num_points" in out:
        out["train_num_points"] = int(out["train_num_points"])
    return out


def build_eomt_config(
    size: str,
    *,
    nc: int,
    image_size: int = DEFAULT_IMAGE_SIZE,
    patch_size: int = PATCH_SIZE,
    names: dict[int, str] | None = None,
    num_upscale_blocks: int | None = None,
    no_object_weight: float = 0.1,
    class_weight: float = 2.0,
    mask_weight: float = 5.0,
    dice_weight: float = 5.0,
    train_num_points: int = 12544,
    oversample_ratio: float = 3.0,
    importance_sample_ratio: float = 0.75,
):
    """Build a ``transformers.EomtConfig`` for the given size code.

    Args:
        size: One of ``"s"``, ``"b"``, ``"l"``.
        nc: Number of foreground classes (the HF model adds the +1 null class).
        image_size: Square input size; must be divisible by ``patch_size``.
        patch_size: Patch size (14 for DINOv2-with-registers).
        names: Optional ``{class_index: name}`` mapping for ``id2label``.
        num_upscale_blocks: Mask-head upsampling blocks (``None`` = size preset
            default, 2). Each block doubles the mask-logit resolution; raising it
            sharpens small/thin masks but changes the ``upscale_block`` weight
            shapes (a 2-block checkpoint cannot warm-start a 3-block head 1:1).
        no_object_weight: CE weight on the null class for unmatched queries; lower
            ⇒ the model fires more queries (higher recall / more detections).
        class_weight / mask_weight / dice_weight: matcher + loss weights for the
            classification, per-pixel mask BCE and (scale-invariant) dice terms.
        train_num_points / oversample_ratio / importance_sample_ratio: PointRend
            sampling for the mask loss; more points ⇒ sharper boundaries.
    """
    from transformers import EomtConfig

    if size not in EOMT_CONFIGS:
        raise ValueError(f"Unknown EoMT size {size!r}; choose from {list(EOMT_CONFIGS)}.")
    if image_size % patch_size:
        raise ValueError(
            f"image_size={image_size} must be divisible by patch_size={patch_size} "
            "(DINOv2 grid)."
        )
    cfg = EOMT_CONFIGS[size]

    id2label = names if names is not None else {i: f"class_{i}" for i in range(nc)}
    id2label = {int(k): str(v) for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

    return EomtConfig(
        hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_queries=cfg.num_queries,
        num_blocks=cfg.num_blocks,
        num_register_tokens=cfg.num_register_tokens,
        num_upscale_blocks=(
            cfg.num_upscale_blocks if num_upscale_blocks is None else int(num_upscale_blocks)
        ),
        image_size=image_size,
        patch_size=patch_size,
        id2label=id2label,
        label2id=label2id,
        no_object_weight=no_object_weight,
        class_weight=class_weight,
        mask_weight=mask_weight,
        dice_weight=dice_weight,
        train_num_points=int(train_num_points),
        oversample_ratio=oversample_ratio,
        importance_sample_ratio=importance_sample_ratio,
    )

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

from dataclasses import dataclass

#: Default square input size (patch-14 aligned: 644 = 14 x 46).
DEFAULT_IMAGE_SIZE = 644
#: DINOv2 / EoMT patch size.
PATCH_SIZE = 14


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


def build_eomt_config(
    size: str,
    *,
    nc: int,
    image_size: int = DEFAULT_IMAGE_SIZE,
    patch_size: int = PATCH_SIZE,
    names: dict[int, str] | None = None,
):
    """Build a ``transformers.EomtConfig`` for the given size code.

    Args:
        size: One of ``"s"``, ``"b"``, ``"l"``.
        nc: Number of foreground classes (the HF model adds the +1 null class).
        image_size: Square input size; must be divisible by ``patch_size``.
        patch_size: Patch size (14 for DINOv2-with-registers).
        names: Optional ``{class_index: name}`` mapping for ``id2label``.
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
        num_upscale_blocks=cfg.num_upscale_blocks,
        image_size=image_size,
        patch_size=patch_size,
        id2label=id2label,
        label2id=label2id,
    )

"""EoMT architecture wrapper and DINOv2 backbone initialization.

``EoMTModel`` is a thin ``nn.Module`` around HuggingFace's
``EomtForUniversalSegmentation``. Its forward returns the HF segmentation loss
during training (class CE + mask CE + dice, computed inside the HF model with
Hungarian matching, PointRend point sampling and per-layer auxiliary losses) and
the raw query logits dict at inference.

``load_dinov2_backbone`` initializes the ViT encoder from pretrained
DINOv2-with-registers weights; the mask/class prediction head is left random and
learned during training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from .config import (
    DEFAULT_IMAGE_SIZE,
    EOMT_CONFIGS,
    PATCH_SIZE,
    AuxHeadSpec,
    build_eomt_config,
)


def _patch_eomt_sample_point_for_amp() -> None:
    """Make EoMT's PointRend sampler AMP-safe (idempotent module-level patch).

    ``EomtHungarianMatcher`` and ``EomtLoss`` build point coordinates with
    ``torch.rand(...)`` — fp32 — and feed them to ``grid_sample`` alongside the
    mask logits, which are fp16 under ``torch.autocast``. ``grid_sample`` requires
    the grid and the input to share a dtype, so a CUDA+AMP training step otherwise
    fails with ``"expected scalar type Half but found Float"``. We wrap the
    module-level ``sample_point`` once to cast the coordinates to the feature
    dtype; the matcher/loss resolve the name at call time, so the one shim covers
    every call site (internal loss and our re-run matcher in ``aux_cls``).
    """
    try:
        from transformers.models.eomt import modeling_eomt as _m
    except Exception:  # pragma: no cover - transformers internal layout changed
        return
    if getattr(_m, "_eomt_amp_sample_point_patched", False):
        return
    _orig = _m.sample_point

    def _sample_point(input_features, point_coordinates, add_dim=False, **kwargs):
        if point_coordinates.dtype != input_features.dtype:
            point_coordinates = point_coordinates.to(input_features.dtype)
        return _orig(input_features, point_coordinates, add_dim=add_dim, **kwargs)

    _m.sample_point = _sample_point
    _m._eomt_amp_sample_point_patched = True


#: Default secondary-head architecture for new runs: a small 2-layer MLP.
#: ``hidden=None`` means "use the encoder ``hidden_size``"; ``layers=1`` is a bare
#: linear probe (the pre-MLP behaviour, kept for old checkpoints).
DEFAULT_AUX_HEAD_ARCH: dict = {"layers": 2, "hidden": None, "dropout": 0.0}


def _normalize_aux_arch(arch: dict | None) -> dict:
    """Fill an aux-head arch dict with defaults (``layers`` / ``hidden`` / ``dropout``)."""
    arch = dict(arch or {})
    return {
        "layers": int(arch.get("layers", DEFAULT_AUX_HEAD_ARCH["layers"])),
        "hidden": arch.get("hidden", DEFAULT_AUX_HEAD_ARCH["hidden"]),
        "dropout": float(arch.get("dropout", DEFAULT_AUX_HEAD_ARCH["dropout"])),
    }


def _build_aux_head(in_dim: int, num_classes: int, arch: dict) -> nn.Module:
    """Build one secondary-head module from a normalized arch dict.

    ``layers <= 1`` ⇒ a bare ``nn.Linear`` (linear probe). Otherwise a small MLP:
    ``(Linear -> LayerNorm -> GELU -> [Dropout]) x (layers-1) -> Linear``.
    """
    layers = int(arch["layers"])
    if layers <= 1:
        return nn.Linear(in_dim, num_classes)
    hidden = int(arch["hidden"] or in_dim)
    dropout = float(arch["dropout"])
    mods: list[nn.Module] = []
    d = in_dim
    for _ in range(layers - 1):
        mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU()]
        if dropout > 0:
            mods.append(nn.Dropout(dropout))
        d = hidden
    mods.append(nn.Linear(d, num_classes))
    return nn.Sequential(*mods)


def build_model(
    size: str = "l",
    *,
    nc: int = 80,
    imgsz: int = DEFAULT_IMAGE_SIZE,
    names: dict[int, str] | None = None,
    family: str = "instance",
    aux_heads: list[AuxHeadSpec] | None = None,
    aux_head_arch: dict | None = None,
) -> "EoMTModel":
    """Build an :class:`EoMTModel`.

    ``family`` is accepted for forward-compatibility (a detect/box-head family
    and a semantic family may be added later); only ``"instance"`` is supported.
    ``aux_heads`` adds one secondary per-instance classifier per spec;
    ``aux_head_arch`` sets their shared network shape (see ``_build_aux_head``).
    """
    if family != "instance":
        raise NotImplementedError(
            f"family={family!r} is not implemented yet; only 'instance' is supported."
        )
    return EoMTModel(
        size=size,
        nc=nc,
        image_size=imgsz,
        patch_size=PATCH_SIZE,
        names=names,
        aux_heads=aux_heads,
        aux_head_arch=aux_head_arch,
    )


class EoMTModel(nn.Module):
    """Thin ``nn.Module`` wrapping ``EomtForUniversalSegmentation``.

    The forward returns the raw HF output dict during inference, or the scalar
    loss when ``mask_labels`` / ``class_labels`` are supplied (training).
    """

    def __init__(
        self,
        size: str = "l",
        nc: int = 80,
        image_size: int = 644,
        patch_size: int = PATCH_SIZE,
        names: dict[int, str] | None = None,
        aux_heads: list[AuxHeadSpec] | None = None,
        aux_head_arch: dict | None = None,
    ):
        super().__init__()
        from transformers import EomtForUniversalSegmentation

        _patch_eomt_sample_point_for_amp()  # AMP-safe grid_sample (see fn docstring)
        self.size = size
        self.nc = nc
        self.image_size = image_size
        self.patch_size = patch_size
        self.names = names
        config = build_eomt_config(
            size, nc=nc, image_size=image_size, patch_size=patch_size, names=names
        )
        self.config = config
        self.eomt = EomtForUniversalSegmentation(config)

        # Secondary per-instance heads (attributes). Each reads the per-query
        # embedding — the input to ``class_predictor`` — captured by a hook.
        # ``aux_head_arch`` (a small MLP by default) is the shared head shape; it is
        # persisted in the checkpoint so reload rebuilds the same modules.
        self.aux_specs: list[AuxHeadSpec] = list(aux_heads or [])
        self.aux_head_arch: dict = _normalize_aux_arch(aux_head_arch)
        self.aux_heads = nn.ModuleDict(
            {
                s.name: _build_aux_head(config.hidden_size, s.num_classes, self.aux_head_arch)
                for s in self.aux_specs
            }
        )
        self._query_embed: torch.Tensor | None = None
        if self.aux_heads:
            self.eomt.class_predictor.register_forward_hook(self._capture_query_embed)

    def _capture_query_embed(self, _module, inputs, _output):
        # input to class_predictor is the per-query embedding [B, Q, hidden];
        # the last call per forward is the final layer — exactly what we want.
        self._query_embed = inputs[0]

    def forward(
        self,
        pixel_values: torch.Tensor,
        mask_labels: list[torch.Tensor] | None = None,
        class_labels: list[torch.Tensor] | None = None,
    ):
        self._query_embed = None
        out = self.eomt(
            pixel_values=pixel_values,
            mask_labels=mask_labels,
            class_labels=class_labels,
        )
        training = mask_labels is not None and class_labels is not None
        result = {
            "masks_queries_logits": out.masks_queries_logits,
            "class_queries_logits": out.class_queries_logits,
        }
        if self.aux_heads and self._query_embed is not None:
            q = self._query_embed
            result["query_embed"] = q
            # Per-query logits over ALL queries are only consumed by postprocess at
            # inference; in training the aux loss applies each head to the matched
            # subset only, so skip the full-query application here.
            if not training:
                result["aux_queries_logits"] = {
                    name: head(q) for name, head in self.aux_heads.items()
                }
        if training:
            result["loss"] = out.loss
        return result


# ---------------------------------------------------------------------------
# DINOv2 backbone weight loading
# ---------------------------------------------------------------------------


def _remap_dinov2_key(key: str) -> str | None:
    """Map a ``Dinov2WithRegistersModel`` state-dict key onto an EoMT param name.

    Returns ``None`` for DINOv2 keys with no EoMT counterpart (e.g. mask_token).
    """
    # Final layernorm + embeddings are shared 1:1, except position embeddings
    # (DINOv2: plain parameter; EoMT: nn.Embedding -> ``.weight``) handled by caller.
    if key == "embeddings.mask_token":
        return None
    if key in (
        "embeddings.cls_token",
        "embeddings.register_tokens",
        "embeddings.patch_embeddings.projection.weight",
        "embeddings.patch_embeddings.projection.bias",
        "layernorm.weight",
        "layernorm.bias",
    ):
        return key

    if key.startswith("encoder.layer."):
        rest = key[len("encoder.layer.") :]
        idx, tail = rest.split(".", 1)
        tail = (
            tail.replace("attention.attention.query", "attention.q_proj")
            .replace("attention.attention.key", "attention.k_proj")
            .replace("attention.attention.value", "attention.v_proj")
            .replace("attention.output.dense", "attention.out_proj")
        )
        return f"layers.{idx}.{tail}"

    return None


def _resize_patch_projection(weight: torch.Tensor, target_patch: int) -> torch.Tensor:
    """Bilinearly resize a ``(out, 3, p, p)`` patch-embed conv kernel to ``target_patch``."""
    if weight.shape[-1] == target_patch:
        return weight
    return F.interpolate(
        weight, size=(target_patch, target_patch), mode="bilinear", align_corners=False
    )


def _resize_pos_embed(dinov2_pos: torch.Tensor, target_weight: torch.Tensor) -> torch.Tensor:
    """Map DINOv2 position embeddings onto EoMT's patches-only pos-embed table.

    ``dinov2_pos`` is ``(1, 1 + Hd*Wd, dim)`` (cls + patches). EoMT's
    ``position_embeddings.weight`` is patches-only ``(He*We, dim)`` — the cls /
    register / query tokens carry no learned position there. The DINOv2 cls
    position is dropped; the patch grid is bicubically interpolated when the grids
    differ (e.g. DINOv2 pretrained at 518 -> our 644).
    """
    import math

    dim = dinov2_pos.shape[-1]
    src_patches = dinov2_pos[0][1:]  # drop cls position
    n_src = src_patches.shape[0]
    n_tgt = target_weight.shape[0]
    if n_src == n_tgt:
        return src_patches
    hs = int(math.isqrt(n_src))
    ht = int(math.isqrt(n_tgt))
    grid = src_patches.reshape(1, hs, hs, dim).permute(0, 3, 1, 2)
    grid = F.interpolate(grid, size=(ht, ht), mode="bicubic", align_corners=False)
    return grid.permute(0, 2, 3, 1).reshape(n_tgt, dim)


def load_dinov2_backbone(model: EoMTModel, *, verbose: bool = True) -> dict[str, int]:
    """Initialize the EoMT ViT encoder from pretrained DINOv2-with-registers weights.

    Loads ``facebook/dinov2-with-registers-<size>`` (Apache-2.0) and copies the
    encoder/embedding tensors into the EoMT model in place. The EoMT prediction
    head (queries, upscale blocks, mask MLP, class head) stays randomly
    initialized and is learned during training.

    Returns a ``{"matched", "skipped", "interpolated"}`` count dict.
    """
    from transformers import Dinov2WithRegistersModel

    size_cfg = EOMT_CONFIGS[model.size]
    if verbose:
        print(f"[eomt] loading DINOv2 backbone: {size_cfg.dinov2_repo}")
    dino = Dinov2WithRegistersModel.from_pretrained(size_cfg.dinov2_repo)
    dino_sd = dino.state_dict()

    eomt_sd = model.eomt.state_dict()
    target_patch = model.patch_size

    new_sd: dict[str, torch.Tensor] = {}
    matched = skipped = interpolated = 0

    for dk, dv in dino_sd.items():
        if dk == "embeddings.position_embeddings":
            tgt = eomt_sd["embeddings.position_embeddings.weight"]
            resized = _resize_pos_embed(dv, tgt)
            if resized.shape != tgt.shape:
                skipped += 1
                continue
            new_sd["embeddings.position_embeddings.weight"] = resized
            matched += 1
            if dv.shape[1] - 1 != tgt.shape[0]:
                interpolated += 1
            continue

        ek = _remap_dinov2_key(dk)
        if ek is None or ek not in eomt_sd:
            skipped += 1
            continue

        tv = eomt_sd[ek]
        if ek == "embeddings.patch_embeddings.projection.weight":
            dv = _resize_patch_projection(dv, target_patch)
            if dv.shape[-1] != target_patch:
                interpolated += 1
        if dv.shape != tv.shape:
            skipped += 1
            continue
        new_sd[ek] = dv
        matched += 1

    model.eomt.load_state_dict(new_sd, strict=False)
    if verbose:
        print(
            f"[eomt] DINOv2 -> EoMT: matched={matched} skipped={skipped} "
            f"interpolated={interpolated}; prediction head left random."
        )
    # Guard against a silent half-random backbone: if a future transformers key
    # rename broke the remap, ``strict=False`` would hide it and only this count
    # would drop. Expect ~ all encoder/embedding tensors to copy across.
    expected = sum(1 for dk in dino_sd if _remap_dinov2_key(dk) is not None) + 1  # +pos-embed
    if matched < 0.8 * expected:
        import warnings

        warnings.warn(
            f"[eomt] DINOv2 backbone load matched only {matched}/{expected} expected "
            "tensors — the encoder is largely RANDOM. The DINOv2->EoMT key remap is "
            "likely stale (transformers version change). Training will be far worse.",
            RuntimeWarning,
            stacklevel=2,
        )
    return {"matched": matched, "skipped": skipped, "interpolated": interpolated}

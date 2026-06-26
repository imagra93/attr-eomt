"""EoMT architecture, model wrapper and DINOv2 backbone initialization.

This module holds the full pure-PyTorch EoMT forward path plus the trainable
wrapper around it:

* :class:`EoMTEncoder` reimplements ``transformers.EomtForUniversalSegmentation``'s
  forward — a DINOv2-with-registers ViT whose last ``num_blocks`` layers carry
  learnable queries with masked attention, plus the mask/class prediction heads —
  using only ``torch`` (no transformers *model code*). Submodule attribute names
  mirror the HF model **exactly**, so a checkpoint trained with the HF model loads
  with zero missing/unexpected keys and produces numerically identical outputs
  (verified on CPU/fp32 in ``tests/test_parity.py``). Only the config
  (``transformers.EomtConfig``) and the DINOv2 backbone loader are still sourced
  from transformers.
* :class:`EoMTModel` is a thin ``nn.Module`` around :class:`EoMTEncoder`. Its
  forward returns the segmentation loss during training (class CE + mask CE + dice,
  computed with Hungarian matching, PointRend point sampling and per-layer deep
  supervision) and the raw query logits dict at inference. It also owns the
  optional secondary per-instance classification heads (attributes).
* :func:`load_dinov2_backbone` initializes the ViT encoder from pretrained
  DINOv2-with-registers weights; the mask/class prediction head is left random and
  learned during training.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from .box_loss import DetectionLoss
from .config import (
    DEFAULT_IMAGE_SIZE,
    EOMT_CONFIGS,
    PATCH_SIZE,
    AuxHeadSpec,
    build_eomt_config,
    normalize_loss_weights,
)
from .loss import EoMTLoss
from .preprocess import IMAGENET_MEAN, IMAGENET_STD

#: Supported task families. ``"instance"`` is the original mask-classification head;
#: ``"detect"`` swaps the mask head for a DETR-style box-regression head.
FAMILIES = ("instance", "detect")


# ---------------------------------------------------------------------------
# EoMT architecture (pure PyTorch; submodule names mirror the HF model exactly)
# ---------------------------------------------------------------------------


@dataclass
class EoMTOutput:
    """Minimal output object; attribute access mirrors the HF dataclass.

    ``masks_queries_logits`` is populated for the ``"instance"`` family;
    ``pred_boxes`` (per-query normalized ``cxcywh``) for the ``"detect"`` family.
    """

    loss: torch.Tensor | None = None
    class_queries_logits: torch.Tensor | None = None
    masks_queries_logits: torch.Tensor | None = None
    pred_boxes: torch.Tensor | None = None
    last_hidden_state: torch.Tensor | None = None


def _act(name: str):
    """Resolve an activation by config name (only 'gelu' is used by the presets)."""
    if name == "gelu":
        return nn.GELU()  # exact erf gelu, matches ACT2FN['gelu']
    raise NotImplementedError(f"hidden_act={name!r} not supported by EoMTEncoder.")


class PatchEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        ps = config.patch_size
        self.num_channels = config.num_channels
        self.num_patches = (config.image_size // ps) * (config.image_size // ps)
        self.projection = nn.Conv2d(config.num_channels, config.hidden_size, kernel_size=ps, stride=ps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.projection(pixel_values).flatten(2).transpose(1, 2)


class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.register_tokens = nn.Parameter(torch.zeros(1, config.num_register_tokens, config.hidden_size))
        self.patch_embeddings = PatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.num_prefix_tokens = 1 + config.num_register_tokens
        self.position_embeddings = nn.Embedding(num_patches, config.hidden_size)
        self.register_buffer("position_ids", torch.arange(num_patches).expand((1, -1)), persistent=False)
        # The pos-embed table is stored at the *trained* grid; ``forward`` interpolates
        # it to whatever grid the actual input implies, so the model runs at any size.
        self.patch_size = config.patch_size
        g = config.image_size // config.patch_size
        self.base_grid = (g, g)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embeddings.projection.weight.dtype
        embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        register_tokens = self.register_tokens.expand(batch_size, -1, -1)
        # Native size -> use the stored table directly (identical numerics); otherwise
        # bicubically interpolate the trained grid onto the actual input grid.
        grid = (pixel_values.shape[-2] // self.patch_size, pixel_values.shape[-1] // self.patch_size)
        if grid == self.base_grid:
            pos = self.position_embeddings(self.position_ids)
        else:
            pos = _interp_pos_grid(self.position_embeddings.weight, self.base_grid, grid)[None]
        embeddings = embeddings + pos
        embeddings = torch.cat([cls_tokens, register_tokens, embeddings], dim=1)
        return self.dropout(embeddings)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads.")
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, n, _ = hidden_states.shape
        shape = (b, n, self.num_heads, self.head_dim)
        q = self.q_proj(hidden_states).view(shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(shape).transpose(1, 2)
        # Same op HF's sdpa path runs: an additive float mask (or None) is added to
        # the scaled QK^T scores. dropout only in training.
        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            scale=self.scale,
            is_causal=False,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attn = attn.transpose(1, 2).reshape(b, n, self.embed_dim)
        return self.out_proj(attn)


class LayerScale(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lambda1 = nn.Parameter(config.layerscale_value * torch.ones(config.hidden_size))

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return hidden_state * self.lambda1


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_features = int(config.hidden_size * config.mlp_ratio)
        self.fc1 = nn.Linear(config.hidden_size, hidden_features, bias=True)
        self.activation = _act(config.hidden_act)
        self.fc2 = nn.Linear(hidden_features, config.hidden_size, bias=True)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(hidden_state)))


class Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.use_swiglu_ffn:
            raise NotImplementedError("use_swiglu_ffn=True is not supported by EoMTEncoder.")
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = Attention(config)
        self.layer_scale1 = LayerScale(config)
        self.drop_path = nn.Identity()  # drop_path_rate is 0 for all presets
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MLP(config)
        self.layer_scale2 = LayerScale(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        attn = self.attention(self.norm1(hidden_states), attention_mask)
        attn = self.layer_scale1(attn)
        hidden_states = self.drop_path(attn) + hidden_states
        out = self.norm2(hidden_states)
        out = self.mlp(out)
        out = self.layer_scale2(out)
        return self.drop_path(out) + hidden_states


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = hidden_state.permute(0, 2, 3, 1)
        hidden_state = F.layer_norm(hidden_state, self.normalized_shape, self.weight, self.bias, self.eps)
        return hidden_state.permute(0, 3, 1, 2)


class ScaleLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        h = config.hidden_size
        self.conv1 = nn.ConvTranspose2d(h, h, kernel_size=2, stride=2)
        self.activation = _act(config.hidden_act)
        self.conv2 = nn.Conv2d(h, h, kernel_size=3, padding=1, groups=h, bias=False)
        self.layernorm2d = LayerNorm2d(h)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.conv2(hidden_states)
        return self.layernorm2d(hidden_states)


class ScaleBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block = nn.ModuleList([ScaleLayer(config) for _ in range(config.num_upscale_blocks)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for block in self.block:
            hidden_states = block(hidden_states)
        return hidden_states


class MaskHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        h = config.hidden_size
        self.fc1 = nn.Linear(h, h)
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, h)
        self.activation = _act(config.hidden_act)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.activation(self.fc1(hidden_states))
        hidden_states = self.activation(self.fc2(hidden_states))
        return self.fc3(hidden_states)


class BoxHead(nn.Module):
    """DETR-style box-regression head: per-query MLP -> sigmoid ``(cx, cy, w, h)``.

    Reads the same per-query embedding the mask head does and emits a normalized
    ``cxcywh`` box in ``[0, 1]`` (relative to the square model input). A 3-layer MLP
    (mirrors :class:`MaskHead`'s depth); the final layer maps to 4.
    """

    def __init__(self, config):
        super().__init__()
        h = config.hidden_size
        self.fc1 = nn.Linear(h, h)
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, 4)
        self.activation = _act(config.hidden_act)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.activation(self.fc1(hidden_states))
        hidden_states = self.activation(self.fc2(hidden_states))
        return self.fc3(hidden_states).sigmoid()


class EoMTEncoder(nn.Module):
    """Pure-PyTorch ``EomtForUniversalSegmentation`` (forward + loss).

    ``family="instance"`` is the original mask-classification head (mask + class).
    ``family="detect"`` swaps the mask head for a :class:`BoxHead` and the mask/dice
    criterion for :class:`~eomt.box_loss.DetectionLoss`; masked attention is not used
    (there is no predicted mask to focus on), so the last blocks do full attention.
    """

    def __init__(self, config, family: str = "instance"):
        super().__init__()
        self.config = config
        self.family = family
        self.num_hidden_layers = config.num_hidden_layers
        self.embeddings = Embeddings(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.query = nn.Embedding(config.num_queries, config.hidden_size)
        self.layers = nn.ModuleList([Layer(config) for _ in range(config.num_hidden_layers)])
        self.class_predictor = nn.Linear(config.hidden_size, config.num_labels + 1)
        self.grid_size = (config.image_size // config.patch_size, config.image_size // config.patch_size)

        if family == "detect":
            # Box head only; no mask head / upscale block, so no unused params are
            # saved into the checkpoint. ``l1_weight``/``giou_weight`` are stashed on
            # the config by ``EoMTModel`` (EomtConfig has no box-weight fields).
            self.box_head = BoxHead(config)
            self.weight_dict = {
                "loss_cross_entropy": config.class_weight,
                "loss_bbox": float(getattr(config, "l1_weight", 5.0)),
                "loss_giou": float(getattr(config, "giou_weight", 2.0)),
            }
            self.criterion = DetectionLoss(config=config, weight_dict=self.weight_dict)
        else:
            self.upscale_block = ScaleBlock(config)
            self.mask_head = MaskHead(config)
            self.weight_dict = {
                "loss_cross_entropy": config.class_weight,
                "loss_mask": config.mask_weight,
                "loss_dice": config.dice_weight,
            }
            self.criterion = EoMTLoss(config=config, weight_dict=self.weight_dict)
        self.register_buffer("attn_mask_probs", torch.ones(config.num_blocks))

    # --- loss plumbing (mirrors HF get_loss_dict / get_loss) ----------------

    def get_loss_dict(self, masks_queries_logits, class_queries_logits, mask_labels, class_labels):
        loss_dict = self.criterion(
            masks_queries_logits=masks_queries_logits,
            class_queries_logits=class_queries_logits,
            mask_labels=mask_labels,
            class_labels=class_labels,
            auxiliary_predictions=None,
        )
        for key, weight in self.weight_dict.items():
            for loss_key, loss in loss_dict.items():
                if key in loss_key:
                    loss_dict[loss_key] = loss * weight
        return loss_dict

    def get_loss(self, loss_dict):
        return sum(loss_dict.values())

    # --- prediction heads (mirrors HF predict) ------------------------------

    def predict(self, logits: torch.Tensor, grid_size: tuple[int, int] | None = None):
        """Per-query heads. Returns ``(geometry, class_logits)``: mask logits
        ``(B, Q, h, w)`` for the instance family, box preds ``(B, Q, 4)`` for detect.

        ``grid_size`` is the actual patch grid of the current input; it defaults to
        the trained ``self.grid_size`` so legacy callers keep working.
        """
        grid_size = grid_size or self.grid_size
        num_queries = self.config.num_queries
        query_tokens = logits[:, :num_queries, :]
        class_logits = self.class_predictor(query_tokens)

        if self.family == "detect":
            return self.box_head(query_tokens), class_logits

        num_prefix = self.embeddings.num_prefix_tokens
        prefix_tokens = logits[:, num_queries + num_prefix :, :]
        prefix_tokens = prefix_tokens.transpose(1, 2)
        prefix_tokens = prefix_tokens.reshape(prefix_tokens.shape[0], -1, *grid_size)

        query_tokens = self.mask_head(query_tokens)
        prefix_tokens = self.upscale_block(prefix_tokens)
        mask_logits = torch.einsum("bqc, bchw -> bqhw", query_tokens, prefix_tokens)
        return mask_logits, class_logits

    @staticmethod
    def _disable_attention_mask(attn_mask, prob, num_query_tokens, encoder_start_tokens, device):
        if prob < 1:
            random_queries = torch.rand(attn_mask.shape[0], num_query_tokens, device=device) > prob
            attn_mask[:, :num_query_tokens, encoder_start_tokens:][random_queries] = 1
        return attn_mask

    def _build_attention_mask(self, hidden_states, masks_queries_logits, prob, grid_size=None):
        grid_size = grid_size or self.grid_size
        num_query_tokens = self.config.num_queries
        encoder_start_tokens = num_query_tokens + self.embeddings.num_prefix_tokens
        attention_mask = torch.ones(
            hidden_states.shape[0], hidden_states.shape[1], hidden_states.shape[1],
            device=hidden_states.device, dtype=torch.bool,
        )
        interpolated_logits = F.interpolate(masks_queries_logits, size=grid_size, mode="bilinear")
        interpolated_logits = interpolated_logits.view(
            interpolated_logits.size(0), interpolated_logits.size(1), -1
        )
        attention_mask[:, :num_query_tokens, encoder_start_tokens:] = interpolated_logits > 0
        attention_mask = self._disable_attention_mask(
            attention_mask, prob, num_query_tokens, encoder_start_tokens, attention_mask.device
        )
        attention_mask = attention_mask[:, None, ...].expand(-1, self.config.num_attention_heads, -1, -1)
        return attention_mask.float().masked_fill(~attention_mask, -1e9)

    def forward(
        self,
        pixel_values: torch.Tensor,
        mask_labels: list[torch.Tensor] | None = None,
        class_labels: list[torch.Tensor] | None = None,
        box_labels: list[torch.Tensor] | None = None,
    ) -> EoMTOutput:
        # Per-layer geometry (mask logits or box preds) for deep supervision.
        geom_per_layer, class_per_layer = (), ()
        attention_mask = None
        is_detect = self.family == "detect"
        # GT geometry for this family: boxes for detect, masks for instance.
        geom_labels = box_labels if is_detect else mask_labels

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Actual patch grid for this input (may differ from the trained grid when
        # running at a non-native image size); threaded into the prediction heads.
        ps = self.config.patch_size
        grid = (pixel_values.shape[-2] // ps, pixel_values.shape[-1] // ps)

        hidden_states = self.embeddings(pixel_values)

        for idx, layer_module in enumerate(self.layers):
            if idx == self.num_hidden_layers - self.config.num_blocks:
                query = (
                    self.query.weight[None, :, :]
                    .expand(hidden_states.shape[0], -1, -1)
                    .to(hidden_states.device)
                )
                hidden_states = torch.cat((query, hidden_states), dim=1)

            block_idx = idx - self.num_hidden_layers + self.config.num_blocks
            in_query_blocks = idx >= self.num_hidden_layers - self.config.num_blocks
            if is_detect:
                # Deep supervision: predict per query-block during training. No masked
                # attention (no masks to focus on) — queries do full attention.
                if in_query_blocks and self.training:
                    norm_hidden_states = self.layernorm(hidden_states)
                    geom, cls = self.predict(norm_hidden_states, grid)
                    geom_per_layer += (geom,)
                    class_per_layer += (cls,)
            elif in_query_blocks and (self.training or self.attn_mask_probs[block_idx] > 0):
                norm_hidden_states = self.layernorm(hidden_states)
                masks_queries_logits, class_queries_logits = self.predict(norm_hidden_states, grid)
                geom_per_layer += (masks_queries_logits,)
                class_per_layer += (class_queries_logits,)
                attention_mask = self._build_attention_mask(
                    hidden_states, masks_queries_logits, self.attn_mask_probs[block_idx], grid
                )

            hidden_states = layer_module(hidden_states, attention_mask)

        sequence_output = self.layernorm(hidden_states)
        geom_final, class_queries_logits = self.predict(sequence_output, grid)
        geom_per_layer += (geom_final,)
        class_per_layer += (class_queries_logits,)

        loss = None
        if geom_labels is not None and class_labels is not None:
            loss = 0.0
            for g, c in zip(geom_per_layer, class_per_layer):
                loss = loss + self.get_loss(self.get_loss_dict(g, c, geom_labels, class_labels))

        return EoMTOutput(
            loss=loss,
            masks_queries_logits=None if is_detect else geom_final,
            pred_boxes=geom_final if is_detect else None,
            class_queries_logits=class_queries_logits,
            last_hidden_state=sequence_output,
        )


# ---------------------------------------------------------------------------
# Trainable model wrapper + secondary heads
# ---------------------------------------------------------------------------


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
    loss_weights: dict | None = None,
    num_upscale_blocks: int | None = None,
) -> "EoMTModel":
    """Build an :class:`EoMTModel`.

    ``family`` selects the prediction head: ``"instance"`` (mask + class, the
    default) or ``"detect"`` (DETR-style box + class). ``aux_heads`` adds one
    secondary per-instance classifier per spec; ``aux_head_arch`` sets their shared
    network shape (see ``_build_aux_head``). ``loss_weights`` overrides the
    criterion weights (mask/dice for instance, L1/GIoU for detect — see
    ``normalize_loss_weights``); ``num_upscale_blocks`` overrides the mask-head
    upsampling depth (instance only; ``None`` = size preset default).
    """
    if family not in FAMILIES:
        raise ValueError(f"family={family!r} is not one of {FAMILIES}.")
    return EoMTModel(
        size=size,
        nc=nc,
        image_size=imgsz,
        patch_size=PATCH_SIZE,
        names=names,
        family=family,
        aux_heads=aux_heads,
        aux_head_arch=aux_head_arch,
        loss_weights=loss_weights,
        num_upscale_blocks=num_upscale_blocks,
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
        family: str = "instance",
        aux_heads: list[AuxHeadSpec] | None = None,
        aux_head_arch: dict | None = None,
        loss_weights: dict | None = None,
        num_upscale_blocks: int | None = None,
    ):
        super().__init__()
        self.size = size
        self.nc = nc
        self.image_size = image_size
        self.patch_size = patch_size
        self.names = names
        self.family = family
        # Pixel normalization (ImageNet by default), persisted in the checkpoint so
        # eval/predict reproduce preprocessing from the file alone.
        self.pixel_mean: tuple[float, float, float] = tuple(float(x) for x in IMAGENET_MEAN)
        self.pixel_std: tuple[float, float, float] = tuple(float(x) for x in IMAGENET_STD)
        # Criterion weights and mask-head depth, persisted in the checkpoint so a
        # tuned objective / head shape is rebuilt identically on reload. The valid
        # keys depend on the family (mask/dice vs box L1/GIoU).
        self.loss_weights = normalize_loss_weights(loss_weights, family=family)
        # ``build_eomt_config`` only knows the mask/dice + shared keys; pass those
        # through and keep mask/dice at their defaults for detect (unused there).
        cfg_weights = {
            k: v for k, v in self.loss_weights.items()
            if k in ("no_object_weight", "class_weight", "mask_weight", "dice_weight",
                     "train_num_points", "oversample_ratio", "importance_sample_ratio")
        }
        config = build_eomt_config(
            size,
            nc=nc,
            image_size=image_size,
            patch_size=patch_size,
            names=names,
            num_upscale_blocks=num_upscale_blocks,
            **cfg_weights,
        )
        # Box-loss weights have no EomtConfig field; stash them on the config so
        # ``EoMTEncoder`` (which only receives the config) can read them.
        if family == "detect":
            config.l1_weight = float(self.loss_weights["l1_weight"])
            config.giou_weight = float(self.loss_weights["giou_weight"])
        self.config = config
        # Effective upscale depth (resolved from the preset when not overridden).
        self.num_upscale_blocks = int(config.num_upscale_blocks)
        self.eomt = EoMTEncoder(config, family=family)

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
        box_labels: list[torch.Tensor] | None = None,
    ):
        self._query_embed = None
        out = self.eomt(
            pixel_values=pixel_values,
            mask_labels=mask_labels,
            class_labels=class_labels,
            box_labels=box_labels,
        )
        geom_labels = box_labels if self.family == "detect" else mask_labels
        training = geom_labels is not None and class_labels is not None
        if self.family == "detect":
            result = {
                "pred_boxes": out.pred_boxes,
                "class_queries_logits": out.class_queries_logits,
            }
        else:
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


def _interp_pos_grid(
    weight: torch.Tensor, src_hw: tuple[int, int], dst_hw: tuple[int, int]
) -> torch.Tensor:
    """Bicubically resize a flat ``(src_h*src_w, dim)`` patch pos-embed table to ``dst_hw``.

    Returns a ``(dst_h*dst_w, dim)`` table; a no-op (returns ``weight`` unchanged)
    when the grids already match. Used both at load time (DINOv2 -> EoMT) and at
    inference time (trained grid -> the actual input grid) so the model can run at
    any image size without a fixed-resolution pos-embed.
    """
    if src_hw == dst_hw:
        return weight
    sh, sw = src_hw
    dh, dw = dst_hw
    dim = weight.shape[-1]
    grid = weight.reshape(1, sh, sw, dim).permute(0, 3, 1, 2).float()
    grid = F.interpolate(grid, size=(dh, dw), mode="bicubic", align_corners=False)
    return grid.permute(0, 2, 3, 1).reshape(dh * dw, dim).to(weight.dtype)


def _resize_pos_embed(dinov2_pos: torch.Tensor, target_weight: torch.Tensor) -> torch.Tensor:
    """Map DINOv2 position embeddings onto EoMT's patches-only pos-embed table.

    ``dinov2_pos`` is ``(1, 1 + Hd*Wd, dim)`` (cls + patches). EoMT's
    ``position_embeddings.weight`` is patches-only ``(He*We, dim)`` — the cls /
    register / query tokens carry no learned position there. The DINOv2 cls
    position is dropped; the patch grid is bicubically interpolated when the grids
    differ (e.g. DINOv2 pretrained at 518 -> our 644).
    """
    import math

    src_patches = dinov2_pos[0][1:]  # drop cls position
    hs = int(math.isqrt(src_patches.shape[0]))
    ht = int(math.isqrt(target_weight.shape[0]))
    return _interp_pos_grid(src_patches, (hs, hs), (ht, ht))


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

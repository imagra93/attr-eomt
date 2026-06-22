"""Pure-PyTorch EoMT forward, checkpoint-compatible with the HF implementation.

``NativeEoMT`` reimplements ``transformers.EomtForUniversalSegmentation``'s forward
path — a DINOv2-with-registers ViT whose last ``num_blocks`` layers carry learnable
queries with masked attention, plus the mask/class prediction heads — using only
``torch`` (no transformers *model code*). Submodule attribute names mirror the HF
model **exactly**, so a checkpoint trained with the HF model loads with zero
missing/unexpected keys and produces numerically identical outputs (verified on
CPU/fp32 in ``tests/test_native_parity.py``).

The attention uses ``F.scaled_dot_product_attention`` with ``scale=head_dim**-0.5``
— the same op HF dispatches to (``_attn_implementation='sdpa'``) — and the masked
layers add the identical ``-1e9`` additive bias, so parity holds.

Only the config (``transformers.EomtConfig``) and the DINOv2 backbone loader are
still sourced from transformers; the model code is fully local.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from .loss import NativeEomtLoss


@dataclass
class NativeEoMTOutput:
    """Minimal output object; attribute access mirrors the HF dataclass."""

    loss: torch.Tensor | None = None
    class_queries_logits: torch.Tensor | None = None
    masks_queries_logits: torch.Tensor | None = None
    last_hidden_state: torch.Tensor | None = None


def _act(name: str):
    """Resolve an activation by config name (only 'gelu' is used by the presets)."""
    if name == "gelu":
        return nn.GELU()  # exact erf gelu, matches ACT2FN['gelu']
    raise NotImplementedError(f"hidden_act={name!r} not supported by NativeEoMT.")


class NativePatchEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        ps = config.patch_size
        self.num_channels = config.num_channels
        self.num_patches = (config.image_size // ps) * (config.image_size // ps)
        self.projection = nn.Conv2d(config.num_channels, config.hidden_size, kernel_size=ps, stride=ps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.projection(pixel_values).flatten(2).transpose(1, 2)


class NativeEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.register_tokens = nn.Parameter(torch.zeros(1, config.num_register_tokens, config.hidden_size))
        self.patch_embeddings = NativePatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.num_prefix_tokens = 1 + config.num_register_tokens
        self.position_embeddings = nn.Embedding(num_patches, config.hidden_size)
        self.register_buffer("position_ids", torch.arange(num_patches).expand((1, -1)), persistent=False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embeddings.projection.weight.dtype
        embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        register_tokens = self.register_tokens.expand(batch_size, -1, -1)
        embeddings = embeddings + self.position_embeddings(self.position_ids)
        embeddings = torch.cat([cls_tokens, register_tokens, embeddings], dim=1)
        return self.dropout(embeddings)


class NativeAttention(nn.Module):
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


class NativeLayerScale(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lambda1 = nn.Parameter(config.layerscale_value * torch.ones(config.hidden_size))

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return hidden_state * self.lambda1


class NativeMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_features = int(config.hidden_size * config.mlp_ratio)
        self.fc1 = nn.Linear(config.hidden_size, hidden_features, bias=True)
        self.activation = _act(config.hidden_act)
        self.fc2 = nn.Linear(hidden_features, config.hidden_size, bias=True)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(hidden_state)))


class NativeLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.use_swiglu_ffn:
            raise NotImplementedError("use_swiglu_ffn=True is not supported by NativeEoMT.")
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = NativeAttention(config)
        self.layer_scale1 = NativeLayerScale(config)
        self.drop_path = nn.Identity()  # drop_path_rate is 0 for all presets
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = NativeMLP(config)
        self.layer_scale2 = NativeLayerScale(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        attn = self.attention(self.norm1(hidden_states), attention_mask)
        attn = self.layer_scale1(attn)
        hidden_states = self.drop_path(attn) + hidden_states
        out = self.norm2(hidden_states)
        out = self.mlp(out)
        out = self.layer_scale2(out)
        return self.drop_path(out) + hidden_states


class NativeLayerNorm2d(nn.LayerNorm):
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = hidden_state.permute(0, 2, 3, 1)
        hidden_state = F.layer_norm(hidden_state, self.normalized_shape, self.weight, self.bias, self.eps)
        return hidden_state.permute(0, 3, 1, 2)


class NativeScaleLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        h = config.hidden_size
        self.conv1 = nn.ConvTranspose2d(h, h, kernel_size=2, stride=2)
        self.activation = _act(config.hidden_act)
        self.conv2 = nn.Conv2d(h, h, kernel_size=3, padding=1, groups=h, bias=False)
        self.layernorm2d = NativeLayerNorm2d(h)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.conv2(hidden_states)
        return self.layernorm2d(hidden_states)


class NativeScaleBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block = nn.ModuleList([NativeScaleLayer(config) for _ in range(config.num_upscale_blocks)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for block in self.block:
            hidden_states = block(hidden_states)
        return hidden_states


class NativeMaskHead(nn.Module):
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


class NativeEoMT(nn.Module):
    """Pure-PyTorch ``EomtForUniversalSegmentation`` (forward + loss)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_hidden_layers = config.num_hidden_layers
        self.embeddings = NativeEmbeddings(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.query = nn.Embedding(config.num_queries, config.hidden_size)
        self.layers = nn.ModuleList([NativeLayer(config) for _ in range(config.num_hidden_layers)])
        self.upscale_block = NativeScaleBlock(config)
        self.mask_head = NativeMaskHead(config)
        self.class_predictor = nn.Linear(config.hidden_size, config.num_labels + 1)

        self.grid_size = (config.image_size // config.patch_size, config.image_size // config.patch_size)
        self.weight_dict: dict[str, float] = {
            "loss_cross_entropy": config.class_weight,
            "loss_mask": config.mask_weight,
            "loss_dice": config.dice_weight,
        }
        self.criterion = NativeEomtLoss(config=config, weight_dict=self.weight_dict)
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

    def predict(self, logits: torch.Tensor):
        num_queries = self.config.num_queries
        num_prefix = self.embeddings.num_prefix_tokens
        query_tokens = logits[:, :num_queries, :]
        class_logits = self.class_predictor(query_tokens)

        prefix_tokens = logits[:, num_queries + num_prefix :, :]
        prefix_tokens = prefix_tokens.transpose(1, 2)
        prefix_tokens = prefix_tokens.reshape(prefix_tokens.shape[0], -1, *self.grid_size)

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

    def _build_attention_mask(self, hidden_states, masks_queries_logits, prob):
        num_query_tokens = self.config.num_queries
        encoder_start_tokens = num_query_tokens + self.embeddings.num_prefix_tokens
        attention_mask = torch.ones(
            hidden_states.shape[0], hidden_states.shape[1], hidden_states.shape[1],
            device=hidden_states.device, dtype=torch.bool,
        )
        interpolated_logits = F.interpolate(masks_queries_logits, size=self.grid_size, mode="bilinear")
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
    ) -> NativeEoMTOutput:
        masks_per_layer, class_per_layer = (), ()
        attention_mask = None

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

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
            if idx >= self.num_hidden_layers - self.config.num_blocks and (
                self.training or self.attn_mask_probs[block_idx] > 0
            ):
                norm_hidden_states = self.layernorm(hidden_states)
                masks_queries_logits, class_queries_logits = self.predict(norm_hidden_states)
                masks_per_layer += (masks_queries_logits,)
                class_per_layer += (class_queries_logits,)
                attention_mask = self._build_attention_mask(
                    hidden_states, masks_queries_logits, self.attn_mask_probs[block_idx]
                )

            hidden_states = layer_module(hidden_states, attention_mask)

        sequence_output = self.layernorm(hidden_states)
        masks_queries_logits, class_queries_logits = self.predict(sequence_output)
        masks_per_layer += (masks_queries_logits,)
        class_per_layer += (class_queries_logits,)

        loss = None
        if mask_labels is not None and class_labels is not None:
            loss = 0.0
            for m, c in zip(masks_per_layer, class_per_layer):
                loss = loss + self.get_loss(self.get_loss_dict(m, c, mask_labels, class_labels))

        return NativeEoMTOutput(
            loss=loss,
            masks_queries_logits=masks_queries_logits,
            class_queries_logits=class_queries_logits,
            last_hidden_state=sequence_output,
        )

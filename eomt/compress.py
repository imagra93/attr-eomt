"""Int8 weight-only compression for EoMT models via PyTorch ``torchao``.

Quantizes the weights of the ViT transformer blocks — the bulk of the parameters —
to int8, while keeping the precision-sensitive prediction heads (class/mask/box),
the patch embedding and all ``LayerNorm``s in full precision. This is **data-free**:
int8 weight-only quant just re-rounds the existing weights, so there is no
calibration step and accuracy is preserved in practice (≈3.4× smaller state dict on
ViT-L with no measurable mAP loss).

The recipe applied is recorded in the checkpoint so :func:`eomt.serialization.load_model`
can re-create the quantized layout before loading the (subclassed) weight tensors.

``torchao`` is a regular dependency, but it is imported lazily here so an unusual
environment that lacks it still surfaces a clear, actionable error.
"""

from __future__ import annotations

from typing import Any, Callable

import torch.nn as nn

#: FQN prefix of the ViT transformer blocks inside :class:`~eomt.model.EoMTModel`
#: (``eomt`` is the encoder, ``layers`` the per-block ``ModuleList``). Everything
#: else — patch embedding, ``class_predictor``, ``mask_head``, ``upscale_block``,
#: ``box_head``, ``aux_heads`` and all ``LayerNorm``s — is left in full precision.
_BLOCKS_PREFIX = "eomt.layers."


def _require_torchao() -> None:
    """Import ``torchao`` lazily, with an actionable error if it is missing."""
    try:
        import torchao  # noqa: F401
    except ImportError as e:  # pragma: no cover - depends on optional install
        raise ImportError(
            "compression needs 'torchao' (pip install torchao>=0.7.0)."
        ) from e


def build_filter_fn(model: nn.Module) -> Callable[[nn.Module, str], bool]:
    """Return a ``torchao`` ``filter_fn`` selecting the transformer-block ``nn.Linear``s.

    Targets the qkv/proj/MLP linears inside ``eomt.layers.*`` (≈90% of params) and
    skips the prediction heads, patch embedding and norms — quantizing those tends
    to cost mask/class quality for little extra saving.
    """

    def filter_fn(module: nn.Module, fqn: str) -> bool:
        return isinstance(module, nn.Linear) and fqn.startswith(_BLOCKS_PREFIX)

    return filter_fn


def _int8_config():
    """Build a ``torchao`` int8 weight-only quant config, tolerant of API versions."""
    try:
        from torchao.quantization import Int8WeightOnlyConfig

        return Int8WeightOnlyConfig()
    except ImportError:  # older torchao: function-style factory
        from torchao.quantization import int8_weight_only

        return int8_weight_only()


def compress_model(model: nn.Module, recipe: str = "int8") -> nn.Module:
    """Quantize ``model``'s transformer-block weights to int8 in place and return it.

    ``recipe`` must be ``"int8"`` (the only supported recipe). ``torchao`` swaps the
    targeted linear weights for int8 tensor subclasses; the model is mutated in place
    and also returned for convenience.
    """
    if recipe != "int8":
        raise ValueError(f"unsupported recipe {recipe!r}; only 'int8' is supported.")
    _require_torchao()
    from torchao.quantization import quantize_

    quantize_(model, _int8_config(), filter_fn=build_filter_fn(model))
    return model


def compression_meta(recipe: str = "int8") -> dict[str, Any]:
    """Build the checkpoint metadata describing how the model was compressed.

    Stored under the ``compression`` key so :func:`eomt.serialization.load_model`
    can re-apply the same recipe before loading the quantized weights.
    """
    meta: dict[str, Any] = {"library": "torchao", "recipe": recipe}
    try:
        from importlib.metadata import version

        meta["library_version"] = version("torchao")
    except Exception:  # noqa: BLE001 - version is informational only
        pass
    return meta

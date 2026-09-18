"""Identify what a weight file is by the namespace of its tensor names, without
needing diffusers/transformers to load it. Works off just the safetensors
header (a handful of KB, readable via a range request) so it applies equally
to repos that never declare a `model_index.json` at all.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TensorSignature:
    role: str
    architecture: str | None
    required_substrings: tuple[str, ...]
    excluded_substrings: tuple[str, ...] = ()
    confidence: str = "high"


# Checked in order; more specific/unambiguous signatures come first so a
# generic one can't shadow a more precise match.
SIGNATURES: tuple[TensorSignature, ...] = (
    TensorSignature("transformer", "FluxTransformer2DModel", ("double_blocks.", "single_blocks.")),
    TensorSignature("transformer", "SD3Transformer2DModel", ("joint_blocks.",)),
    TensorSignature(
        "vae", "AutoencoderKL", ("encoder.down_blocks.", "decoder.up_blocks.", "quant_conv")
    ),
    TensorSignature("vae", None, ("first_stage_model.encoder", "first_stage_model.decoder")),
    TensorSignature("unet", "UNet2DConditionModel", ("down_blocks.", "up_blocks.", "time_embedding.")),
    TensorSignature("unet", None, ("input_blocks.", "output_blocks.", "time_embed.")),
    TensorSignature(
        "text_encoder",
        "T5EncoderModel",
        ("encoder.block.", "DenseReluDense", "shared.weight"),
        excluded_substrings=("decoder.block.",),
    ),
    TensorSignature(
        "text_encoder", "CLIPTextModelWithProjection", ("transformer.resblocks.", "text_projection")
    ),
    TensorSignature(
        "text_encoder",
        "CLIPTextModel",
        ("text_model.encoder.layers.", "text_model.embeddings.token_embedding"),
    ),
    TensorSignature("audio_codec", None, ("quantizer.vq", "codebook"), confidence="low"),
)


def identify_by_tensor_names(tensor_keys: list[str]) -> tuple[str, str | None, str] | None:
    """Returns (role, architecture, confidence) for the first matching signature, or None."""
    if not tensor_keys:
        return None
    blob = " ".join(tensor_keys)
    for signature in SIGNATURES:
        if any(excluded in blob for excluded in signature.excluded_substrings):
            continue
        if all(required in blob for required in signature.required_substrings):
            return signature.role, signature.architecture, signature.confidence
    return None

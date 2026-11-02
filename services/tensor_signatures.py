"""Identify what a weight file is by the namespace of its tensor names, without
needing diffusers/transformers to load it. Works off just the safetensors
header (a handful of KB, readable via a range request) so it applies equally
to repos that never declare a `model_index.json` at all.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any


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
    # llama.cpp names a T5 encoder's layers `enc.blk.`, which the LLM GGUF
    # signature below would otherwise claim.
    TensorSignature(
        "text_encoder",
        "T5EncoderModel",
        ("enc.blk.", "token_embd."),
        excluded_substrings=("dec.blk.",),
    ),
    # Decoder-only LLMs (as models in their own right, or as a pipeline's text
    # encoder): llama.cpp's GGUF naming, then transformers'. Last, since the
    # encoder signatures above are stricter.
    TensorSignature("language_model", None, ("token_embd.", "blk.")),
    TensorSignature("language_model", None, ("embed_tokens.", ".layers.")),
)

# Dotted segments multimodal checkpoints keep their vision tower and projector
# under: Qwen-VL `visual.`, LLaVA/Gemma3/Mistral3 `vision_tower.` +
# `multi_modal_projector.`, Idefics/Mllama/CLIP `vision_model.`, Pixtral
# `vision_encoder.`, Phi-3-vision `vision_embed_tokens.`, MiniCPM-V `vpm.`.
VISION_SEGMENTS = frozenset(
    {
        "visual",
        "vision_tower",
        "vision_model",
        "vision_encoder",
        "vision_embed_tokens",
        "multi_modal_projector",
        "mm_projector",
        "vpm",
    }
)
# llama.cpp mmproj files: `v.blk.N.`, `v.patch_embd`, ... plus `mm.N.` for the projector.
GGUF_VISION_PREFIXES = ("v.", "mm.")
# A real vision tower is hundreds of tensors; a few stray matches aren't one.
MIN_VISION_TENSORS = 8
VISION_ONLY_SHARE = 0.9


def _is_vision_key(key: str) -> bool:
    return key.startswith(GGUF_VISION_PREFIXES) or any(
        segment in VISION_SEGMENTS for segment in key.split(".")
    )


def has_vision_tensors(tensor_keys: list[str]) -> bool:
    """Whether a file (or component) carries vision encoder weights - e.g. a
    VL model used as a text encoder, or a multimodal LLM checkpoint."""
    return sum(1 for key in _weight_keys(tensor_keys) if _is_vision_key(key)) >= MIN_VISION_TENSORS


def _is_vision_only(tensor_keys: list[str]) -> bool:
    keys = _weight_keys(tensor_keys)
    hits = sum(1 for key in keys if _is_vision_key(key))
    return hits >= MIN_VISION_TENSORS and hits / len(keys) >= VISION_ONLY_SHARE


# GGUF architectures that carry a tokenizer but aren't decoder LLMs; their
# tensor names still need reading to say what they are.
GGUF_ENCODER_ARCHITECTURES = frozenset({"t5", "t5encoder", "bert", "nomic-bert", "jina-bert-v2", "modern-bert"})


def identify_by_gguf_metadata(metadata: dict[str, Any]) -> tuple[str, bool] | None:
    """(role, has vision weights) from a GGUF's leading metadata alone, when
    that's decisive - which spares reading an LLM's multi-MB header (its
    tokenizer comes before the tensor table). llama.cpp always splits a VL
    model's vision tower into a separate `clip` mmproj, so an LLM GGUF never
    has vision weights of its own."""
    architecture = metadata.get("general.architecture")
    if architecture == "clip":
        # Omni models' projectors may be audio-only.
        if metadata.get("clip.has_vision_encoder") is False:
            return None
        return "vision_encoder", True
    if (
        isinstance(architecture, str)
        and architecture not in GGUF_ENCODER_ARCHITECTURES
        and "tokenizer.ggml.model" in metadata
    ):
        return "language_model", False
    return None


def identify_by_tensor_names(tensor_keys: list[str]) -> tuple[str, str | None, str] | None:
    """Returns (role, architecture, confidence) for the first matching signature, or None."""
    if not tensor_keys:
        return None
    # A standalone vision encoder (an LLM's mmproj, a CLIP/SigLIP vision
    # tower) - checked first, since an mmproj's `v.blk.` also contains `blk.`.
    if _is_vision_only(tensor_keys):
        return "vision_encoder", None, "high"
    blob = " ".join(tensor_keys)
    for signature in SIGNATURES:
        if any(excluded in blob for excluded in signature.excluded_substrings):
            continue
        if all(required in blob for required in signature.required_substrings):
            return signature.role, signature.architecture, signature.confidence
    return None


# Side tensors that quantizers add next to (or instead of the dtype of) a real
# weight: comfy fp8/nvfp4 scales, MLX group scales/biases, bitsandbytes state.
# They say nothing about which model the file is, so they're ignored when
# comparing against a reference.
QUANT_AUX_SUFFIXES = (
    ".weight_scale",
    ".weight_scale_2",
    ".input_scale",
    ".scale_weight",
    ".scale_input",
    ".comfy_quant",
    ".comfy_attention.config",
    ".scales",
    ".biases",
    ".absmax",
    ".quant_map",
    ".SCB",
)
QUANT_AUX_MARKERS = (".quant_state.",)

LORA_MARKERS = (".lora_A.", ".lora_B.", ".lora_down.", ".lora_up.", ".lora.down.", ".lora.up.", "lora_unet_", "lora_te")

# How many leading dotted segments may be dropped when aligning two key sets.
# Covers container prefixes (`model.diffusion_model.`, `first_stage_model.`)
# and wrapper renames like transformers' `model.layers.` ->
# `model.language_model.layers.`, while always keeping at least two segments
# so bare tails like `weight` can't match.
MAX_PREFIX_SEGMENTS_DROPPED = 2

HIGH_CONFIDENCE_COVERAGE = 0.9
MIN_CANDIDATE_COVERAGE = 0.8
MIN_REFERENCE_COVERAGE = 0.25
MIN_SHAPE_OVERLAP = 0.6
MAX_TENSOR_COUNT_DRIFT = 0.05


@dataclass(frozen=True)
class ReferenceMatch:
    # Share of the candidate's tensors that exist in the reference (is this
    # file *part of* that component?) and share of the reference's tensors the
    # candidate has (is it the *whole* component, or e.g. just one shard?).
    candidate_coverage: float
    reference_coverage: float
    confidence: str
    method: str  # "names" | "shapes" | "parameters"

    @property
    def score(self) -> float:
        return (self.confidence == "high") * 10 + self.candidate_coverage + self.reference_coverage


def is_lora(tensor_keys: list[str]) -> bool:
    if not tensor_keys:
        return False
    hits = sum(1 for key in tensor_keys if any(marker in key for marker in LORA_MARKERS))
    return hits / len(tensor_keys) >= 0.5


def _weight_keys(tensor_keys: list[str]) -> list[str]:
    return [
        key
        for key in tensor_keys
        if not key.endswith(QUANT_AUX_SUFFIXES) and not any(marker in key for marker in QUANT_AUX_MARKERS)
    ]


def _suffix_variants(key: str) -> list[str]:
    segments = key.split(".")
    max_drop = min(MAX_PREFIX_SEGMENTS_DROPPED, len(segments) - 2)
    return [".".join(segments[drop:]) for drop in range(max(max_drop, 0) + 1)]


def _coverage(keys: list[str], other_variants: set[str]) -> float:
    if not keys:
        return 0.0
    hits = sum(1 for key in keys if any(variant in other_variants for variant in _suffix_variants(key)))
    return hits / len(keys)


def _packed_stems(tensor_keys: list[str]) -> set[str]:
    """Names a quantizer's side tensors hang off: the module (comfy
    `x.weight_scale`, MLX `x.scales`) or the weight itself (bitsandbytes
    `x.weight.absmax`). The weights under them may be stored packed."""
    stems: set[str] = set()
    for key in tensor_keys:
        suffix = next((suffix for suffix in QUANT_AUX_SUFFIXES if key.endswith(suffix)), None)
        if suffix is not None:
            stems.add(key[: -len(suffix)])
            continue
        for marker in QUANT_AUX_MARKERS:
            if marker in key:
                stems.add(key.split(marker, 1)[0])
    return stems


def shapes_agree(candidate: dict[str, tuple[int, ...]], peer: dict[str, tuple[int, ...]]) -> bool:
    """Whether a file that shares a peer's tensor names also has its shapes,
    apart from weights a quantizer packed (which have side tensors). A
    quantized sibling passes; a same-family variant with a resized layer - a
    FLUX.1 Fill/Canny checkpoint next to dev, named identically - doesn't."""
    peer_shapes: dict[str, tuple[int, ...]] = {}
    for key in _weight_keys(list(peer)):
        for variant in _suffix_variants(key):
            peer_shapes.setdefault(variant, peer[key])
    candidate_packed = _packed_stems(list(candidate))
    peer_packed = _packed_stems(list(peer))
    for key in _weight_keys(list(candidate)):
        module = key.rsplit(".", 1)[0]
        if {key, module} & (candidate_packed | peer_packed):
            continue
        shape = next((peer_shapes[variant] for variant in _suffix_variants(key) if variant in peer_shapes), None)
        if shape is not None and shape != candidate[key]:
            return False
    return True


def _matrix_parameter_count(shapes: dict[str, tuple[int, ...]]) -> int:
    """Parameters held in matrices/conv kernels. Vectors (norm scales, biases,
    pad tokens, however they're reshaped) are left out: converters routinely
    add or drop one - ComfyUI's Lumina 2 keeps a `norm_final.weight` diffusers
    discards - while the weight matrices themselves never change."""
    return sum(
        math.prod(shape) for shape in shapes.values() if sum(1 for dim in shape if dim > 1) >= 2
    )


def _shape_overlap(candidate: dict[str, tuple[int, ...]], reference: dict[str, tuple[int, ...]]) -> float:
    # Dims are sorted so a transposed/reshaped-but-equivalent tensor still
    # counts; this is a coarse structural signal, only used when names fail.
    candidate_shapes = Counter(tuple(sorted(shape)) for shape in candidate.values())
    reference_shapes = Counter(tuple(sorted(shape)) for shape in reference.values())
    shared = sum((candidate_shapes & reference_shapes).values())
    return shared / max(len(candidate), len(reference))


def match_reference(
    candidate: dict[str, tuple[int, ...]],
    reference_keys: list[str],
    reference_shapes: dict[str, tuple[int, ...]] | None = None,
) -> ReferenceMatch | None:
    """Does a weight file (`candidate`: tensor name -> shape) hold the same
    model as a known component (`reference_keys`, plus shapes when known)?

    Tensor names are compared first, tolerant of container prefixes and
    quantization side tensors. If names don't line up - e.g. a ComfyUI-format
    VAE vs the diffusers one, which share weights but not a naming scheme -
    falls back to matching tensor count and the multiset of shapes, at low
    confidence. Failing that, an identical parameter count across the weight
    matrices still identifies the component: BFL/ComfyUI-native checkpoints
    and the GGUFs converted from them fuse q/k/v and rename every layer, which
    changes the tensor count by a third, but no conversion adds or drops a
    weight matrix (GGUF records the unquantized dims). Exact equality keeps
    same-family variants with a resized layer (FLUX.1 Fill/Canny vs dev) from
    matching.
    Quantized files whose shapes change (MLX, nvfp4 or SVDQuant packing) only
    match by name."""
    candidate_keys = _weight_keys(list(candidate))
    reference_keys = _weight_keys(reference_keys)
    if not candidate_keys or not reference_keys:
        return None

    reference_variants = {variant for key in reference_keys for variant in _suffix_variants(key)}
    candidate_variants = {variant for key in candidate_keys for variant in _suffix_variants(key)}
    candidate_coverage = _coverage(candidate_keys, reference_variants)
    reference_coverage = _coverage(reference_keys, candidate_variants)

    if candidate_coverage >= MIN_CANDIDATE_COVERAGE and reference_coverage >= MIN_REFERENCE_COVERAGE:
        confident = min(candidate_coverage, reference_coverage) >= HIGH_CONFIDENCE_COVERAGE
        return ReferenceMatch(
            candidate_coverage, reference_coverage, "high" if confident else "low", "names"
        )

    if reference_shapes:
        candidate_weights = {key: candidate[key] for key in candidate_keys}
        reference_weights = {key: reference_shapes[key] for key in reference_keys if key in reference_shapes}
        drift = abs(len(candidate_weights) - len(reference_weights)) / len(reference_weights)
        if drift <= MAX_TENSOR_COUNT_DRIFT:
            overlap = _shape_overlap(candidate_weights, reference_weights)
            if overlap >= MIN_SHAPE_OVERLAP:
                return ReferenceMatch(overlap, overlap, "low", "shapes")
        reference_parameters = _matrix_parameter_count(reference_weights)
        if reference_parameters and _matrix_parameter_count(candidate_weights) == reference_parameters:
            return ReferenceMatch(1.0, 1.0, "high", "parameters")

    return None

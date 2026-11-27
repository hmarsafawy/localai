"""How much GPU memory llama-server needs for a model, worked out the way
llama.cpp itself allocates it, so whether a model fits is known before it's
downloaded or launched:

    total = weights + KV cache + recurrent state + compute buffer
            + vision encoder + the GPU runtime's own memory

and it fits when `total + FIT_MARGIN` is free. Every term and constant is
checked against what llama-server actually allocated; docs/memory-estimate.md
has the reasoning and the measurements.

Pure functions over a `ModelShape` - nothing here touches the network or the GPU.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from services.model_shape import ModelShape

MiB = 1024**2
_F32 = 4

# The context a model is rated at when nobody asked for one: room for a long
# conversation or a few files. Capped at what the model was trained for.
TARGET_CONTEXT = 16384
# `--fit` never shrinks the context below this (`--fit-ctx`); it moves layers
# to the CPU instead.
MIN_FIT_CONTEXT = 4096
# llama.cpp rounds the context up to a multiple of this, and the sliding-window
# cache too.
CONTEXT_PADDING = 256
# What `--fit` leaves free on the GPU (`--fit-target`'s default).
FIT_MARGIN_MIB = 1024
# llama-server's defaults: four slots sharing one KV cache, with prompts
# processed 512 tokens at a time.
DEFAULT_SLOTS = 4
DEFAULT_UBATCH = 512

# A vision encoder's memory on top of its file. Qwen3.5-9B's 876 MiB encoder
# took 1,580 MiB on ROCm (llama.cpp b12359): a 248 MiB compute buffer, plus
# 456 MiB the GPU runtime keeps after llama.cpp warms the encoder up on a
# worst-case image. On an older Vulkan build, with no such warmup, 1,127 MiB.
# With a context of 64K or more the runtime kept only 92 MiB after the warmup,
# and the encoder took 1,216 MiB, so this errs 364 MiB high there.
VISION_OVERHEAD_BYTES = 768 * MiB

# What the GPU runtime takes in llama-server's process beyond llama.cpp's own
# buffers: the driver context, kernels, and the scratch memory large matrix
# products leave cached. llama.cpp's own breakdown never shows it, but the free
# memory read before launch has to cover it. Measured on an RX 9070 XT, after a
# 6K-token prompt: ROCm 271-335 MiB, Vulkan 81-128 MiB. Backends not measured
# yet take ROCm's figure; CUDA shares its code.
RUNTIME_OVERHEAD_BYTES = {"rocm": 384 * MiB, "vulkan": 192 * MiB, "cpu": 0}
DEFAULT_RUNTIME_OVERHEAD_BYTES = 384 * MiB

# Bytes per block and values per block of each KV cache type llama.cpp takes
# (`--cache-type-k/v`). f16 is its default.
CACHE_TYPES = {
    "f32": (4, 1),
    "f16": (2, 1),
    "bf16": (2, 1),
    "q8_0": (34, 32),
    "q5_1": (24, 32),
    "q5_0": (22, 32),
    "q4_1": (20, 32),
    "q4_0": (18, 32),
    "iq4_nl": (18, 32),
}

# The compute buffer holds f32 copies of the hidden state too: the residual
# stream, its normed copy, a layer's output and the like. How many widths of
# them, fitted to llama-server's own buffers for SmolLM2, LFM3 and Qwen3.5 and
# rounded up: 2.8 alongside the feed-forward, 5 to 6.5 alongside attention
# scores.
_HIDDEN_COPIES_FFN = 4
_HIDDEN_COPIES_ATTENTION = 7
# A gated feed-forward holds its gate, its up projection and their product at once.
_FFN_INTERMEDIATES = 3
# For a model with no feed-forward (Mamba, RWKV), how much wider than the hidden
# state its mixing layer projects: Mamba's in_proj is 4x.
_NO_FFN_WIDTH = 4
# A vocabulary metadata didn't give: the largest in common use (Gemma's).
UNKNOWN_VOCAB = 262144

# The input layer: tables a token's row is looked up in, rather than
# multiplied by, so llama.cpp keeps them in system RAM whatever the GPU layers.
INPUT_TENSORS = frozenset(
    {"token_embd.weight", "position_embd.weight", "token_types.weight", "per_layer_token_embd.weight"}
)
_BLOCK = re.compile(r"blk\.(\d+)\.")


@dataclass(frozen=True)
class RuntimeTraits:
    """How llama-server will run the model, as far as memory goes."""

    backend: str
    # llama-server turns it on wherever the backend has the kernels: measured
    # on with ROCm and Vulkan, and CUDA and Metal have them. Off, attention
    # scores take memory that grows with the context.
    flash_attention: bool = True
    slots: int = DEFAULT_SLOTS
    n_ubatch: int = DEFAULT_UBATCH
    cache_type: str = "f16"

    def __post_init__(self) -> None:
        if self.cache_type not in CACHE_TYPES:
            raise ValueError(f"unknown KV cache type {self.cache_type!r}")

    @property
    def runtime_overhead_bytes(self) -> int:
        return RUNTIME_OVERHEAD_BYTES.get(self.backend, DEFAULT_RUNTIME_OVERHEAD_BYTES)


@dataclass(frozen=True)
class MemoryEstimate:
    """What the GPU holds with the model loaded at one context size, in bytes."""

    context: int
    weights: int
    kv_cache: int
    recurrent_state: int
    compute: int
    vision: int
    runtime: int

    @property
    def total(self) -> int:
        return self.weights + self.kv_cache + self.recurrent_state + self.compute + self.vision + self.runtime

    def fits(self, free_bytes: int) -> bool:
        """Whether llama.cpp would keep all of it on the GPU: `--fit` moves
        layers off until its margin is left free."""
        return self.total + FIT_MARGIN_MIB * MiB <= free_bytes


def padded_context(context: int) -> int:
    """The KV cells llama.cpp allocates for a context: rounded up to 256."""
    return -(-max(context, 1) // CONTEXT_PADDING) * CONTEXT_PADDING


def target_context(shape: ModelShape, requested: int | None = None) -> int:
    """The context to rate a model at: what was asked for, else TARGET_CONTEXT,
    but never more than the model was trained for."""
    wanted = requested or TARGET_CONTEXT
    return min(wanted, shape.context_length) if shape.context_length else wanted


def _cells(shape: ModelShape, context: int, traits: RuntimeTraits) -> tuple[int, int]:
    """KV cells per full-attention layer and per sliding-window layer. A window
    layer keeps the last `window` tokens of every slot, plus the batch being
    added (`llama_kv_cache_iswa`, with the unified cache llama-server uses)."""
    full = padded_context(context)
    if not shape.sliding_window:
        return full, full
    window = padded_context(shape.sliding_window * traits.slots + traits.n_ubatch)
    return full, min(full, window)


def _row_bytes(values: int, cache_type: str) -> int:
    block_bytes, block_values = CACHE_TYPES[cache_type]
    return -(-values // block_values) * block_bytes


def kv_cache_bytes(shape: ModelShape, context: int, traits: RuntimeTraits) -> int:
    """Every attention layer keeps K and V for each cell: `n_head_kv x
    (key_length + value_length)` values per token, 2 bytes each in f16. Layers
    with no KV heads (recurrent ones) keep none."""
    full, window = _cells(shape, context, traits)
    total = 0
    for heads, sliding in zip(shape.n_head_kv, shape.swa_layers):
        if heads:
            per_cell = _row_bytes(heads * shape.key_length, traits.cache_type) + _row_bytes(
                heads * shape.value_length, traits.cache_type
            )
            total += (window if sliding else full) * per_cell
    return total


def recurrent_state_bytes(shape: ModelShape, traits: RuntimeTraits) -> int:
    """A recurrent layer keeps one f32 state per slot, whatever the context."""
    return sum(shape.recurrent_layers) * shape.recurrent_state_size * _F32 * traits.slots


def compute_bytes(shape: ModelShape, context: int, traits: RuntimeTraits) -> int:
    """The scratch space for one forward pass, which llama.cpp reserves for its
    worst case up front: a full batch of `n_ubatch` tokens, with logits for one
    token per slot. Its parts:

    - Attention masks, one per kind of attention layer, `cells x n_ubatch`, in
      f16 with flash attention and f32 without. They grow with the context.
    - The logits: `vocab x slots` f32.
    - Then whichever is bigger, the feed-forward's intermediates or, without
      flash attention, the attention scores: `n_head x cells x n_ubatch` f32,
      which also grow with the context. Each comes with a few f32 copies of the
      hidden state.
    """
    full, window = _cells(shape, context, traits)
    has_full = any(heads and not sliding for heads, sliding in zip(shape.n_head_kv, shape.swa_layers))
    has_window = any(heads and sliding for heads, sliding in zip(shape.n_head_kv, shape.swa_layers))
    batch = traits.n_ubatch

    mask_cells = (full if has_full else 0) + (window if has_window else 0)
    masks = mask_cells * batch * (2 if traits.flash_attention else _F32)
    logits = (shape.vocab_size or UNKNOWN_VOCAB) * traits.slots * _F32

    dense = _FFN_INTERMEDIATES * (shape.feed_forward_length or _NO_FFN_WIDTH * shape.n_embd)
    experts = 0
    if shape.expert_used_count and shape.expert_feed_forward_length:
        # A token's chosen experts run side by side, and their outputs are held
        # apart until they're summed.
        experts = shape.expert_used_count * (_FFN_INTERMEDIATES * shape.expert_feed_forward_length + shape.n_embd)
    feed_forward = (max(dense, experts) + _HIDDEN_COPIES_FFN * shape.n_embd) * batch * _F32

    scores = 0
    if not traits.flash_attention and (has_full or has_window):
        cells = full if has_full else window
        scores = (shape.n_head * cells + _HIDDEN_COPIES_ATTENTION * shape.n_embd) * batch * _F32
    return masks + logits + max(feed_forward, scores)


def vision_encoder_bytes(file_bytes: int) -> int:
    """A vision encoder's file and its working memory; nothing without one."""
    return file_bytes + VISION_OVERHEAD_BYTES if file_bytes else 0


def gpu_weight_bytes(shape: ModelShape, file_bytes: int) -> int:
    """The weights llama.cpp puts on the GPU, judged from a GGUF's size alone.

    An MTP head is stored as extra layers that aren't loaded, so their share
    comes off, taking every layer to be the same size.

    The token embeddings stay in, so this can overstate. llama.cpp keeps them on
    the CPU, but a model whose output layer reuses them gets a copy of them on
    the GPU, and metadata doesn't say which kind a model is. When the output
    layer is separate, this is `vocab x width` at the embeddings' bits too high:
    546 MiB for Qwen3.5-9B Q4_K_M. A downloaded file's tensor list settles it:
    see `gpu_tensor_bytes`."""
    if not shape.nextn_layers:
        return file_bytes
    return file_bytes * shape.n_layer // (shape.n_layer + shape.nextn_layers)


def gpu_tensor_bytes(shape: ModelShape, tensor_bytes: Mapping[str, int]) -> int:
    """The weights llama.cpp puts on the GPU, from the size of each tensor in
    the file, with every layer offloaded. That's every tensor but:

    - the input layer (`INPUT_TENSORS`), which stays in system RAM;
    - an MTP head's layers, numbered from `n_layer` up, which aren't loaded.

    A model with no `output.weight` reuses the token embeddings as its output
    layer, so a copy of them does go on the GPU."""
    total = 0
    for name, size in tensor_bytes.items():
        block = _BLOCK.match(name)
        if name in INPUT_TENSORS or (block is not None and int(block.group(1)) >= shape.n_layer):
            continue
        total += size
    if "output.weight" not in tensor_bytes:
        total += tensor_bytes.get("token_embd.weight", 0)
    return total


def estimate(
    shape: ModelShape, weights_bytes: int, context: int, traits: RuntimeTraits, vision_file_bytes: int = 0
) -> MemoryEstimate:
    """The memory for running the model at `context` tokens, with
    `weights_bytes` of it on the GPU and a vision encoder of that size, if any."""
    return MemoryEstimate(
        context=context,
        weights=weights_bytes,
        kv_cache=kv_cache_bytes(shape, context, traits),
        recurrent_state=recurrent_state_bytes(shape, traits),
        compute=compute_bytes(shape, context, traits),
        vision=vision_encoder_bytes(vision_file_bytes),
        runtime=traits.runtime_overhead_bytes,
    )


def max_gpu_context(
    shape: ModelShape, weights_bytes: int, free_bytes: int, traits: RuntimeTraits, vision_file_bytes: int = 0
) -> int | None:
    """The longest context that keeps the whole model on the GPU, the way
    `--fit` would choose it: a multiple of 256, at least 4096 and at most what
    the model was trained for (or TARGET_CONTEXT, if it doesn't say). None if
    even the shortest doesn't fit, which leaves `--fit` moving layers off."""

    def fits(context: int) -> bool:
        return estimate(shape, weights_bytes, context, traits, vision_file_bytes).fits(free_bytes)

    limit = shape.context_length or TARGET_CONTEXT
    if fits(limit):
        return limit
    shortest = min(MIN_FIT_CONTEXT, limit)
    if not fits(shortest):
        return None
    # Memory only grows with the context, so the answer is found by bisection,
    # in steps of 256: `low` always fits and `high` never does.
    low, high = shortest // CONTEXT_PADDING, -(-limit // CONTEXT_PADDING)
    while high - low > 1:
        middle = (low + high) // 2
        if fits(middle * CONTEXT_PADDING):
            low = middle
        else:
            high = middle
    return low * CONTEXT_PADDING

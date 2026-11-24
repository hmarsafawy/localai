"""What a language model is made of, as far as its memory goes: how many layers
it runs, what a token costs in each layer's KV cache, which layers only look
back a sliding window, which keep a fixed recurrent state instead, and how wide
the parts are that size llama.cpp's working buffers.

Read from a GGUF's metadata, which a small range request reaches, or from a
safetensors model's `config.json`, so it's known before anything is downloaded.
Pure functions - nothing here touches the network.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from services.weight_headers import GgufArray

# llama.cpp's own patterns for architectures whose GGUFs don't store one: of
# every n layers, the first n-1 look back a sliding window and the last sees
# everything (`set_swa_pattern` in llama-hparams.cpp).
GGUF_SWA_PATTERNS = {"gemma2": 2, "gemma3": 6, "gemma3n": 5, "gpt-oss": 2, "cohere2": 4}
CONFIG_SWA_PATTERNS = {"gemma2": 2, "gemma3": 6, "gemma3_text": 6, "gpt_oss": 2, "cohere2": 4}
# Every layer has both attention and a recurrent state.
PARALLEL_HYBRIDS = {"falcon-h1", "falcon_h1"}
# RWKV keeps this many shifted tokens per layer unless its metadata says otherwise.
RWKV_TOKEN_SHIFTS = 2
# DeepSeek's convention, for an MLA model that doesn't say.
MLA_ROPE_DIMENSIONS = 64

# `layer_types` values in a transformers config.
_SLIDING_LAYER_TYPES = {"sliding_attention", "chunked_attention"}
_ATTENTION_LAYER_TYPES = {"full_attention", "attention", *_SLIDING_LAYER_TYPES}


@dataclass(frozen=True)
class ModelShape:
    architecture: str
    # The layers llama.cpp runs. A multi-token-prediction head is stored as
    # extra layers that it doesn't load, so those are counted in `nextn_layers`.
    n_layer: int
    n_embd: int
    n_head: int
    # Per layer. 0 where the layer has no KV cache, being recurrent or a convolution.
    n_head_kv: tuple[int, ...]
    # What one head's K and V hold per token, in elements. An MLA model caches
    # one compressed latent instead, with V read out of it, so V is 0.
    key_length: int
    value_length: int
    # Per layer: whether it only looks back `sliding_window` tokens.
    swa_layers: tuple[bool, ...]
    sliding_window: int | None
    # Per layer: whether it keeps a state of `recurrent_state_size` f32 values
    # per sequence, whatever the context.
    recurrent_layers: tuple[bool, ...]
    recurrent_state_size: int
    # What it was trained for (`n_ctx_train`), and so the most llama.cpp gives it.
    context_length: int | None
    vocab_size: int | None
    # The widest dense feed-forward, and for a mixture of experts the size of each.
    feed_forward_length: int | None
    expert_count: int | None = None
    expert_used_count: int | None = None
    expert_feed_forward_length: int | None = None
    nextn_layers: int = 0
    # Whether the output layer reuses the token embeddings. None when unknown,
    # as it is from metadata alone: it shows only in the tensor list.
    tied_embeddings: bool | None = None

    @property
    def attention_layers(self) -> int:
        return sum(1 for heads in self.n_head_kv if heads > 0)


def _int(value: object) -> int | None:
    # `bool` is an `int` too, and never means a count.
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _first_int(source: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = _int(source.get(key))
        if value is not None:
            return value
    return None


def _per_layer(value: object, n_layer: int) -> tuple[int, ...] | None:
    """A value given once for every layer, or once per layer."""
    single = _int(value)
    if single is not None:
        return (single,) * n_layer
    if isinstance(value, Sequence) and not isinstance(value, str) and len(value) >= n_layer:
        items = [_int(item) for item in value[:n_layer]]
        if all(item is not None for item in items):
            return tuple(item for item in items if item is not None)
    return None


def _flags(value: object, n_layer: int) -> tuple[bool, ...] | None:
    if isinstance(value, Sequence) and not isinstance(value, str) and len(value) >= n_layer:
        return tuple(bool(item) for item in value[:n_layer])
    return None


def swa_pattern(every: int, n_layer: int) -> tuple[bool, ...]:
    """llama.cpp's `set_swa_pattern`: 0 makes every layer a sliding one, 1 none."""
    return tuple(every == 0 or layer % every < every - 1 for layer in range(n_layer))


def _ssm_state(conv_kernel: int | None, inner: int | None, state: int | None, groups: int | None) -> int:
    """llama.cpp's `n_embd_r() + n_embd_s()` for Mamba and gated delta nets:
    the convolution's last few inputs, then the state itself."""
    if not inner or not state:
        return 0
    convolution = max((conv_kernel or 0) - 1, 0) * (inner + 2 * (groups or 1) * state)
    return convolution + state * inner


def _assemble(
    architecture: str,
    n_layer: int,
    n_embd: int,
    n_head: int,
    n_head_kv: tuple[int, ...],
    key_length: int,
    value_length: int,
    swa_layers: tuple[bool, ...],
    sliding_window: int | None,
    recurrent_layers: tuple[bool, ...],
    recurrent_state_size: int,
    parallel: bool,
    **rest: Any,
) -> ModelShape | None:
    if not parallel:
        # A recurrent layer has no KV cache, whatever the head count says for all layers.
        n_head_kv = tuple(0 if recurrent else heads for heads, recurrent in zip(n_head_kv, recurrent_layers))
    if not any(n_head_kv) and not any(recurrent_layers):
        return None  # Neither attention nor a state: not a model this can size.
    if any(n_head_kv) and key_length + value_length <= 0:
        return None
    if not sliding_window:
        swa_layers, sliding_window = (False,) * n_layer, None
    return ModelShape(
        architecture=architecture,
        n_layer=n_layer,
        n_embd=n_embd,
        n_head=n_head,
        n_head_kv=n_head_kv,
        key_length=key_length,
        value_length=value_length,
        swa_layers=swa_layers,
        sliding_window=sliding_window,
        recurrent_layers=recurrent_layers,
        recurrent_state_size=recurrent_state_size if any(recurrent_layers) else 0,
        **rest,
    )


def shape_from_gguf(metadata: Mapping[str, Any]) -> ModelShape | None:
    """From a GGUF's `{architecture}.*` keys, as llama.cpp reads them. None if
    the ones every model has are missing, as when the bytes read stop short."""
    architecture = metadata.get("general.architecture")
    if not isinstance(architecture, str):
        return None

    def get(name: str) -> object:
        return metadata.get(f"{architecture}.{name}")

    block_count = _int(get("block_count"))
    n_embd = _int(get("embedding_length"))
    if not block_count or not n_embd:
        return None
    nextn = _int(get("nextn_predict_layers")) or 0
    n_layer = block_count - nextn
    if n_layer <= 0:
        return None

    head_counts = _per_layer(get("attention.head_count"), n_layer) or (0,) * n_layer
    n_head = max(head_counts)
    # GGUF's rule: no KV head count means one per query head.
    n_head_kv = _per_layer(get("attention.head_count_kv"), n_layer) or head_counts
    default_length = n_embd // n_head if n_head else 0
    key_length = _int(get("attention.key_length")) or default_length
    value_length = _int(get("attention.value_length")) or default_length
    kv_lora_rank = _int(get("attention.kv_lora_rank"))
    if kv_lora_rank:
        # MLA caches the compressed latent plus the rotary part, as one head,
        # and reads V out of the same cache.
        key_length = kv_lora_rank + (_int(get("rope.dimension_count")) or MLA_ROPE_DIMENSIONS)
        value_length = 0
        n_head_kv = tuple(1 if heads else 0 for heads in n_head_kv)

    parallel = architecture in PARALLEL_HYBRIDS
    wkv_head_size = _int(get("wkv.head_size"))
    l_cache = _int(get("shortconv.l_cache"))
    interval = _int(get("full_attention_interval"))
    if wkv_head_size:  # RWKV
        recurrent = (True,) * n_layer
        token_shifts = _int(get("token_shift_count")) or RWKV_TOKEN_SHIFTS
        state = token_shifts * n_embd + n_embd * wkv_head_size
    else:
        if parallel:
            recurrent = (True,) * n_layer
        elif interval:
            recurrent = tuple((layer + 1) % interval != 0 for layer in range(n_layer))
        else:
            recurrent = tuple(heads == 0 for heads in n_head_kv)
        if l_cache:  # LFM2's short convolutions
            state = n_embd * (l_cache - 1)
        else:
            state = _ssm_state(
                _int(get("ssm.conv_kernel")),
                _int(get("ssm.inner_size")),
                _int(get("ssm.state_size")),
                _int(get("ssm.group_count")),
            )
        if not state:
            # Layers without attention but with no state either (an encoder,
            # or keys this doesn't know) - not recurrent then, just unsized.
            recurrent = (False,) * n_layer

    sliding_window = _int(get("attention.sliding_window"))
    pattern = get("attention.sliding_window_pattern")
    swa = _flags(pattern, n_layer)
    if swa is None:
        every = _int(pattern) if _int(pattern) is not None else GGUF_SWA_PATTERNS.get(architecture)
        # Without a pattern, every layer is taken to see the whole context. That
        # can only overstate the cache.
        swa = swa_pattern(every, n_layer) if every is not None else (False,) * n_layer

    tokens = metadata.get("tokenizer.ggml.tokens")
    vocab_size = _int(get("vocab_size")) or (
        tokens.length if isinstance(tokens, GgufArray) else len(tokens) if isinstance(tokens, tuple) else None
    )
    feed_forward = _per_layer(get("feed_forward_length"), n_layer)
    return _assemble(
        architecture,
        n_layer,
        n_embd,
        n_head,
        n_head_kv,
        key_length,
        value_length,
        swa,
        sliding_window,
        recurrent,
        state,
        parallel,
        context_length=_int(get("context_length")),
        vocab_size=vocab_size,
        feed_forward_length=max(feed_forward) if feed_forward else None,
        expert_count=_int(get("expert_count")),
        expert_used_count=_int(get("expert_used_count")),
        expert_feed_forward_length=_int(get("expert_feed_forward_length")),
        nextn_layers=nextn,
    )


def shape_from_config(config: Mapping[str, Any]) -> ModelShape | None:
    """From a transformers `config.json`, for a model llama.cpp would convert.
    A vision model's language model is its `text_config`. Rougher than the
    GGUF: configs name things however each model family likes."""
    text = config.get("text_config")
    source: Mapping[str, Any] = {**config, **text} if isinstance(text, Mapping) else config
    architecture = str(source.get("model_type") or "unknown")

    n_layer = _first_int(source, "num_hidden_layers", "n_layer", "num_layers", "n_layers")
    n_embd = _first_int(source, "hidden_size", "d_model", "n_embd", "dim")
    if not n_layer or not n_embd:
        return None
    head_counts = (
        _per_layer(
            next((source[key] for key in ("num_attention_heads", "n_head", "n_heads") if key in source), None), n_layer
        )
        or (0,) * n_layer
    )
    n_head = max(head_counts)
    kv_heads = next(
        (source[key] for key in ("num_key_value_heads", "num_kv_heads", "n_head_kv") if key in source), None
    )
    n_head_kv = _per_layer(kv_heads, n_layer) or (
        tuple(1 if heads else 0 for heads in head_counts) if source.get("multi_query") else head_counts
    )
    head_dim = _first_int(source, "head_dim") or (n_embd // n_head if n_head else 0)
    key_length = value_length = head_dim
    kv_lora_rank = _first_int(source, "kv_lora_rank")
    if kv_lora_rank:
        key_length = kv_lora_rank + (_first_int(source, "qk_rope_head_dim") or MLA_ROPE_DIMENSIONS)
        value_length = 0
        n_head_kv = tuple(1 if heads else 0 for heads in n_head_kv)

    layer_types = source.get("layer_types") or source.get("layers_block_type")
    interval = _first_int(source, "full_attention_interval")
    parallel = architecture in PARALLEL_HYBRIDS
    if isinstance(layer_types, Sequence) and not isinstance(layer_types, str) and len(layer_types) >= n_layer:
        kinds = [str(kind) for kind in layer_types[:n_layer]]
        recurrent = tuple(kind not in _ATTENTION_LAYER_TYPES for kind in kinds)
        swa: tuple[bool, ...] | None = tuple(kind in _SLIDING_LAYER_TYPES for kind in kinds)
    else:
        if parallel:
            recurrent = (True,) * n_layer
        elif interval:
            recurrent = tuple((layer + 1) % interval != 0 for layer in range(n_layer))
        else:
            recurrent = tuple(heads == 0 for heads in n_head_kv)
        swa = None
    if parallel:
        recurrent = (True,) * n_layer

    l_cache = _first_int(source, "conv_L_cache")
    if l_cache:
        state = n_embd * (l_cache - 1)
    else:
        linear_heads = _first_int(source, "linear_num_value_heads")
        linear_dim = _first_int(source, "linear_value_head_dim")
        mamba_heads = _first_int(source, "mamba_n_heads")
        mamba_dim = _first_int(source, "mamba_d_head")
        expand = _first_int(source, "mamba_expand", "expand")
        inner = (
            (linear_heads * linear_dim if linear_heads and linear_dim else None)
            or _first_int(source, "mamba_d_ssm")
            or (mamba_heads * mamba_dim if mamba_heads and mamba_dim else None)
            or (expand * n_embd if expand else None)
        )
        state = _ssm_state(
            _first_int(source, "linear_conv_kernel_dim", "mamba_d_conv", "conv_kernel", "d_conv"),
            inner,
            _first_int(source, "linear_key_head_dim", "mamba_d_state", "ssm_state_size", "state_size", "d_state"),
            _first_int(source, "linear_num_key_heads", "mamba_n_groups", "n_groups"),
        )
    if not state and not l_cache:
        recurrent = (False,) * n_layer

    sliding_window = _first_int(source, "sliding_window", "attention_chunk_size")
    if source.get("use_sliding_window") is False:
        sliding_window = None
    if swa is None:
        every = _first_int(source, "sliding_window_pattern")
        every = every if every is not None else CONFIG_SWA_PATTERNS.get(architecture)
        swa = swa_pattern(every, n_layer) if every is not None else (False,) * n_layer

    feed_forward = _per_layer(source.get("intermediate_size"), n_layer)
    expert_count = _first_int(source, "num_local_experts", "num_experts", "n_routed_experts")
    tied = source.get("tie_word_embeddings")
    return _assemble(
        architecture,
        n_layer,
        n_embd,
        n_head,
        n_head_kv,
        key_length,
        value_length,
        swa,
        sliding_window,
        recurrent,
        state,
        parallel,
        context_length=_first_int(
            source, "max_position_embeddings", "n_positions", "max_sequence_length", "seq_length"
        ),
        vocab_size=_first_int(source, "vocab_size"),
        feed_forward_length=max(feed_forward) if feed_forward else None,
        expert_count=expert_count,
        expert_used_count=_first_int(source, "num_experts_per_tok", "moe_top_k") if expert_count else None,
        expert_feed_forward_length=(
            _first_int(source, "moe_intermediate_size", "expert_intermediate_size", "intermediate_size")
            if expert_count
            else None
        ),
        nextn_layers=_first_int(source, "num_nextn_predict_layers", "mtp_num_hidden_layers") or 0,
        tied_embeddings=tied if isinstance(tied, bool) else None,
    )

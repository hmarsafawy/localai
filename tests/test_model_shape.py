"""Metadata here is copied from real GGUFs (keys that don't affect the shape
left out) and real `config.json` files on the Hub."""

from __future__ import annotations

import struct
from typing import Any

from services.model_shape import ModelShape, shape_from_config, shape_from_gguf, swa_pattern
from services.weight_headers import GGUF_METADATA_FETCH_BYTES, GgufArray, parse_gguf_metadata
from tests.test_model_options import gguf_bytes, string_array

LLAMA_3_8B_GGUF: dict[str, Any] = {
    "general.architecture": "llama",
    "llama.block_count": 32,
    "llama.context_length": 8192,
    "llama.embedding_length": 4096,
    "llama.feed_forward_length": 14336,
    "llama.attention.head_count": 32,
    "llama.attention.head_count_kv": 8,
    "llama.rope.dimension_count": 128,
    "llama.vocab_size": 128256,
    "tokenizer.ggml.model": "gpt2",
}

LLAMA_3_8B_CONFIG: dict[str, Any] = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "max_position_embeddings": 8192,
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 8,
    "tie_word_embeddings": False,
    "vocab_size": 128256,
}

# DavidAU's Qwen3.5-9B MTP build: a hybrid of gated delta net layers and full
# attention every fourth layer, plus one MTP layer llama.cpp doesn't load.
QWEN35_9B_MTP_GGUF: dict[str, Any] = {
    "general.architecture": "qwen35",
    "qwen35.block_count": 33,
    "qwen35.context_length": 262144,
    "qwen35.embedding_length": 4096,
    "qwen35.feed_forward_length": 12288,
    "qwen35.attention.head_count": 16,
    "qwen35.attention.head_count_kv": 4,
    "qwen35.rope.dimension_sections": (11, 11, 10, 0),
    "qwen35.attention.key_length": 256,
    "qwen35.attention.value_length": 256,
    "qwen35.ssm.conv_kernel": 4,
    "qwen35.ssm.state_size": 128,
    "qwen35.ssm.group_count": 16,
    "qwen35.ssm.time_step_rank": 32,
    "qwen35.ssm.inner_size": 4096,
    "qwen35.full_attention_interval": 4,
    "qwen35.rope.dimension_count": 64,
    "qwen35.nextn_predict_layers": 1,
    "tokenizer.ggml.model": "gpt2",
    "tokenizer.ggml.tokens": GgufArray(8, 248320),
}

# LiquidAI's LFM3-1.7B: short convolution layers between attention ones, some
# of those with a sliding window.
LFM3_GGUF: dict[str, Any] = {
    "general.architecture": "lfm3",
    "lfm3.block_count": 24,
    "lfm3.context_length": 262144,
    "lfm3.embedding_length": 2048,
    "lfm3.feed_forward_length": 7680,
    "lfm3.attention.head_count": 16,
    "lfm3.attention.head_count_kv": (0, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0),
    "lfm3.attention.key_length": 128,
    "lfm3.attention.value_length": 128,
    "lfm3.attention.sliding_window_pattern": tuple(layer % 4 == 0 and layer > 0 for layer in range(24)),
    "lfm3.attention.sliding_window": 256,
    "lfm3.shortconv.l_cache": 7,
    "tokenizer.ggml.tokens": GgufArray(8, 128000),
}

GEMMA_3_12B_GGUF: dict[str, Any] = {
    "general.architecture": "gemma3",
    "gemma3.block_count": 48,
    "gemma3.context_length": 131072,
    "gemma3.embedding_length": 3840,
    "gemma3.feed_forward_length": 15360,
    "gemma3.attention.head_count": 16,
    "gemma3.attention.head_count_kv": 8,
    "gemma3.attention.key_length": 256,
    "gemma3.attention.value_length": 256,
    "gemma3.attention.sliding_window": 1024,
    "tokenizer.ggml.tokens": GgufArray(8, 262144),
}

GPT_OSS_20B_GGUF: dict[str, Any] = {
    "general.architecture": "gpt-oss",
    "gpt-oss.block_count": 24,
    "gpt-oss.context_length": 131072,
    "gpt-oss.embedding_length": 2880,
    "gpt-oss.feed_forward_length": 2880,
    "gpt-oss.attention.head_count": 64,
    "gpt-oss.attention.head_count_kv": 8,
    "gpt-oss.attention.key_length": 64,
    "gpt-oss.attention.value_length": 64,
    "gpt-oss.attention.sliding_window": 128,
    "gpt-oss.expert_count": 32,
    "gpt-oss.expert_used_count": 4,
    "gpt-oss.expert_feed_forward_length": 2880,
    "tokenizer.ggml.tokens": GgufArray(8, 201088),
}

DEEPSEEK_V3_GGUF: dict[str, Any] = {
    "general.architecture": "deepseek2",
    "deepseek2.block_count": 61,
    "deepseek2.context_length": 163840,
    "deepseek2.embedding_length": 7168,
    "deepseek2.feed_forward_length": 18432,
    "deepseek2.attention.head_count": 128,
    "deepseek2.attention.head_count_kv": 128,
    "deepseek2.attention.key_length": 192,
    "deepseek2.attention.value_length": 128,
    "deepseek2.attention.kv_lora_rank": 512,
    "deepseek2.rope.dimension_count": 64,
    "deepseek2.expert_count": 256,
    "deepseek2.expert_used_count": 8,
    "deepseek2.expert_feed_forward_length": 2048,
}

MAMBA_GGUF: dict[str, Any] = {
    "general.architecture": "mamba",
    "mamba.block_count": 64,
    "mamba.context_length": 1048576,
    "mamba.embedding_length": 2560,
    "mamba.feed_forward_length": 0,
    "mamba.attention.head_count": 0,
    "mamba.ssm.conv_kernel": 4,
    "mamba.ssm.inner_size": 5120,
    "mamba.ssm.state_size": 16,
    "mamba.ssm.time_step_rank": 160,
}


def attention_layers(shape: ModelShape) -> list[int]:
    return [layer for layer, heads in enumerate(shape.n_head_kv) if heads]


def test_a_plain_transformer() -> None:
    shape = shape_from_gguf(LLAMA_3_8B_GGUF)
    assert shape is not None
    assert shape.n_layer == 32
    assert shape.n_head == 32
    assert shape.n_head_kv == (8,) * 32
    assert (shape.key_length, shape.value_length) == (128, 128)
    assert shape.context_length == 8192
    assert shape.vocab_size == 128256
    assert shape.feed_forward_length == 14336
    assert not any(shape.swa_layers) and shape.sliding_window is None
    assert not any(shape.recurrent_layers) and shape.recurrent_state_size == 0


def test_the_config_of_a_model_gives_the_same_shape_as_its_gguf() -> None:
    from_gguf = shape_from_gguf(LLAMA_3_8B_GGUF)
    from_config = shape_from_config(LLAMA_3_8B_CONFIG)
    assert from_gguf is not None and from_config is not None
    for field in (
        "n_layer",
        "n_embd",
        "n_head",
        "n_head_kv",
        "key_length",
        "value_length",
        "context_length",
        "vocab_size",
        "feed_forward_length",
        "swa_layers",
        "recurrent_layers",
    ):
        assert getattr(from_config, field) == getattr(from_gguf, field), field
    assert from_config.tied_embeddings is False


def test_mtp_layers_arent_counted_and_recurrent_layers_have_no_kv_cache() -> None:
    shape = shape_from_gguf(QWEN35_9B_MTP_GGUF)
    assert shape is not None
    assert shape.n_layer == 32
    assert shape.nextn_layers == 1
    assert attention_layers(shape) == [3, 7, 11, 15, 19, 23, 27, 31]
    assert shape.n_head_kv[3] == 4
    assert sum(shape.recurrent_layers) == 24
    # 3 inputs of the convolution over 4096 + 2*16*128 channels, then a 128 x 4096 state.
    assert shape.recurrent_state_size == 3 * 8192 + 128 * 4096
    assert (shape.key_length, shape.value_length) == (256, 256)
    assert shape.vocab_size == 248320


def test_per_layer_head_counts_and_a_stored_sliding_window_pattern() -> None:
    shape = shape_from_gguf(LFM3_GGUF)
    assert shape is not None
    assert attention_layers(shape) == list(range(2, 24, 2))
    assert sum(shape.recurrent_layers) == 13
    assert shape.recurrent_state_size == 2048 * 6
    assert [layer for layer, sliding in enumerate(shape.swa_layers) if sliding] == [4, 8, 12, 16, 20]
    assert shape.sliding_window == 256
    assert shape.vocab_size == 128000


def test_llama_cpps_own_pattern_when_the_gguf_has_none() -> None:
    shape = shape_from_gguf(GEMMA_3_12B_GGUF)
    assert shape is not None
    assert sum(shape.swa_layers) == 40
    assert [layer for layer, sliding in enumerate(shape.swa_layers) if not sliding] == [5, 11, 17, 23, 29, 35, 41, 47]
    assert shape.sliding_window == 1024


def test_a_mixture_of_experts_with_alternating_sliding_layers() -> None:
    shape = shape_from_gguf(GPT_OSS_20B_GGUF)
    assert shape is not None
    assert shape.swa_layers == tuple(layer % 2 == 0 for layer in range(24))
    assert (shape.expert_count, shape.expert_used_count, shape.expert_feed_forward_length) == (32, 4, 2880)


def test_mla_caches_one_compressed_head() -> None:
    shape = shape_from_gguf(DEEPSEEK_V3_GGUF)
    assert shape is not None
    assert shape.n_head_kv == (1,) * 61
    assert (shape.key_length, shape.value_length) == (512 + 64, 0)


def test_a_pure_state_space_model_has_no_kv_cache() -> None:
    shape = shape_from_gguf(MAMBA_GGUF)
    assert shape is not None
    assert shape.attention_layers == 0
    assert all(shape.recurrent_layers)
    assert shape.recurrent_state_size == 3 * (5120 + 2 * 16) + 16 * 5120


def test_unknown_without_the_keys_every_model_has() -> None:
    assert shape_from_gguf({"general.architecture": "llama", "llama.context_length": 8192}) is None
    assert shape_from_gguf({}) is None
    assert shape_from_config({"model_type": "llama"}) is None


def test_swa_pattern_follows_llama_cpp() -> None:
    assert swa_pattern(1, 3) == (False, False, False)
    assert swa_pattern(0, 2) == (True, True)
    assert swa_pattern(3, 6) == (True, True, False, True, True, False)


def test_a_vision_models_text_config_with_a_sliding_window_pattern() -> None:
    config = {
        "model_type": "gemma3",
        "text_config": {
            "model_type": "gemma3_text",
            "hidden_size": 3840,
            "num_hidden_layers": 48,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 256,
            "sliding_window": 1024,
            "sliding_window_pattern": 6,
            "max_position_embeddings": 131072,
            "vocab_size": 262208,
        },
        "vision_config": {"hidden_size": 1152, "num_hidden_layers": 27},
    }
    shape = shape_from_config(config)
    gguf = shape_from_gguf(GEMMA_3_12B_GGUF)
    assert shape is not None and gguf is not None
    assert shape.swa_layers == gguf.swa_layers
    assert (shape.n_layer, shape.key_length, shape.n_head_kv) == (48, 256, (8,) * 48)


def test_layer_types_in_a_config() -> None:
    config = {
        "model_type": "gpt_oss",
        "hidden_size": 2880,
        "num_hidden_layers": 24,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "sliding_window": 128,
        "layer_types": ["sliding_attention", "full_attention"] * 12,
        "num_local_experts": 32,
        "num_experts_per_tok": 4,
        "intermediate_size": 2880,
    }
    shape = shape_from_config(config)
    gguf = shape_from_gguf(GPT_OSS_20B_GGUF)
    assert shape is not None and gguf is not None
    assert shape.swa_layers == gguf.swa_layers
    assert (shape.expert_count, shape.expert_used_count, shape.expert_feed_forward_length) == (32, 4, 2880)


def test_a_hybrid_config_with_linear_attention() -> None:
    config = {
        "model_type": "qwen3_next",
        "hidden_size": 2048,
        "num_hidden_layers": 48,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "full_attention_interval": 4,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_value_head_dim": 128,
        "max_position_embeddings": 262144,
    }
    shape = shape_from_config(config)
    assert shape is not None
    assert shape.attention_layers == 12
    assert shape.recurrent_state_size == 3 * (4096 + 2 * 16 * 128) + 128 * 4096


def test_a_sliding_window_thats_switched_off_is_ignored() -> None:
    config = {**LLAMA_3_8B_CONFIG, "model_type": "qwen2", "sliding_window": 32768, "use_sliding_window": False}
    shape = shape_from_config(config)
    assert shape is not None
    assert shape.sliding_window is None and not any(shape.swa_layers)


def test_per_layer_values_and_the_vocabulary_are_read_from_a_small_prefix() -> None:
    """In a real LLM GGUF the tokenizer runs to megabytes; the shape has to
    come out of the first 64 KB all the same."""
    layers = 4
    data = gguf_bytes(
        ("general.architecture", 8, struct.pack("<Q", 4) + b"lfm3"),
        ("lfm3.block_count", 4, struct.pack("<I", layers)),
        ("lfm3.embedding_length", 4, struct.pack("<I", 2048)),
        ("lfm3.attention.head_count", 4, struct.pack("<I", 16)),
        ("lfm3.attention.head_count_kv", 9, struct.pack("<IQ", 5, layers) + struct.pack("<4i", 0, 4, 0, 4)),
        ("lfm3.shortconv.l_cache", 4, struct.pack("<I", 3)),
        ("tokenizer.ggml.tokens", 9, string_array(50_000, 8)),
        ("tokenizer.ggml.token_type", 9, struct.pack("<IQ", 5, 50_000) + b"\0" * 4 * 50_000),
    )
    metadata = parse_gguf_metadata(data[:GGUF_METADATA_FETCH_BYTES])
    assert metadata["lfm3.attention.head_count_kv"] == (0, 4, 0, 4)
    assert metadata["tokenizer.ggml.tokens"] == GgufArray(8, 50_000)
    assert "tokenizer.ggml.token_type" not in metadata  # Past the prefix.

    shape = shape_from_gguf(metadata)
    assert shape is not None
    assert shape.n_head_kv == (0, 4, 0, 4)
    assert shape.vocab_size == 50_000


def test_long_numeric_arrays_keep_only_their_length() -> None:
    data = gguf_bytes(("tokenizer.ggml.token_type", 9, struct.pack("<IQ", 5, 5000) + b"\0" * 4 * 5000))
    assert parse_gguf_metadata(data)["tokenizer.ggml.token_type"] == GgufArray(5, 5000)

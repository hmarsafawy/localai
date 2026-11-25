"""The memory estimate, checked by hand against llama.cpp's formulas and
against what llama-server actually allocated (llama.cpp b12359, ROCm, an RX
9070 XT; docs/memory-estimate.md has the runs)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from services.memory_estimate import (
    FIT_MARGIN_MIB,
    MIN_FIT_CONTEXT,
    TARGET_CONTEXT,
    VISION_OVERHEAD_BYTES,
    MiB,
    RuntimeTraits,
    compute_bytes,
    estimate,
    gpu_weight_bytes,
    kv_cache_bytes,
    max_gpu_context,
    padded_context,
    recurrent_state_bytes,
    target_context,
    vision_encoder_bytes,
)
from services.model_shape import ModelShape, shape_from_gguf
from services.weight_headers import GgufArray
from tests.test_model_shape import (
    DEEPSEEK_V3_GGUF,
    GEMMA_3_12B_GGUF,
    GPT_OSS_20B_GGUF,
    LFM3_GGUF,
    LLAMA_3_8B_GGUF,
    MAMBA_GGUF,
    QWEN35_9B_MTP_GGUF,
)

GiB = 1024 * MiB
KiB = 1024

SMOLLM2_135M_GGUF: dict[str, Any] = {
    "general.architecture": "llama",
    "llama.block_count": 30,
    "llama.context_length": 8192,
    "llama.embedding_length": 576,
    "llama.feed_forward_length": 1536,
    "llama.attention.head_count": 9,
    "llama.attention.head_count_kv": 3,
    "llama.rope.dimension_count": 64,
    "tokenizer.ggml.tokens": GgufArray(8, 49152),
}

ROCM = RuntimeTraits("rocm")


def shape(metadata: dict[str, Any]) -> ModelShape:
    result = shape_from_gguf(metadata)
    assert result is not None
    return result


def test_llama_3_8b_keeps_128_kib_per_token() -> None:
    # 32 layers x 8 KV heads x (128 + 128) values x 2 bytes.
    llama = shape(LLAMA_3_8B_GGUF)
    assert kv_cache_bytes(llama, 256, ROCM) == 256 * 128 * KiB
    assert kv_cache_bytes(llama, 1, ROCM) == kv_cache_bytes(llama, 256, ROCM)  # Padded to 256 cells.
    assert kv_cache_bytes(llama, 16384, ROCM) == 2 * GiB
    assert kv_cache_bytes(llama, 8192, ROCM) == 1 * GiB


def test_the_context_is_padded_to_256() -> None:
    llama = shape(LLAMA_3_8B_GGUF)
    assert padded_context(1000) == 1024
    assert padded_context(1024) == 1024
    assert kv_cache_bytes(llama, 1000, ROCM) == kv_cache_bytes(llama, 1024, ROCM)


def test_a_quantized_cache_is_smaller_by_its_block_size() -> None:
    llama = shape(LLAMA_3_8B_GGUF)
    # q8_0 stores 32 values in 34 bytes, against 64 bytes in f16.
    assert kv_cache_bytes(llama, 16384, RuntimeTraits("rocm", cache_type="q8_0")) == 2 * GiB * 34 // 64
    with pytest.raises(ValueError):
        RuntimeTraits("rocm", cache_type="q3_k")


def test_sliding_window_layers_keep_a_window_per_slot_plus_a_batch() -> None:
    gemma = shape(GEMMA_3_12B_GGUF)
    per_cell = 8 * (256 + 256) * 2  # One layer's K and V for one token.
    window = 1024 * 4 + 512  # 4,608 cells, already a multiple of 256.
    assert kv_cache_bytes(gemma, 16384, ROCM) == (8 * 16384 + 40 * window) * per_cell
    # About a fifth of what every layer seeing the whole context would take.
    assert kv_cache_bytes(gemma, 131072, ROCM) < 48 * 131072 * per_cell / 5
    # A window bigger than the context is cut to the context.
    assert kv_cache_bytes(gemma, 2048, ROCM) == 48 * 2048 * per_cell


def test_mla_caches_one_latent_per_token() -> None:
    deepseek = shape(DEEPSEEK_V3_GGUF)
    assert kv_cache_bytes(deepseek, 16384, ROCM) == 61 * 576 * 2 * 16384
    # Each of the 128 heads keeping its own K and V would be 71 times as much.
    assert 61 * 128 * (192 + 128) * 2 // (61 * 576 * 2) == 71


def test_recurrent_state_grows_with_the_slots_and_not_the_context() -> None:
    qwen = shape(QWEN35_9B_MTP_GGUF)
    # 24 delta net layers x 548,864 f32 values x 4 slots: what llama-server allocated.
    assert recurrent_state_bytes(qwen, ROCM) == 201 * MiB
    assert recurrent_state_bytes(qwen, RuntimeTraits("rocm", slots=1)) == 24 * 548864 * 4
    assert estimate(qwen, 0, 4096, ROCM).recurrent_state == estimate(qwen, 0, 65536, ROCM).recurrent_state
    # Only the 8 attention layers have a KV cache.
    assert kv_cache_bytes(qwen, 16384, ROCM) == 8 * 4 * 512 * 2 * 16384


def test_a_pure_state_space_model_needs_the_same_memory_at_any_context() -> None:
    mamba = shape(MAMBA_GGUF)
    assert kv_cache_bytes(mamba, 65536, ROCM) == 0
    assert estimate(mamba, GiB, 4096, ROCM).total == estimate(mamba, GiB, 1048576, ROCM).total


# What llama-server b12359 allocated on ROCm, from its log: (KV cache plus
# recurrent state in bytes, compute buffer in MiB).
MEASURED = [
    (SMOLLM2_135M_GGUF, 4096, True, 90 * MiB, 17.51),
    (SMOLLM2_135M_GGUF, 8192, True, 180 * MiB, 21.51),
    (SMOLLM2_135M_GGUF, 16384, False, 360 * MiB, 326.38),
    (LFM3_GGUF, 16384, True, (192 + 15) * MiB + 2_555_904, 78.71),
    (LFM3_GGUF, 65536, True, (768 + 15) * MiB + 2_555_904, 126.71),
    (LFM3_GGUF, 16384, False, (192 + 15) * MiB + 2_555_904, 571.01),
    (QWEN35_9B_MTP_GGUF, 16384, True, (512 + 201) * MiB, 120.02),
    (QWEN35_9B_MTP_GGUF, 65536, True, (2048 + 201) * MiB, 168.02),
    (QWEN35_9B_MTP_GGUF, 16384, False, (512 + 201) * MiB, 596.02),
]


@pytest.mark.parametrize(("metadata", "context", "flash_attention", "kv_and_state", "compute_mib"), MEASURED)
def test_what_llama_server_allocated(
    metadata: dict[str, Any], context: int, flash_attention: bool, kv_and_state: int, compute_mib: float
) -> None:
    model = shape(metadata)
    traits = RuntimeTraits("rocm", flash_attention=flash_attention)
    assert kv_cache_bytes(model, context, traits) + recurrent_state_bytes(model, traits) == kv_and_state
    # The compute buffer is approximated from its biggest parts: never under,
    # and over by at most a few MiB.
    over = compute_bytes(model, context, traits) / MiB - compute_mib
    assert 0 <= over <= 12


def test_without_flash_attention_the_scores_grow_with_the_context() -> None:
    qwen = shape(QWEN35_9B_MTP_GGUF)
    off = RuntimeTraits("rocm", flash_attention=False)

    def extra(context: int) -> int:
        return compute_bytes(qwen, context, off) - compute_bytes(qwen, context, ROCM)

    assert extra(65536) > extra(16384) > 0
    # 16 heads x 65,536 cells x 512 tokens of f32 scores dominate it.
    assert extra(65536) >= 16 * 65536 * 512 * 4


def test_sliding_and_full_layers_each_have_a_mask() -> None:
    gemma = shape(GEMMA_3_12B_GGUF)
    no_window = replace(gemma, swa_layers=(False,) * 48, sliding_window=None)
    # The window's mask is 4,608 cells x 512 tokens x 2 bytes.
    assert compute_bytes(gemma, 16384, ROCM) - compute_bytes(no_window, 16384, ROCM) == 4608 * 512 * 2


def test_a_mixture_of_experts_holds_every_chosen_experts_output() -> None:
    gpt_oss = shape(GPT_OSS_20B_GGUF)
    dense = replace(gpt_oss, expert_count=None, expert_used_count=None, expert_feed_forward_length=None)
    # 4 experts x (3 x 2,880 intermediates + a 2,880 output) against 3 x 2,880.
    assert compute_bytes(gpt_oss, 4096, ROCM) - compute_bytes(dense, 4096, ROCM) == (
        (4 * (3 * 2880 + 2880) - 3 * 2880) * 512 * 4
    )


def test_the_logits_are_for_one_token_per_slot() -> None:
    qwen = shape(QWEN35_9B_MTP_GGUF)
    small_vocab = replace(qwen, vocab_size=1000)
    assert compute_bytes(qwen, 4096, ROCM) - compute_bytes(small_vocab, 4096, ROCM) == (248320 - 1000) * 4 * 4


def test_the_estimate_adds_up() -> None:
    qwen = shape(QWEN35_9B_MTP_GGUF)
    result = estimate(qwen, 6 * GiB, 16384, ROCM, vision_file_bytes=900 * MiB)
    assert result.vision == 900 * MiB + VISION_OVERHEAD_BYTES
    assert result.runtime == 384 * MiB
    assert result.total == 6 * GiB + (512 + 201) * MiB + result.compute + result.vision + 384 * MiB
    assert result.fits(result.total + FIT_MARGIN_MIB * MiB)
    assert not result.fits(result.total + FIT_MARGIN_MIB * MiB - 1)


def test_the_gpu_runtime_takes_a_measured_amount_per_backend() -> None:
    assert RuntimeTraits("rocm").runtime_overhead_bytes == 384 * MiB
    assert RuntimeTraits("vulkan").runtime_overhead_bytes == 192 * MiB
    assert RuntimeTraits("cpu").runtime_overhead_bytes == 0
    # Not measured yet: ROCm's figure, as CUDA shares its code.
    assert RuntimeTraits("cuda").runtime_overhead_bytes == 384 * MiB


def test_a_vision_encoder_adds_its_working_memory() -> None:
    assert vision_encoder_bytes(0) == 0
    assert vision_encoder_bytes(918166080) == 918166080 + 768 * MiB


def test_mtp_layers_come_off_the_weights() -> None:
    qwen = shape(QWEN35_9B_MTP_GGUF)
    assert gpu_weight_bytes(qwen, 33 * GiB) == 32 * GiB
    assert gpu_weight_bytes(shape(LLAMA_3_8B_GGUF), 5 * GiB) == 5 * GiB


def test_the_target_context_is_capped_at_what_the_model_was_trained_for() -> None:
    assert target_context(shape(QWEN35_9B_MTP_GGUF)) == TARGET_CONTEXT == 16384
    assert target_context(shape(LLAMA_3_8B_GGUF)) == 8192
    assert target_context(shape(QWEN35_9B_MTP_GGUF), 65536) == 65536
    assert target_context(replace(shape(LLAMA_3_8B_GGUF), context_length=None), 65536) == 65536


def test_the_longest_context_for_qwen35_on_this_16_gb_card() -> None:
    # 12,800 MiB free, the Q4_K_M build's exact 5,956.55 MiB of GPU weights and
    # its 876 MiB encoder. llama-server's own `--fit` chose 119,552, which put
    # some of its memory in system RAM: it leaves out the GPU runtime's share.
    qwen = shape(QWEN35_9B_MTP_GGUF)
    weights = round(5956.55 * MiB)
    longest = max_gpu_context(qwen, weights, 12800 * MiB, ROCM, 918166080)
    assert longest == 108032
    assert estimate(qwen, weights, longest, ROCM, 918166080).fits(12800 * MiB)
    assert not estimate(qwen, weights, longest + 256, ROCM, 918166080).fits(12800 * MiB)


def test_the_longest_context_grows_with_free_memory() -> None:
    qwen = shape(QWEN35_9B_MTP_GGUF)
    found = [max_gpu_context(qwen, 6 * GiB, free * GiB, ROCM) for free in (9, 10, 11, 12, 14)]
    assert all(context is not None and context % 256 == 0 for context in found)
    assert found == sorted(found)
    assert found[0] < found[-1]


def test_the_longest_context_is_at_most_what_the_model_was_trained_for() -> None:
    llama = shape(LLAMA_3_8B_GGUF)
    assert max_gpu_context(llama, 5 * GiB, 80 * GiB, ROCM) == 8192
    # A model that doesn't say isn't promised more than the target.
    assert max_gpu_context(replace(llama, context_length=None), 5 * GiB, 80 * GiB, ROCM) == TARGET_CONTEXT


def test_no_context_when_even_the_shortest_doesnt_fit() -> None:
    llama = shape(LLAMA_3_8B_GGUF)
    shortest = estimate(llama, 5 * GiB, MIN_FIT_CONTEXT, ROCM).total + FIT_MARGIN_MIB * MiB
    assert max_gpu_context(llama, 5 * GiB, shortest, ROCM) == MIN_FIT_CONTEXT
    assert max_gpu_context(llama, 5 * GiB, shortest - 1, ROCM) is None


def test_a_state_space_model_gets_its_whole_context_or_nothing() -> None:
    mamba = shape(MAMBA_GGUF)
    needed = estimate(mamba, 2 * GiB, MIN_FIT_CONTEXT, ROCM).total + FIT_MARGIN_MIB * MiB
    assert max_gpu_context(mamba, 2 * GiB, needed, ROCM) == 1048576
    assert max_gpu_context(mamba, 2 * GiB, needed - 1, ROCM) is None

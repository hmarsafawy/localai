"""Rating a model's options with the memory estimate: what the rating is made
of, how it moves with the context, and the file-size fallback for a model
whose layers couldn't be read."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from fastapi.testclient import TestClient

import routes.huggingface_router as huggingface_router
import services.huggingface as huggingface_module
from main import app
from models.hf_model import HfModelRequirements, HfRequirementRole, HfRoleOption
from services.huggingface import HuggingFaceService
from services.memory_estimate import (
    FIT_MARGIN_MIB,
    VISION_OVERHEAD_BYTES,
    MiB,
    RuntimeTraits,
    estimate,
)
from services.model_options import (
    FILE_SIZE_FIT_NOTE,
    MemoryBudget,
    annotate_fit,
    attach_options,
)
from services.model_shape import ModelShape, shape_from_gguf
from tests.test_model_shape import LLAMA_3_8B_GGUF, QWEN35_9B_MTP_GGUF

GiB = 1024 * MiB


def shape_of(metadata: dict) -> ModelShape:
    shape = shape_from_gguf(metadata)
    assert shape is not None
    return shape


LLAMA_3_8B = shape_of(LLAMA_3_8B_GGUF)
# Llama 3.1 8B has Llama 3's layers, trained for 128K instead of 8K.
LLAMA_31_8B = replace(LLAMA_3_8B, context_length=131072)
# The sizes of a typical Llama 3.1 8B repo's files.
LLAMA_FILES = {
    "Model-Q4_K_M.gguf": 4_920_000_000,
    "Model-Q6_K.gguf": 6_600_000_000,
    "Model-Q8_0.gguf": 8_540_000_000,
    "Model-F16.gguf": 16_070_000_000,
}
MMPROJ = {"mmproj-F16.gguf": 876_000_000, "mmproj-BF16.gguf": 876_000_000}
ROCM_12_GIB = MemoryBudget(True, 12 * GiB, 32 * GiB, backend="rocm")


def requirements_for(
    shape: ModelShape | None, files: dict[str, int] = LLAMA_FILES, vision: dict[str, int] | None = None
) -> HfModelRequirements:
    roles = [HfRequirementRole(role="language_model", present=True, files=list(files))]
    if vision:
        roles.append(HfRequirementRole(role="vision_encoder", required=False, present=True, files=list(vision)))
    requirements = HfModelRequirements(model_id="repo", roles=roles)
    attach_options(requirements, {**files, **(vision or {})})
    for option in requirements.roles[0].options:
        option.shape = shape
    return requirements


def option_of(requirements: HfModelRequirements, option_id: str, role: int = 0) -> HfRoleOption:
    return next(option for option in requirements.roles[role].options if option.id == option_id)


def recommended(requirements: HfModelRequirements) -> str | None:
    return next((option.id for option in requirements.roles[0].options if option.recommended), None)


def test_each_option_says_what_its_memory_is_made_of() -> None:
    requirements = requirements_for(LLAMA_31_8B)
    annotate_fit(requirements, ROCM_12_GIB)
    option = option_of(requirements, "Q4_K_M")
    memory = option.memory
    assert memory is not None
    assert memory.context == 16384
    assert memory.weights_bytes == 4_920_000_000
    # 32 layers x 8 KV heads x 128 x (K + V) x 2 bytes = 128 KiB a token.
    assert memory.kv_cache_bytes == 16384 * 128 * 1024
    assert memory.recurrent_state_bytes == 0
    assert memory.vision_bytes == 0
    assert memory.runtime_bytes == 384 * MiB
    assert memory.total_bytes == (
        memory.weights_bytes + memory.kv_cache_bytes + memory.compute_bytes + memory.runtime_bytes
    )
    assert option.estimated_memory_bytes == memory.total_bytes
    # The layers are only for working it out, not part of the API.
    assert "shape" not in option.model_dump()
    assert any("16,384 tokens" in note for note in requirements.notes)
    assert FILE_SIZE_FIT_NOTE not in requirements.notes


@pytest.mark.parametrize(
    ("context_size", "expected"),
    [
        # 12 GiB, less the 1 GiB margin and ROCm's 384 MiB, leaves 10,880 MiB.
        # At 16K, Q8_0 needs 8,144 MiB of weights + 2,048 of KV cache + 134 of
        # compute = 10,326.
        (None, "Q8_0"),
        # At 32K its KV cache doubles: 12,390. Q6_K needs 6,294 + 4,096 + 150 = 10,540.
        (32768, "Q6_K"),
        # At 64K even Q4_K_M needs 4,692 + 8,192 + 182 = 13,066, so it's run
        # partly on the CPU, and Q4_K_M is what's recommended for that.
        (65536, "Q4_K_M"),
    ],
)
def test_the_recommendation_follows_the_context(context_size: int | None, expected: str) -> None:
    requirements = requirements_for(LLAMA_31_8B)
    annotate_fit(requirements, ROCM_12_GIB, context_size)
    assert recommended(requirements) == expected


def test_file_size_alone_would_have_said_q8_0_fits_at_any_context() -> None:
    requirements = requirements_for(None)
    annotate_fit(requirements, ROCM_12_GIB, 65536)
    assert recommended(requirements) == "Q8_0"
    assert option_of(requirements, "Q8_0").memory is None


def test_at_64k_nothing_fits_the_gpu_but_it_all_runs_partly_on_the_cpu() -> None:
    requirements = requirements_for(LLAMA_31_8B)
    annotate_fit(requirements, ROCM_12_GIB, 65536)
    assert {option.id: option.fit for option in requirements.roles[0].options} == {
        "Q4_K_M": "partial_gpu",
        "Q6_K": "partial_gpu",
        "Q8_0": "partial_gpu",
        "F16": "partial_gpu",
    }


def test_the_context_is_never_more_than_the_model_was_trained_for() -> None:
    requirements = requirements_for(LLAMA_3_8B)
    annotate_fit(requirements, ROCM_12_GIB, 32768)
    assert all(option.memory and option.memory.context == 8192 for option in requirements.roles[0].options)
    assert any("8,192 tokens" in note for note in requirements.notes)


def test_max_gpu_context_is_the_longest_that_still_fits() -> None:
    requirements = requirements_for(LLAMA_31_8B)
    annotate_fit(requirements, ROCM_12_GIB)
    traits = RuntimeTraits("rocm")
    longest: dict[str, int | None] = {}
    for option in requirements.roles[0].options:
        assert option.memory is not None
        context = option.max_gpu_context
        longest[option.id] = context
        if context is None:
            continue
        weights = option.memory.weights_bytes
        assert estimate(LLAMA_31_8B, weights, context, traits).fits(12 * GiB)
        assert not estimate(LLAMA_31_8B, weights, context + 256, traits).fits(12 * GiB)
        # Rated at 16K, an option fits the GPU exactly when 16K is within its longest.
        assert (option.fit == "fits_gpu") == (context >= option.memory.context)
    assert longest["F16"] is None  # Not even 4K: its weights alone are 15,325 MiB.
    q8_0, q6_k, q4_k_m = (longest[key] or 0 for key in ("Q8_0", "Q6_K", "Q4_K_M"))
    assert 0 < q8_0 < q6_k < q4_k_m


def test_the_vision_encoder_downloaded_with_it_is_counted_unless_left_out() -> None:
    requirements = requirements_for(LLAMA_31_8B, vision=MMPROJ)
    annotate_fit(requirements, ROCM_12_GIB)
    assert option_of(requirements, "Q8_0").memory.vision_bytes == 876_000_000 + VISION_OVERHEAD_BYTES  # type: ignore[union-attr]
    assert option_of(requirements, "F16", role=1).recommended
    # With the encoder Q8_0 no longer fits at 16K.
    assert recommended(requirements) == "Q6_K"

    without = requirements_for(LLAMA_31_8B, vision=MMPROJ)
    annotate_fit(without, ROCM_12_GIB, with_vision=False)
    assert option_of(without, "Q8_0").memory.vision_bytes == 0  # type: ignore[union-attr]
    assert recommended(without) == "Q8_0"


@pytest.mark.parametrize(
    ("budget", "runtime"),
    [
        (MemoryBudget(True, 12 * GiB, 32 * GiB, backend="vulkan"), 192 * MiB),
        (MemoryBudget(True, 12 * GiB, 32 * GiB, backend="cuda"), 384 * MiB),
        (MemoryBudget(False, None, 32 * GiB, backend="cpu"), 0),
        (None, 384 * MiB),
    ],
)
def test_the_backend_sets_the_runtimes_own_share(budget: MemoryBudget | None, runtime: int) -> None:
    requirements = requirements_for(LLAMA_31_8B)
    annotate_fit(requirements, budget)
    assert option_of(requirements, "Q4_K_M").memory.runtime_bytes == runtime  # type: ignore[union-attr]


Q4_K_M = {"Model-Q4_K_M.gguf": 4_920_000_000}


def rate_q4(budget: MemoryBudget) -> HfRoleOption:
    requirements = requirements_for(LLAMA_31_8B, files=Q4_K_M)
    annotate_fit(requirements, budget)
    return requirements.roles[0].options[0]


def test_on_the_cpu_there_is_no_gpu_context_and_no_margin() -> None:
    needed = estimate(LLAMA_31_8B, 4_920_000_000, 16384, RuntimeTraits("cpu")).total
    option = rate_q4(MemoryBudget(False, None, needed, backend="cpu"))
    assert option.fit == "cpu_only"
    assert option.max_gpu_context is None
    assert rate_q4(MemoryBudget(False, None, needed - 1, backend="cpu")).fit == "too_large"


def test_a_fit_on_the_gpu_leaves_llama_cpps_margin_free() -> None:
    needed = estimate(LLAMA_31_8B, 4_920_000_000, 16384, RuntimeTraits("rocm")).total + FIT_MARGIN_MIB * MiB
    assert rate_q4(MemoryBudget(True, needed, 32 * GiB, backend="rocm")).fit == "fits_gpu"
    assert rate_q4(MemoryBudget(True, needed - 1, 32 * GiB, backend="rocm")).fit == "partial_gpu"
    # Moving layers off still leaves the margin free on the GPU, so RAM has to hold the rest.
    assert rate_q4(MemoryBudget(True, needed - GiB, GiB, backend="rocm")).fit == "partial_gpu"
    assert rate_q4(MemoryBudget(True, needed - GiB, GiB - 1, backend="rocm")).fit == "too_large"


def test_an_mtp_build_leaves_out_the_layers_llama_cpp_doesnt_load() -> None:
    qwen = shape_of(QWEN35_9B_MTP_GGUF)
    requirements = requirements_for(qwen, files={"Qwen3.5-9B-MTP-Q4_K_M.gguf": 6_600_000_000})
    annotate_fit(requirements, ROCM_12_GIB)
    memory = requirements.roles[0].options[0].memory
    assert memory is not None
    # 32 layers run of the 33 stored.
    assert memory.weights_bytes == 6_600_000_000 * 32 // 33
    assert memory.recurrent_state_bytes > 0


def test_a_model_whose_layers_arent_known_is_judged_from_its_file_size() -> None:
    requirements = requirements_for(None, files={"Model-Q4_K_M.gguf": 10**10})
    annotate_fit(requirements, MemoryBudget(True, 11 * 10**9, 32 * 10**9, backend="rocm"))
    option = requirements.roles[0].options[0]
    assert option.memory is None and option.max_gpu_context is None
    assert option.estimated_memory_bytes == 12 * 10**9
    assert requirements.notes == [FILE_SIZE_FIT_NOTE]


# Reading the layers from a GGUF repo


@pytest.fixture
def reads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every GGUF whose metadata is read, standing in for the Hub: an MTP build
    gets Qwen3.5's, anything else Llama 3's, and a name with `broken` fails."""
    read: list[str] = []

    def read_gguf_metadata(token: str | None, repo_id: str, filename: str) -> dict[str, Any]:
        read.append(filename)
        if "broken" in filename:
            raise OSError("connection reset")
        return QWEN35_9B_MTP_GGUF if "mtp" in filename else LLAMA_3_8B_GGUF

    monkeypatch.setattr(huggingface_module, "read_gguf_metadata", read_gguf_metadata)
    return read


def gguf_repo(*names: str) -> HfModelRequirements:
    requirements = HfModelRequirements(
        model_id="repo", roles=[HfRequirementRole(role="language_model", present=True, files=list(names))]
    )
    attach_options(requirements, {name: (position + 1) * 10**9 for position, name in enumerate(names)})
    return requirements


def test_a_models_layers_are_read_once_per_variant(reads: list[str]) -> None:
    service = HuggingFaceService(api=object())  # type: ignore[arg-type]
    requirements = gguf_repo("Model-Q4_K_M.gguf", "Model-Q8_0.gguf", "MTP/mtp-Model-Q4_K_M.gguf")
    service._attach_shapes("repo", requirements)
    shapes = {option.id: option.shape for option in requirements.roles[0].options}
    assert shapes["Q4_K_M"] == shapes["Q8_0"] == LLAMA_3_8B
    assert shapes["mtp-Q4_K_M"] == shape_of(QWEN35_9B_MTP_GGUF)
    assert sorted(reads) == ["MTP/mtp-Model-Q4_K_M.gguf", "Model-Q4_K_M.gguf"]


def test_a_split_model_is_read_from_its_first_shard(reads: list[str]) -> None:
    service = HuggingFaceService(api=object())  # type: ignore[arg-type]
    requirements = gguf_repo("Model-Q8_0-00002-of-00002.gguf", "Model-Q8_0-00001-of-00002.gguf")
    service._attach_shapes("repo", requirements)
    assert reads == ["Model-Q8_0-00001-of-00002.gguf"]
    assert requirements.roles[0].options[0].shape == LLAMA_3_8B


def test_metadata_read_lately_isnt_read_again_but_a_failed_read_is(reads: list[str]) -> None:
    service = HuggingFaceService(api=object())  # type: ignore[arg-type]
    for _ in range(3):
        requirements = gguf_repo("Model-Q4_K_M.gguf", "Model-broken-Q8_0.gguf")
        service._attach_shapes("repo", requirements)
    assert sorted(reads) == ["Model-Q4_K_M.gguf", *["Model-broken-Q8_0.gguf"] * 3]
    shapes = {option.id: option.shape for option in requirements.roles[0].options}
    assert shapes == {"Q4_K_M": LLAMA_3_8B, "broken-Q8_0": None}


class FakeRequirements:
    def get_model_requirements(self, model_id: str, *args: Any, **kwargs: Any) -> HfModelRequirements:
        return requirements_for(LLAMA_31_8B)


def test_the_requirements_route_rates_for_the_context_asked_for(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(huggingface_router, "service", FakeRequirements())
    monkeypatch.setattr(huggingface_router.llama_runtime_service, "memory_budget", lambda: ROCM_12_GIB)
    client = TestClient(app)

    body = client.get("/models/meta/llama/requirements", params={"context_size": 32768}).json()
    options = {option["id"]: option for option in body["roles"][0]["options"]}
    assert options["Q6_K"]["recommended"] is True
    assert options["Q6_K"]["memory"]["context"] == 32768
    assert options["Q6_K"]["memory"]["kv_cache_bytes"] == 4 * GiB
    assert options["Q6_K"]["max_gpu_context"] >= 32768
    assert "shape" not in options["Q6_K"]

    default = client.get("/models/meta/llama/requirements").json()
    assert next(option["id"] for option in default["roles"][0]["options"] if option["recommended"]) == "Q8_0"
    assert client.get("/models/meta/llama/requirements", params={"context_size": 100}).status_code == 422

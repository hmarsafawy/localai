"""Filenames here are real ones, taken from popular GGUF repos on the Hub."""

from __future__ import annotations

import struct

import pytest

from models.hf_model import HfModelRequirements, HfRequirementRole
from services.model_options import (
    FILE_SIZE_FIT_NOTE,
    MemoryBudget,
    annotate_fit,
    attach_options,
    build_options,
    classify_fit,
    parse_quantization,
    quantization_from_file_type,
    selection_for,
    unparsed_gguf_paths,
)
from services.weight_headers import GGUF_METADATA_FETCH_BYTES, parse_gguf_metadata

GB = 10**9


def sizes_for(*paths: str, size: int = GB) -> dict[str, int | None]:
    return {path: size for path in paths}


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("Qwen3.8-27B-Uncensored-Q4_K_M.gguf", "Q4_K_M"),
        ("Qwen3.8-27B-Q4_0.gguf", "Q4_0"),
        ("Qwen3.8-27B-Q4_1.gguf", "Q4_1"),
        ("Qwen3.8-27B-UD-IQ3_XXS.gguf", "UD-IQ3_XXS"),
        ("Qwen3.8-27B-UD-Q4_K_XL.gguf", "UD-Q4_K_XL"),
        ("Qwen3.8-27B-Uncensored-IQ2_M.gguf", "IQ2_M"),
        ("IQ4_NL.gguf", "IQ4_NL"),
        ("Huihui-Qwen3.8-27B-abliterated-Ternary-Bonsai-PTQ1_0.gguf", "PTQ1_0"),
        ("Huihui-Qwen3.8-27B-abliterated-Ternary-Bonsai-PQ2_0.gguf", "PQ2_0"),
        ("kitten-tts2-20261021-tq2_1.gguf", "TQ2_1"),
        ("Qwen3.8-27B-Q8_K_P.gguf", "Q8_K_P"),
        ("mmproj-Qwen3.8-27B-BF16.gguf", "BF16"),
        ("mmproj-F16.gguf", "F16"),
        ("SheetSage2-F32.gguf", "F32"),
        ("Qwen3.8-27B-iMatrix-NVFP4-MTP.gguf", "NVFP4"),
        ("Qwen3.6-35B-A3B-MXFP4_MOE.gguf", "MXFP4_MOE"),
        ("qwen-image-2.1-UC-fp8.safetensors", "fp8"),
        ("qwen-image-2.1-UC-int8_convrot.safetensors", "int8_convrot"),
        ("qwen3vl_8b_int8_convrot.safetensors", "int8_convrot"),
        ("qwen-image-2.1-UC-MLX-4bit.safetensors", "MLX-4bit"),
        ("qwen_image_2.1_vae_bf16.safetensors", "BF16"),
        ("apollo-orig.gguf", "orig"),
        ("locate-anything-q4_k.gguf", "Q4_K"),
        ("BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf", "BF16"),
        ("diffusion_pytorch_model.fp16.safetensors", "fp16"),
        ("diffusion_pytorch_model-00001-of-00003.fp16.safetensors", "fp16"),
        ("Q4_K_M/model-00001-of-00002.gguf", "Q4_K_M"),
    ],
)
def test_parse_quantization(path: str, expected: str) -> None:
    assert parse_quantization(path) == expected


@pytest.mark.parametrize(
    "path",
    [
        "Qwen3.8-27B.gguf",
        "model.safetensors",
        # A folder only counts when that's all its name is.
        "MiniMax-H3-Q4-GGUF/dit.gguf",
        "Laguna-XS-2.1-APEX-Balanced.gguf",
        # Run-together names: not the token, so don't pretend to read it.
        "DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-Q8Attn-chat-v2.gguf",
    ],
)
def test_parse_quantization_refuses_to_guess(path: str) -> None:
    assert parse_quantization(path) is None


def test_shards_collapse_into_one_option() -> None:
    paths = [
        "BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf",
        "BF16/Qwen3.8-27B-BF16-00002-of-00002.gguf",
        "Qwen3.8-27B-Q4_0.gguf",
    ]
    options = build_options(paths, {paths[0]: 30 * GB, paths[1]: 25 * GB, paths[2]: 16 * GB})
    assert [option.id for option in options] == ["Q4_0", "BF16"]
    bf16 = options[1]
    assert bf16.shards == 2
    assert bf16.size_bytes == 55 * GB
    assert bf16.files == paths[:2]


def test_safetensors_shards_keep_their_index_and_configs() -> None:
    paths = [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "config.json",
    ]
    options = build_options(paths, sizes_for(*paths))
    assert len(options) == 1
    assert options[0].shards == 2
    assert set(options[0].files) == set(paths)
    assert options[0].format == "safetensors"


def test_precision_variants_of_shards_stay_apart() -> None:
    paths = [
        "diffusion_pytorch_model-00001-of-00002.safetensors",
        "diffusion_pytorch_model-00002-of-00002.safetensors",
        "diffusion_pytorch_model.safetensors.index.json",
        "diffusion_pytorch_model-00001-of-00002.fp16.safetensors",
        "diffusion_pytorch_model-00002-of-00002.fp16.safetensors",
        "diffusion_pytorch_model.safetensors.index.fp16.json",
        "config.json",
    ]
    options = build_options(paths, sizes_for(*paths))
    assert len(options) == 2
    fp16 = next(option for option in options if option.quantization == "fp16")
    assert "diffusion_pytorch_model.safetensors.index.fp16.json" in fp16.files
    assert "diffusion_pytorch_model.safetensors.index.json" not in fp16.files
    # A config belongs to both choices.
    assert all("config.json" in option.files for option in options)


def test_variants_are_what_differs_between_names() -> None:
    paths = [
        "Qwen3.8-27B-Uncensored-Q4_K_M.gguf",
        "Qwen3.8-27B-Uncensored-draft-Q4_0.gguf",
        "Qwen3.8-27B-Uncensored-noMTP-Q4_K_M.gguf",
    ]
    options = {option.id: option for option in build_options(paths, sizes_for(*paths))}
    assert set(options) == {"Q4_K_M", "draft-Q4_0", "noMTP-Q4_K_M"}
    assert options["Q4_K_M"].variant is None
    assert options["draft-Q4_0"].variant == "draft"
    assert options["noMTP-Q4_K_M"].label == "Q4_K_M (noMTP)"


def test_variant_can_be_a_prefix() -> None:
    paths = ["MTP/mtp-Qwen3.8-27B-Q4_0.gguf", "Qwen3.8-27B-Q4_0.gguf"]
    options = {option.id: option for option in build_options(paths, sizes_for(*paths))}
    assert set(options) == {"mtp-Q4_0", "Q4_0"}
    assert options["mtp-Q4_0"].variant == "mtp"
    assert options["Q4_0"].variant is None


def test_variant_can_be_a_suffix_after_the_quantization() -> None:
    paths = ["Huihui-abliterated-GSQ-RCO-IQ3_S-mtp.gguf", "Huihui-abliterated-GSQ-RCO-IQ3_S.gguf"]
    variants = {option.id: option.variant for option in build_options(paths, sizes_for(*paths))}
    assert variants == {"mtp-IQ3_S": "mtp", "IQ3_S": None}


def test_several_models_in_one_repo_are_told_apart() -> None:
    paths = ["minimax_h3_fl2va_pruned-Q2_K.gguf", "minimax_h3_ref2va_pruned-Q2_K.gguf"]
    ids = {option.id for option in build_options(paths, sizes_for(*paths))}
    assert ids == {"fl2va-Q2_K", "ref2va-Q2_K"}


def test_same_name_in_two_folders_stays_distinct() -> None:
    paths = ["base/model.gguf", "turbo/model.gguf"]
    ids = [option.id for option in build_options(paths, sizes_for(*paths))]
    assert sorted(ids) == ["base", "turbo"]


def test_one_quantization_in_two_formats_stays_distinct() -> None:
    paths = ["model-fp16.safetensors", "model-fp16.onnx"]
    ids = [option.id for option in build_options(paths, sizes_for(*paths))]
    assert len(set(ids)) == 2


def test_openvino_pair_is_one_option() -> None:
    paths = ["openvino_model.xml", "openvino_model.bin"]
    options = build_options(paths, sizes_for(*paths))
    assert len(options) == 1
    assert options[0].format == "openvino"


def test_role_without_weights_is_a_single_download_all() -> None:
    paths = ["vocab.json", "merges.txt"]
    options = build_options(paths, sizes_for(*paths))
    assert len(options) == 1
    assert options[0].id == "default"
    assert options[0].files == paths


def test_empty_role_has_no_options() -> None:
    assert build_options([], {}) == []


def test_unknown_weight_size_makes_the_total_unknown() -> None:
    # A total missing the main file would look like a real, far too small size.
    options = build_options(["model.gguf", "config.json"], {"model.gguf": None, "config.json": 100})
    assert options[0].size_bytes is None


def test_metadata_fills_in_a_name_that_says_nothing() -> None:
    options = build_options(["Laguna-XS-2.1-APEX-Balanced.gguf"], {"Laguna-XS-2.1-APEX-Balanced.gguf": GB},
                            {"Laguna-XS-2.1-APEX-Balanced.gguf": "Q4_K_M"})
    assert options[0].quantization == "Q4_K_M"


@pytest.mark.parametrize(
    ("file_type", "expected"),
    [(15, "Q4_K_M"), (32, "BF16"), (15 + 1024, "Q4_K_M"), (37, "TQ2_0"), (39, "NVFP4"), (999, None), ("15", None)],
)
def test_quantization_from_file_type(file_type: object, expected: str | None) -> None:
    assert quantization_from_file_type(file_type) == expected


def test_only_first_shards_of_unnamed_ggufs_get_probed() -> None:
    requirements = HfModelRequirements(
        model_id="repo",
        roles=[
            HfRequirementRole(
                role="language_model",
                present=True,
                files=[
                    "custom-00001-of-00002.gguf",
                    "custom-00002-of-00002.gguf",
                    "Named-Q4_K_M.gguf",
                    "model.safetensors",
                ],
            )
        ],
    )
    assert unparsed_gguf_paths(requirements) == ["custom-00001-of-00002.gguf"]


@pytest.mark.parametrize(
    ("role", "required", "option_count", "expected"),
    [
        ("language_model", True, 5, "one"),
        ("vae", True, 1, "all"),
        ("vision_encoder", False, 2, "optional_one"),
        ("vision_encoder", False, 1, "optional_one"),
        ("lora", False, 3, "any"),
    ],
)
def test_selection(role: str, required: bool, option_count: int, expected: str) -> None:
    paths = [f"model-Q{index}_0.gguf" for index in range(option_count)]
    entry = HfRequirementRole(role=role, required=required, present=True, files=paths)
    entry.options = build_options(paths, sizes_for(*paths))
    assert selection_for(entry) == expected


def test_subfolder_roles_get_full_repo_paths() -> None:
    requirements = HfModelRequirements(
        model_id="repo",
        roles=[HfRequirementRole(role="vae", present=True, subfolder="vae", files=["model.safetensors"])],
    )
    attach_options(requirements, {"vae/model.safetensors": GB})
    assert requirements.roles[0].options[0].files == ["vae/model.safetensors"]
    assert requirements.roles[0].options[0].size_bytes == GB


def test_missing_role_has_no_options() -> None:
    requirements = HfModelRequirements(
        model_id="repo", roles=[HfRequirementRole(role="vae", present=False, source="base_model")]
    )
    attach_options(requirements, {})
    assert requirements.roles[0].options == []


@pytest.mark.parametrize(
    ("required", "budget", "expected"),
    [
        (10 * GB, MemoryBudget(True, 16 * GB, 32 * GB), "fits_gpu"),
        (16 * GB, MemoryBudget(True, 16 * GB, 32 * GB), "fits_gpu"),
        (17 * GB, MemoryBudget(True, 16 * GB, 32 * GB), "partial_gpu"),
        (48 * GB, MemoryBudget(True, 16 * GB, 32 * GB), "partial_gpu"),
        (49 * GB, MemoryBudget(True, 16 * GB, 32 * GB), "too_large"),
        (10 * GB, MemoryBudget(False, None, 32 * GB), "cpu_only"),
        (33 * GB, MemoryBudget(False, None, 32 * GB), "too_large"),
        (10 * GB, MemoryBudget(True, None, 32 * GB), None),
        (17 * GB, MemoryBudget(True, 16 * GB, None), None),
        (10 * GB, MemoryBudget(True, 16 * GB, None), "fits_gpu"),
        (10 * GB, MemoryBudget(False, None, None), None),
    ],
)
def test_classify_fit(required: int, budget: MemoryBudget, expected: str | None) -> None:
    assert classify_fit(required, budget) == expected


def language_model(files: dict[str, int]) -> HfModelRequirements:
    requirements = HfModelRequirements(
        model_id="repo",
        roles=[HfRequirementRole(role="language_model", present=True, files=list(files))],
    )
    attach_options(requirements, dict(files))
    return requirements


def recommended_ids(requirements: HfModelRequirements) -> list[str]:
    return [option.id for role in requirements.roles for option in role.options if option.recommended]


TYPICAL_REPO = {
    "Model-Q2_K.gguf": 5 * GB,
    "Model-Q4_K_M.gguf": 10 * GB,
    "Model-Q6_K.gguf": 15 * GB,
    "Model-Q8_0.gguf": 20 * GB,
    "Model-F16.gguf": 40 * GB,
}


def test_recommends_the_largest_that_fits_the_gpu() -> None:
    requirements = language_model(TYPICAL_REPO)
    annotate_fit(requirements, MemoryBudget(True, 19 * GB, 32 * GB))
    assert recommended_ids(requirements) == ["Q6_K"]


def test_never_recommends_more_than_q8_even_if_full_precision_fits() -> None:
    requirements = language_model(TYPICAL_REPO)
    annotate_fit(requirements, MemoryBudget(True, 100 * GB, 32 * GB))
    assert recommended_ids(requirements) == ["Q8_0"]


def test_recommends_q4_k_m_when_nothing_fits_the_gpu_fully() -> None:
    requirements = language_model(TYPICAL_REPO)
    annotate_fit(requirements, MemoryBudget(True, 2 * GB, 32 * GB))
    assert recommended_ids(requirements) == ["Q4_K_M"]


def test_recommends_q4_k_m_on_cpu_only() -> None:
    requirements = language_model(TYPICAL_REPO)
    annotate_fit(requirements, MemoryBudget(False, None, 32 * GB))
    assert recommended_ids(requirements) == ["Q4_K_M"]


def test_q4_k_m_is_preferred_over_other_q4_k_when_a_repo_only_has_ud_builds() -> None:
    requirements = language_model(
        {"Model-UD-Q4_K_S.gguf": 9 * GB, "Model-UD-Q4_K_M.gguf": 10 * GB, "Model-UD-Q4_K_XL.gguf": 11 * GB}
    )
    annotate_fit(requirements, None)
    assert recommended_ids(requirements) == ["UD-Q4_K_M"]


def test_recommends_smallest_runnable_without_any_q4_k() -> None:
    requirements = language_model({"Model-Q2_K.gguf": 5 * GB, "Model-Q8_0.gguf": 20 * GB})
    annotate_fit(requirements, MemoryBudget(True, 2 * GB, 32 * GB))
    assert recommended_ids(requirements) == ["Q2_K"]


def test_recommends_nothing_when_nothing_can_run() -> None:
    requirements = language_model(TYPICAL_REPO)
    annotate_fit(requirements, MemoryBudget(False, None, 2 * GB))
    assert recommended_ids(requirements) == []


def test_without_hardware_falls_back_to_q4_k_m_and_leaves_fit_null() -> None:
    requirements = language_model(TYPICAL_REPO)
    annotate_fit(requirements, None)
    assert recommended_ids(requirements) == ["Q4_K_M"]
    assert all(option.fit is None for option in requirements.roles[0].options)
    assert FILE_SIZE_FIT_NOTE not in requirements.notes


def test_tagged_variants_arent_recommended_when_a_plain_one_exists() -> None:
    requirements = language_model(
        {"Model-draft-Q8_0.gguf": 3 * GB, "Model-Q4_K_M.gguf": 10 * GB, "Model-noMTP-Q5_K_M.gguf": 12 * GB}
    )
    annotate_fit(requirements, MemoryBudget(True, 100 * GB, 32 * GB))
    assert recommended_ids(requirements) == ["Q4_K_M"]


def test_gguf_beats_other_formats_for_llama_cpp() -> None:
    requirements = language_model({"Model-Q4_K_M.gguf": 10 * GB, "Model-fp8.safetensors": 12 * GB})
    annotate_fit(requirements, MemoryBudget(True, 100 * GB, 32 * GB))
    assert recommended_ids(requirements) == ["Q4_K_M"]


def test_fit_includes_overhead_and_says_its_an_estimate() -> None:
    requirements = language_model({"Model-Q4_K_M.gguf": 10 * GB})
    # 10 GB fits in 11 GB of VRAM on its own, but not once the overhead is counted.
    annotate_fit(requirements, MemoryBudget(True, 11 * GB, 32 * GB))
    option = requirements.roles[0].options[0]
    assert option.estimated_memory_bytes == 12 * GB
    assert option.fit == "partial_gpu"
    assert FILE_SIZE_FIT_NOTE in requirements.notes


def test_fit_is_only_worked_out_for_language_models() -> None:
    requirements = HfModelRequirements(
        model_id="repo",
        roles=[HfRequirementRole(role="transformer", present=True, files=["Model-Q4_K_M.gguf"])],
    )
    attach_options(requirements, {"Model-Q4_K_M.gguf": GB})
    annotate_fit(requirements, MemoryBudget(True, 100 * GB, 32 * GB))
    option = requirements.roles[0].options[0]
    assert option.fit is None
    assert option.recommended is False


def test_vision_encoder_prefers_f16() -> None:
    requirements = HfModelRequirements(
        model_id="repo",
        roles=[
            HfRequirementRole(
                role="vision_encoder", required=False, present=True, files=["mmproj-BF16.gguf", "mmproj-F16.gguf"]
            )
        ],
    )
    attach_options(requirements, {"mmproj-BF16.gguf": GB, "mmproj-F16.gguf": GB})
    annotate_fit(requirements, None)
    assert recommended_ids(requirements) == ["F16"]


def gguf_bytes(*entries: tuple[str, int, bytes]) -> bytes:
    """A GGUF header in the real on-disk format: (key, value type, encoded value)."""
    data = b"GGUF" + struct.pack("<IQQ", 3, 0, len(entries))
    for key, value_type, value in entries:
        data += struct.pack("<Q", len(key)) + key.encode() + struct.pack("<I", value_type) + value
    return data


def string_array(count: int, length: int) -> bytes:
    item = struct.pack("<Q", length) + b"x" * length
    return struct.pack("<IQ", 8, count) + item * count


FILE_TYPE_ENTRY = ("general.file_type", 4, struct.pack("<I", 15))


def test_file_type_before_the_tokenizer_can_be_read_from_a_small_prefix() -> None:
    """How it's laid out in an mmproj or an MTP head."""
    data = gguf_bytes(FILE_TYPE_ENTRY, ("tokenizer.ggml.tokens", 9, string_array(50_000, 8)))
    metadata = parse_gguf_metadata(data[:GGUF_METADATA_FETCH_BYTES])
    assert quantization_from_file_type(metadata.get("general.file_type")) == "Q4_K_M"


def test_file_type_after_the_tokenizer_is_out_of_reach_of_a_small_prefix() -> None:
    """How it's laid out in a quantized LLM. A small read has to come back with
    no answer here, rather than an error or a wrong one, so that the option
    falls back to its variant text."""
    data = gguf_bytes(("tokenizer.ggml.tokens", 9, string_array(50_000, 8)), FILE_TYPE_ENTRY)
    metadata = parse_gguf_metadata(data[:GGUF_METADATA_FETCH_BYTES])
    assert metadata.get("general.file_type") is None

    paths = ["Laguna-XS-2.1-APEX-Balanced.gguf", "Laguna-XS-2.1-APEX-Compact.gguf"]
    options = {option.id: option for option in build_options(paths, sizes_for(*paths))}
    assert options["Balanced"].quantization is None
    assert options["Balanced"].label == "Balanced"

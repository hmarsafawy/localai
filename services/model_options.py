"""Turns a role's flat list of files into the choices a user actually has: one
entry per quantization (or precision, or variant), with shards collapsed into
the option they belong to, and an estimate of whether each one fits this machine.

Pure functions over filenames and sizes - nothing here touches the network.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from models.hf_model import HfModelRequirements, HfRequirementRole, HfRoleOption

WEIGHT_FORMATS_BY_EXTENSION = {
    ".safetensors": "safetensors",
    ".bin": "pytorch",
    ".pt": "pytorch",
    ".pth": "pytorch",
    ".ckpt": "pytorch",
    ".msgpack": "flax",
    ".onnx": "onnx",
    ".onnx_data": "onnx",
    ".xml": "openvino",
    ".gguf": "gguf",
}

# llama.cpp reserves room for its context and buffers on top of the weights.
MEMORY_OVERHEAD_FACTOR = 1.2

FIT_NOTE = (
    "Memory fit is a rough estimate from file size and what's free right now, not a measurement: "
    "the context size and other apps change what actually fits"
)

_QUANT_BODY = r"""
    (?:I|P|PT|T)?Q\d+(?:_(?:K(?:_[A-Z0-9]+)?|\d+(?:_\d+)*|NL|XXS|XS|S|M|L))?
    | NVFP4 | MXFP4(?:_MOE)? | BF16 | F8_E4M3 | F16 | F32 | F64 | FP8 | FP16 | FP32
    | INT8(?:_CONVROT)? | (?:MLX-)?\d+BIT | ORIG
"""
# A quantization must stand as its own token: `Qwen3` and `F16HC` are not ones.
_QUANT_TOKEN = re.compile(
    rf"(?:^|(?<=[-_.\s]))(?P<ud>UD-)?(?P<quant>{_QUANT_BODY})(?=$|[-_.\s])",
    re.IGNORECASE | re.VERBOSE,
)
_QUANT_ONLY = re.compile(rf"(?:UD-)?(?:{_QUANT_BODY})", re.IGNORECASE | re.VERBOSE)

_SHARD_MARKER = re.compile(r"-\d{5}-of-\d{5}(?=\.|$)", re.IGNORECASE)
# `model.safetensors.index.json`, or diffusers' `model.safetensors.index.fp16.json`.
_SHARD_INDEX = re.compile(r"^(?P<stem>.+)\.(?P<ext>safetensors|bin)\.index(?P<variant>\.[^.]+)?\.json$", re.IGNORECASE)
_TOKEN_SPLIT = re.compile(r"([-_.\s]+)")

# `general.file_type` values, from llama.cpp's `llama_ftype` in llama.h.
GGUF_FILE_TYPES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    7: "Q8_0",
    8: "Q5_0",
    9: "Q5_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
    19: "IQ2_XXS",
    20: "IQ2_XS",
    21: "Q2_K_S",
    22: "IQ3_XS",
    23: "IQ3_XXS",
    24: "IQ1_S",
    25: "IQ4_NL",
    26: "IQ3_S",
    27: "IQ3_M",
    28: "IQ2_S",
    29: "IQ2_M",
    30: "IQ4_XS",
    31: "IQ1_M",
    32: "BF16",
    36: "TQ1_0",
    37: "TQ2_0",
    38: "MXFP4_MOE",
    39: "NVFP4",
    40: "Q1_0",
    41: "Q2_0",
    42: "F8_E4M3",
}
# Added to `file_type` when the value was inferred rather than stored.
GGUF_FILE_TYPE_GUESSED = 1024

# Not worth recommending over a real quantization: these are the source weights.
_FULL_PRECISION = {"F16", "BF16", "F32", "F64", "FP16", "FP32", "ORIG"}


class OptionChoiceError(ValueError):
    """A requested option doesn't match, or matches more than one, of those on offer."""


@dataclass(frozen=True)
class MemoryBudget:
    """What a model can use right now. Built by `services.hardware`."""

    has_gpu: bool
    vram_free_bytes: int | None
    ram_available_bytes: int | None


@dataclass
class _Group:
    """The weight files of one option, before it gets its final name."""

    folder: str
    stem: str
    format: str | None
    weights: list[str]
    index_files: list[str]
    quantization: str | None = None
    residual: str = ""
    variant: str | None = None


def _canonical_quantization(token: str) -> str:
    upper = token.upper()
    if upper.startswith(("FP", "INT")) or upper == "ORIG":
        return token.lower()
    if upper.endswith("BIT"):
        return upper[:-3] + "bit"
    return upper


def _basename(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def _extension(path: str) -> str:
    base = _basename(path)
    dot = base.rfind(".")
    return base[dot:].lower() if dot > 0 else ""


def _weight_stem(path: str) -> str:
    """The filename with its extension and any shard numbering removed, so a
    shard, its siblings and their index all reduce to the same stem."""
    base = _basename(path)
    index = _SHARD_INDEX.match(base)
    if index is not None:
        return index["stem"] + (index["variant"] or "")
    dot = base.rfind(".")
    stem = base[:dot] if dot > 0 else base
    # `model-00001-of-00004.fp16.safetensors` has it mid-name.
    return _SHARD_MARKER.sub("", stem)


def _split_quantization(path: str) -> tuple[str | None, str]:
    """(quantization, the stem without it). The last match wins: names put the
    model first, and that's where a false match could come from."""
    stem = _weight_stem(path)
    match = None
    for match in _QUANT_TOKEN.finditer(stem):
        pass
    if match is None:
        folder = path.rpartition("/")[0].rsplit("/", 1)[-1]
        if folder and _QUANT_ONLY.fullmatch(folder):
            return _with_ud_prefix(folder), stem
        return None, stem
    quantization = _canonical_quantization(match["quant"])
    if match["ud"]:
        quantization = "UD-" + quantization
    remainder = stem[: match.start()].rstrip("-_. ") + "-" + stem[match.end() :].lstrip("-_. ")
    return quantization, remainder.strip("-_. ")


def _with_ud_prefix(folder: str) -> str:
    if folder.upper().startswith("UD-"):
        return "UD-" + _canonical_quantization(folder[3:])
    return _canonical_quantization(folder)


def parse_quantization(path: str) -> str | None:
    """The quantization a filename claims, or None - never a guess. Reads the
    basename first, then a parent folder named only for one (`BF16/...`)."""
    return _split_quantization(path)[0]


def quantization_from_file_type(file_type: object) -> str | None:
    if not isinstance(file_type, int):
        return None
    return GGUF_FILE_TYPES.get(file_type & ~GGUF_FILE_TYPE_GUESSED)


def _format_of(path: str, openvino_stems: set[tuple[str, str]]) -> str | None:
    base = _basename(path)
    index = _SHARD_INDEX.match(base)
    if index is not None:
        return "safetensors" if index["ext"].lower() == "safetensors" else "pytorch"
    extension = _extension(path)
    if extension == ".bin" and (path.rpartition("/")[0], _weight_stem(path)) in openvino_stems:
        return "openvino"
    return WEIGHT_FORMATS_BY_EXTENSION.get(extension)


def _common_affix_lengths(token_lists: list[list[str]]) -> tuple[int, int]:
    """How many leading and trailing tokens every name shares, not overlapping."""
    if not token_lists:
        return 0, 0
    shortest = min(len(tokens) for tokens in token_lists)
    prefix = 0
    while prefix < shortest and len({tokens[prefix].casefold() for tokens in token_lists}) == 1:
        prefix += 1
    suffix = 0
    while suffix < shortest - prefix and len({tokens[-1 - suffix].casefold() for tokens in token_lists}) == 1:
        suffix += 1
    return prefix, suffix


def _assign_variants(groups: list[_Group]) -> None:
    """A group's variant is what's left of its name once the quantization and
    everything all the names share are gone: `draft`, `noMTP`, `fl2va_pruned`."""
    split = [_TOKEN_SPLIT.split(group.residual) if group.residual else [] for group in groups]
    # `re.split` with a capture alternates tokens and delimiters; compare tokens only.
    prefix, suffix = _common_affix_lengths([parts[0::2] for parts in split])
    for group, parts in zip(groups, split):
        end = len(parts) - suffix * 2
        remaining = "".join(parts[prefix * 2 : max(end, 0)]).strip("-_. ")
        group.variant = remaining or None


def _option_id(variant: str | None, quantization: str | None, stem: str) -> str:
    joined = "-".join(part for part in (variant, quantization) if part) or stem or "default"
    return re.sub(r"[^A-Za-z0-9._-]+", "-", joined)


def _label(group: _Group, fallback: str) -> str:
    if group.quantization and group.variant:
        return f"{group.quantization} ({group.variant})"
    return group.quantization or group.variant or fallback


def _unique_ids(groups: list[_Group]) -> list[str]:
    """Option ids must be distinct, but two options can read the same once the
    shared name is removed: `base/model.gguf` and `turbo/model.gguf`, or one
    quantization in two formats. Those are told apart by what actually differs."""
    ids = [_option_id(group.variant, group.quantization, group.stem) for group in groups]
    colliding = {identifier for identifier in ids if ids.count(identifier) > 1}
    if not colliding:
        return ids

    for position, group in enumerate(groups):
        if ids[position] not in colliding:
            continue
        rivals = [other for index, other in enumerate(groups) if ids[index] == ids[position]]
        distinction: list[str] = []
        if len({other.folder for other in rivals}) > 1 and group.folder:
            distinction.append(group.folder.rsplit("/", 1)[-1])
        if len({other.format for other in rivals}) > 1 and group.format:
            distinction.append(group.format)
        if distinction:
            group.variant = "-".join(filter(None, [group.variant, *distinction]))

    ids = [_option_id(group.variant, group.quantization, group.stem) for group in groups]
    seen: dict[str, int] = {}
    unique: list[str] = []
    for identifier in ids:
        seen[identifier] = seen.get(identifier, 0) + 1
        unique.append(identifier if seen[identifier] == 1 else f"{identifier}-{seen[identifier]}")
    return unique


def build_options(
    paths: list[str],
    sizes: dict[str, int | None],
    quantization_overrides: dict[str, str] | None = None,
) -> list[HfRoleOption]:
    """One option per distinct set of weights among a role's files (full repo
    paths). Non-weight files such as `config.json` belong to every option."""
    overrides = quantization_overrides or {}
    openvino_stems = {
        (path.rpartition("/")[0], _weight_stem(path)) for path in paths if _extension(path) == ".xml"
    }

    groups: dict[tuple[str, str, str | None], _Group] = {}
    shared: list[str] = []
    for path in paths:
        file_format = _format_of(path, openvino_stems)
        if file_format is None:
            shared.append(path)
            continue
        folder = path.rpartition("/")[0]
        stem = _weight_stem(path)
        group = groups.setdefault(
            (folder, stem.casefold(), file_format), _Group(folder, stem, file_format, [], [])
        )
        if _SHARD_INDEX.match(_basename(path)):
            group.index_files.append(path)
        else:
            group.weights.append(path)

    weighted = [group for group in groups.values() if group.weights]
    # An index with no weights next to it can't be chosen; keep it with the rest.
    shared.extend(path for group in groups.values() if not group.weights for path in group.index_files)

    if not weighted:
        if not paths:
            return []
        return [
            HfRoleOption(
                id="default",
                label="All files",
                files=list(paths),
                shards=0,
                size_bytes=_total_size(paths, sizes),
            )
        ]

    for group in weighted:
        group.weights.sort()
        group.quantization, group.residual = _split_quantization(group.weights[0])
        if group.quantization is None and group.weights[0] in overrides:
            group.quantization = overrides[group.weights[0]]

    _assign_variants(weighted)
    ids = _unique_ids(weighted)

    options = [
        HfRoleOption(
            id=identifier,
            label=_label(group, _SHARD_MARKER.sub("", _basename(group.weights[0]))),
            quantization=group.quantization,
            variant=group.variant,
            format=group.format,
            files=[*group.weights, *group.index_files, *shared],
            shards=len(group.weights),
            size_bytes=_total_size([*group.weights, *group.index_files, *shared], sizes, group.weights),
        )
        for identifier, group in zip(ids, weighted)
    ]
    options.sort(key=lambda option: (option.size_bytes is None, option.size_bytes or 0, option.id))
    return options


def _total_size(
    paths: Iterable[str], sizes: dict[str, int | None], required: Iterable[str] | None = None
) -> int | None:
    """None if a weight's size is unknown - a total missing the main file would
    look like a real, far too small number."""
    if any(sizes.get(path) is None for path in (required if required is not None else paths)):
        return None
    return sum(sizes.get(path) or 0 for path in paths)


def _role_paths(role: HfRequirementRole) -> list[str]:
    if role.subfolder:
        return [f"{role.subfolder}/{name}" for name in role.files]
    return list(role.files)


def selection_for(role: HfRequirementRole) -> str:
    if role.role.startswith("lora"):
        return "any"
    if not role.required:
        return "optional_one"
    return "one" if len(role.options) > 1 else "all"


def unparsed_gguf_paths(requirements: HfModelRequirements) -> list[str]:
    """Each GGUF whose name says nothing about its quantization and whose
    metadata should be asked instead. Only a first shard has any metadata."""
    unparsed: dict[str, None] = {}
    for role in requirements.roles:
        if not role.present:
            continue
        first_of_stem: dict[tuple[str, str], str] = {}
        for path in sorted(_role_paths(role)):
            if _extension(path) == ".gguf" and parse_quantization(path) is None:
                first_of_stem.setdefault((path.rpartition("/")[0], _weight_stem(path)), path)
        unparsed.update(dict.fromkeys(first_of_stem.values()))
    return list(unparsed)


def attach_options(
    requirements: HfModelRequirements,
    sizes: dict[str, int | None],
    quantization_overrides: dict[str, str] | None = None,
) -> None:
    """Fills in each role's choices. A role that's missing from this repo stays
    empty: where to get it is what its `sources` already describe."""
    for role in requirements.roles:
        role.options = (
            build_options(_role_paths(role), sizes, quantization_overrides) if role.present else []
        )
        role.selection = selection_for(role)


def classify_fit(required_bytes: int, budget: MemoryBudget) -> str | None:
    """None when the numbers needed to answer aren't known."""
    ram = budget.ram_available_bytes
    if budget.has_gpu:
        vram = budget.vram_free_bytes
        if vram is None:
            return None
        if required_bytes <= vram:
            return "fits_gpu"
        if ram is None:
            return None
        return "partial_gpu" if required_bytes <= vram + ram else "too_large"
    if ram is None:
        return None
    return "cpu_only" if required_bytes <= ram else "too_large"


def _is_language_model(role: HfRequirementRole) -> bool:
    return role.role.startswith("language_model")


def _is_vision_encoder(role: HfRequirementRole) -> bool:
    return role.role.startswith("vision_encoder")


def _is_q4k(option: HfRoleOption) -> bool:
    return bool(option.quantization) and option.quantization.removeprefix("UD-").startswith("Q4_K")


def _pick_language_model(options: list[HfRoleOption]) -> HfRoleOption | None:
    # llama.cpp only runs GGUF, and tagged variants (draft heads, MTP builds)
    # aren't what someone who just wants the model is after.
    pool = [option for option in options if option.format == "gguf"] or options
    pool = [option for option in pool if option.variant is None] or pool
    pool = [option for option in pool if (option.quantization or "").upper() not in _FULL_PRECISION] or pool

    def q4k_or_none(candidates: list[HfRoleOption]) -> HfRoleOption | None:
        exact = [
            option for option in candidates if (option.quantization or "").removeprefix("UD-") == "Q4_K_M"
        ]
        return (exact or [option for option in candidates if _is_q4k(option)] or [None])[0]

    if all(option.fit is None for option in pool):
        return q4k_or_none(pool)

    fits = [option for option in pool if option.fit == "fits_gpu" and option.size_bytes is not None]
    if fits:
        return max(fits, key=lambda option: option.size_bytes or 0)
    runnable = [option for option in pool if option.fit in ("partial_gpu", "cpu_only")]
    if not runnable:
        return None
    return q4k_or_none(runnable) or min(runnable, key=lambda option: option.size_bytes or 0)


def _pick_vision_encoder(options: list[HfRoleOption]) -> HfRoleOption | None:
    # F16 is what llama.cpp's own conversions default to, so it's the safe choice.
    for preferred in ("F16", "BF16"):
        match = next((option for option in options if option.quantization == preferred), None)
        if match is not None:
            return match
    return options[0] if options else None


def _gguf_options(
    requirements: HfModelRequirements, is_wanted: Callable[[HfRequirementRole], bool]
) -> list[HfRoleOption]:
    return [
        option
        for role in requirements.roles
        if is_wanted(role)
        for option in role.options
        if option.format == "gguf"
    ]


def language_model_options(requirements: HfModelRequirements) -> list[HfRoleOption]:
    """What llama.cpp can run as the model itself: GGUF only."""
    return _gguf_options(requirements, _is_language_model)


def vision_encoder_options(requirements: HfModelRequirements) -> list[HfRoleOption]:
    return _gguf_options(requirements, _is_vision_encoder)


def match_option(options: list[HfRoleOption], wanted: str) -> HfRoleOption:
    """The option someone means by `wanted`: its exact id (`mtp-Q4_0`), or else
    its quantization, with or without `UD-` (`Q4_K_M` finds `UD-Q4_K_M`).
    Raises `OptionChoiceError` rather than guess between several."""
    folded = wanted.casefold()
    exact = [option for option in options if option.id.casefold() == folded]
    if exact:
        return exact[0]

    name = folded.removeprefix("ud-")
    named = [
        option
        for option in options
        if option.quantization and option.quantization.casefold().removeprefix("ud-") == name
    ]
    if len(named) == 1:
        return named[0]
    if named:
        raise OptionChoiceError(
            f"'{wanted}' could mean any of these, so pick one: {', '.join(option.id for option in named)}"
        )
    raise OptionChoiceError(
        f"'{wanted}' isn't available. Available: {', '.join(option.id for option in options) or 'none'}"
    )


def default_option(options: list[HfRoleOption]) -> HfRoleOption | None:
    """The one to use when nobody chose: the recommended option, or the only one
    there is unless it's known not to fit. None if there's no sensible default."""
    recommended = next((option for option in options if option.recommended), None)
    if recommended is not None:
        return recommended
    if len(options) == 1 and options[0].fit != "too_large":
        return options[0]
    return None


def annotate_fit(requirements: HfModelRequirements, budget: MemoryBudget | None) -> None:
    """Adds the memory estimate and picks a default. Fit is only worked out for
    LLMs for now: a diffusion pipeline's memory is the sum of several roles.
    With no budget, `fit` stays null and the default falls back to Q4_K_M."""
    annotated = False
    for role in requirements.roles:
        if _is_language_model(role):
            for option in role.options:
                if option.size_bytes is None:
                    continue
                option.estimated_memory_bytes = math.ceil(option.size_bytes * MEMORY_OVERHEAD_FACTOR)
                if budget is not None:
                    option.fit = classify_fit(option.estimated_memory_bytes, budget)
                    annotated = annotated or option.fit is not None
            chosen = _pick_language_model(role.options)
        elif _is_vision_encoder(role):
            chosen = _pick_vision_encoder(role.options)
        else:
            continue
        if chosen is not None:
            chosen.recommended = True
    if annotated:
        requirements.notes.append(FIT_NOTE)

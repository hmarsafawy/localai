"""Turning a Hugging Face safetensors model into a GGUF that llama.cpp runs:
how big each quantization would come out, and running llama.cpp's converter
and `llama-quantize` with their progress reported.

The converter writes BF16, F16 or Q8_0 itself. Anything smaller is made from
its BF16 by `llama-quantize`, which is how llama.cpp's own quantizations are made.

A vision model's vision encoder is converted on its own, into the separate
`mmproj` GGUF that llama-server takes next to the model.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any

from models.hf_model import HfRoleOption
from services.jobs import JobContext
from services.llama_converter import ConverterCommand, converter_environment, run_command
from services.llama_runtime import LaunchableRuntime, runtime_environment

# Bits per weight, from `llama-quantize --help`'s sizes for Llama-3-8B (8.03B
# parameters). Small models come out bigger: tensors whose width isn't a
# multiple of 256 can't use K-quants and fall back to bigger types.
BITS_PER_WEIGHT = {
    "BF16": 16.0,
    "F16": 16.0,
    "Q8_0": 8.51,
    "Q6_K": 6.57,
    "Q5_K_M": 5.70,
    "Q5_K_S": 5.57,
    "Q4_K_M": 4.90,
    "Q4_K_S": 4.67,
    "Q4_0": 4.64,
    "Q3_K_L": 4.31,
    "Q3_K_M": 4.00,
    "Q3_K_S": 3.65,
    "Q2_K": 3.17,
}
# What the converter writes itself, by its `--outtype` name.
CONVERTER_OUTTYPES = {"BF16": "bf16", "F16": "f16", "Q8_0": "q8_0"}
INTERMEDIATE = "BF16"
# The tokenizer, chat template and other metadata, on top of the weights.
METADATA_BYTES = 20_000_000
# A vision encoder is only written by the converter: `llama-quantize` doesn't
# take one. It's small, so F16 is the default, as in llama.cpp's own repos.
VISION_QUANTIZATIONS = ("F16", "BF16", "Q8_0")
VISION_DEFAULT = "F16"
# An mmproj has no tokenizer, only a few settings for preparing images.
VISION_METADATA_BYTES = 1_000_000
# A big model takes a while on a slow disk. One that takes longer than this has hung.
STEP_TIMEOUT_SECONDS = 6 * 60 * 60
CONVERTER_TAIL_LINES = 5
TOKENIZER_CONFIG = "tokenizer_config.json"

# tqdm's `Writing:  21%|██        | 56.6M/269M [00:00<00:00, 238Mbyte/s]`, in SI units.
_WRITING = re.compile(
    r"Writing:\s+\d+%\|[^|]*\|\s*(?P<done>[\d.]+)(?P<done_unit>[kMGTP]?)/(?P<total>[\d.]+)(?P<total_unit>[kMGTP]?)"
)
_SI = {"": 1, "k": 1e3, "M": 1e6, "G": 1e9, "T": 1e12, "P": 1e15}
# `[ 263/ 272] blk.28.ffn_up.weight - [...], type = bf16, converting to q5_0 .. size = 1.69 MiB -> 0.58 MiB`
_QUANTIZED_TENSOR = re.compile(r"^\s*\[\s*(?P<index>\d+)/\s*(?P<count>\d+)\].*?\bsize\s*=\s*(?P<mib>[\d.]+)\s*MiB")


class ConversionError(Exception):
    """The converter or `llama-quantize` failed. Their message is in it."""


def estimated_size_bytes(parameters: int, quantization: str) -> int:
    return math.ceil(parameters * BITS_PER_WEIGHT[quantization] / 8) + METADATA_BYTES


def gguf_name(repo_id: str, quantization: str) -> str:
    """`SmolLM2-135M-Instruct-Q4_K_M.gguf`, as llama.cpp's own repos name them."""
    return f"{repo_id.rsplit('/', 1)[-1]}-{quantization}.gguf"


def conversion_options(repo_id: str, parameters: int) -> list[HfRoleOption]:
    """What a safetensors model can be converted to, with the size each would
    come out at, for choosing among like a GGUF repo's own quantizations."""
    return [
        HfRoleOption(
            id=quantization,
            label=quantization,
            quantization=quantization,
            format="gguf",
            files=[gguf_name(repo_id, quantization)],
            size_bytes=estimated_size_bytes(parameters, quantization),
        )
        for quantization in BITS_PER_WEIGHT
    ]


def vision_gguf_name(repo_id: str, quantization: str) -> str:
    """`mmproj-SmolVLM-256M-Instruct-F16.gguf`, as llama.cpp's own repos name them."""
    return f"mmproj-{gguf_name(repo_id, quantization)}"


def vision_conversion_options(repo_id: str, parameters: int | None) -> list[HfRoleOption]:
    """What a vision encoder can be converted to. Without its parameter count
    there's no size to give."""
    return [
        HfRoleOption(
            id=quantization,
            label=quantization,
            quantization=quantization,
            format="gguf",
            files=[vision_gguf_name(repo_id, quantization)],
            size_bytes=(
                math.ceil(parameters * BITS_PER_WEIGHT[quantization] / 8) + VISION_METADATA_BYTES
                if parameters
                else None
            ),
        )
        for quantization in VISION_QUANTIZATIONS
    ]


def _first_architecture(config: object) -> str | None:
    architectures = config.get("architectures") if isinstance(config, dict) else None
    if isinstance(architectures, list) and architectures and isinstance(architectures[0], str):
        return architectures[0]
    return None


def model_architecture(config: dict[str, Any]) -> str | None:
    """What the model is, as its `config.json` names it (`LlamaForCausalLM`)."""
    return _first_architecture(config) or _first_architecture(config.get("text_config"))


def text_architectures(config: dict[str, Any]) -> list[str]:
    """What llama.cpp's converter may convert the text model as, the likelier
    first. A vision model's text config names its language model, which is what
    the converter goes by (`VLlama3ForCausalLM` inside SmolVLM's
    `Idefics3ForConditionalGeneration`), except for a few it converts by the
    model's own name. Which of the two it is shows in the converter's list."""
    found = [_first_architecture(config.get("text_config")), _first_architecture(config)]
    return list(dict.fromkeys(name for name in found if name))


def vision_architecture(config: dict[str, Any]) -> str | None:
    """What llama.cpp's converter converts the vision encoder as: the vision
    config's own name for it if it has one, else the model's. None for a model
    whose config declares no vision encoder."""
    vision_config = config.get("vision_config")
    if not isinstance(vision_config, dict):
        return None
    return _first_architecture(vision_config) or model_architecture(config)


def parse_writing(line: str) -> tuple[int, int] | None:
    """The bytes written and to write, from a line of the converter's progress bar."""
    match = _WRITING.search(line)
    if match is None:
        return None
    done = float(match["done"]) * _SI[match["done_unit"]]
    total = float(match["total"]) * _SI[match["total_unit"]]
    return round(done), round(total)


def parse_quantized_tensor(line: str) -> tuple[int, int, int] | None:
    """The tensor's number, how many there are, and its size before quantizing."""
    match = _QUANTIZED_TENSOR.match(line)
    if match is None:
        return None
    return int(match["index"]), int(match["count"]), int(float(match["mib"]) * 1024**2)


def adapt_tokenizer_config(source: Path) -> bool:
    """Rewrites a tokenizer config saved by transformers 5 the way transformers
    4, which the converter runs, reads it. Says whether it changed anything.

    Transformers 5 names the tokenizer that loads `tokenizer.json`
    `TokenizersBackend`, where 4 names it `PreTrainedTokenizerFast`. It also lists
    special tokens under `extra_special_tokens`, where 4 expects a mapping, with
    the list under `additional_special_tokens`."""
    path = source / TOKENIZER_CONFIG
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(config, dict):
        return False

    changed = False
    if config.get("tokenizer_class") == "TokenizersBackend":
        config["tokenizer_class"] = "PreTrainedTokenizerFast"
        changed = True
    extra = config.get("extra_special_tokens")
    if isinstance(extra, list):
        del config["extra_special_tokens"]
        additional = list(config.get("additional_special_tokens") or [])
        additional += [token for token in extra if token not in additional]
        if additional:
            config["additional_special_tokens"] = additional
        changed = True
    if changed:
        path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return changed


def _failure(what: str, code: int, output: list[str]) -> ConversionError:
    useful = [line.strip() for line in output if line.strip() and not _WRITING.search(line)]
    detail = " / ".join(useful[-CONVERTER_TAIL_LINES:]) or "it printed nothing"
    return ConversionError(f"{what} failed (exit code {code}): {detail}")


def convert(
    context: JobContext,
    converter: ConverterCommand,
    source: Path,
    outfile: Path,
    quantization: str,
    vision: bool = False,
) -> None:
    """Converts the model in `source` to `outfile`, as BF16, F16 or Q8_0. With
    `vision`, it's the model's vision encoder that's converted, to an mmproj."""
    adapt_tokenizer_config(source)
    context.begin_transfer(None)

    def show_progress(line: str) -> None:
        if (written := parse_writing(line)) is not None:
            done, total = written
            context.update(bytes_total=total)
            context.progress(done)

    code, output = run_command(
        [
            str(converter.python),
            str(converter.script),
            str(source),
            "--outfile",
            str(outfile),
            "--outtype",
            CONVERTER_OUTTYPES[quantization],
            *(["--mmproj"] if vision else []),
        ],
        context,
        timeout=STEP_TIMEOUT_SECONDS,
        cwd=converter.script.parent,
        env=converter_environment(),
        on_line=show_progress,
    )
    if code != 0:
        raise _failure("llama.cpp's converter", code, output)


def quantize_binary(runtime: LaunchableRuntime) -> Path:
    return runtime.directory / ("llama-quantize.exe" if sys.platform == "win32" else "llama-quantize")


def quantize(context: JobContext, runtime: LaunchableRuntime, source: Path, outfile: Path, quantization: str) -> None:
    """Quantizes the GGUF `source` to `outfile` with the runtime's `llama-quantize`."""
    binary = quantize_binary(runtime)
    if not binary.is_file():
        raise ConversionError(f"{runtime.id} has no {binary.name}")
    context.begin_transfer(source.stat().st_size)
    done = 0

    def show_progress(line: str) -> None:
        nonlocal done
        if (tensor := parse_quantized_tensor(line)) is not None:
            done += tensor[2]
            context.progress(done)

    code, output = run_command(
        [str(binary), str(source), str(outfile), quantization],
        context,
        timeout=STEP_TIMEOUT_SECONDS,
        cwd=runtime.directory,
        env=runtime_environment(runtime.directory, runtime.backend),
        on_line=show_progress,
    )
    if code != 0:
        raise _failure("llama-quantize", code, output)

"""Turning a Hugging Face safetensors model into a GGUF that llama.cpp runs:
how big each quantization would come out, and running llama.cpp's converter
and `llama-quantize` with their progress reported.

The converter writes BF16, F16 or Q8_0 itself. Anything smaller is made from
its BF16 by `llama-quantize`, which is how llama.cpp's own quantizations are made.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

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
# A big model takes a while on a slow disk. One that takes longer than this has hung.
STEP_TIMEOUT_SECONDS = 6 * 60 * 60
CONVERTER_TAIL_LINES = 5

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


def _failure(what: str, code: int, output: list[str]) -> ConversionError:
    useful = [line.strip() for line in output if line.strip() and not _WRITING.search(line)]
    detail = " / ".join(useful[-CONVERTER_TAIL_LINES:]) or "it printed nothing"
    return ConversionError(f"{what} failed (exit code {code}): {detail}")


def convert(
    context: JobContext, converter: ConverterCommand, source: Path, outfile: Path, quantization: str
) -> None:
    """Converts the model in `source` to `outfile`, as BF16, F16 or Q8_0."""
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

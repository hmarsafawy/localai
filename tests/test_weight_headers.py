"""Reading a downloaded GGUF's metadata and tensor sizes, from files written in
the real on-disk layout: the header, the tensor table, then each tensor's data
padded to the alignment, as llama.cpp's writer lays them out."""

from __future__ import annotations

import struct
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from services.weight_headers import (
    GGUF_INITIAL_FETCH_BYTES,
    GgufArray,
    parse_gguf_file,
    parse_gguf_tensor_shapes,
    read_local_gguf,
)

MIB = 1024**2


def _encode(value: Any) -> tuple[int, bytes]:
    """A metadata value as GGUF stores it: its type, then its bytes."""
    if isinstance(value, bool):
        return 7, struct.pack("<?", value)
    if isinstance(value, int):
        return 4, struct.pack("<I", value)
    if isinstance(value, float):
        return 6, struct.pack("<f", value)
    if isinstance(value, str):
        encoded = value.encode()
        return 8, struct.pack("<Q", len(encoded)) + encoded
    if isinstance(value, tuple):
        return 9, struct.pack("<IQ", 5, len(value)) + struct.pack(f"<{len(value)}i", *value)
    if isinstance(value, GgufArray):  # A tokenizer's strings.
        token = struct.pack("<Q", 8) + b"<token8>"
        return 9, struct.pack("<IQ", 8, value.length) + token * value.length
    raise TypeError(value)


def gguf_file(
    metadata: Mapping[str, Any],
    tensors: Sequence[tuple[str, int]],
    alignment: int | None = None,
    table_order: Sequence[int] | None = None,
) -> bytes:
    """A whole GGUF holding `tensors` (name, bytes of data). The table can list
    them in another order than their data, which the format allows."""
    entries = dict(metadata)
    if alignment is not None:
        entries["general.alignment"] = alignment
    align = alignment or 32
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(entries))
    for key, value in entries.items():
        value_type, encoded = _encode(value)
        header += struct.pack("<Q", len(key)) + key.encode() + struct.pack("<I", value_type) + encoded
    offsets, offset = [], 0
    for _, size in tensors:
        offsets.append(offset)
        offset += -(-size // align) * align
    for index in table_order or range(len(tensors)):
        name, size = tensors[index]
        header += struct.pack("<Q", len(name)) + name.encode()
        header += struct.pack("<IQ", 1, size) + struct.pack("<IQ", 0, offsets[index])
    header += b"\0" * (-len(header) % align)
    return header + b"".join(b"\1" * size + b"\0" * (-size % align) for _, size in tensors)


TENSORS = [("token_embd.weight", 4096), ("blk.0.attn_q.weight", 1024), ("output.weight", 2048)]
METADATA = {"general.architecture": "llama", "llama.block_count": 1}


def test_each_tensor_takes_the_bytes_up_to_the_next() -> None:
    data = gguf_file(METADATA, TENSORS)
    gguf = parse_gguf_file(data, len(data))
    assert gguf.tensor_bytes == dict(TENSORS)
    assert gguf.metadata["llama.block_count"] == 1


def test_the_padding_after_a_tensor_counts_as_its_own() -> None:
    """Under the alignment, so the type's block size needn't be known."""
    data = gguf_file(METADATA, [("a", 1000), ("b", 40)])
    assert parse_gguf_file(data, len(data)).tensor_bytes == {"a": 1024, "b": 64}


def test_the_tensors_are_sized_in_the_order_of_their_data_not_the_table() -> None:
    data = gguf_file(METADATA, TENSORS, table_order=[2, 0, 1])
    assert parse_gguf_file(data, len(data)).tensor_bytes == dict(TENSORS)


def test_the_file_can_set_its_own_alignment() -> None:
    """Which moves where the data starts, as well as the padding."""
    data = gguf_file(METADATA, [("a", 100), ("b", 256)], alignment=256)
    assert parse_gguf_file(data, len(data)).tensor_bytes == {"a": 256, "b": 256}


def test_offsets_past_the_end_of_the_file_are_an_error() -> None:
    data = gguf_file(METADATA, TENSORS)
    with pytest.raises(ValueError, match="past the end"):
        parse_gguf_file(data, len(data) - 4096)


def test_the_remote_reader_still_gets_each_tensors_shape() -> None:
    """Shapes come out outermost-first, as PyTorch and safetensors give them."""
    assert parse_gguf_tensor_shapes(gguf_file(METADATA, TENSORS)) == {name: (size,) for name, size in TENSORS}


def test_a_header_bigger_than_the_first_read_is_read_whole(tmp_path: Path) -> None:
    """A big tokenizer comes before the tensor table: 248K tokens, here 4 MB."""
    path = tmp_path / "model.gguf"
    metadata = {**METADATA, "tokenizer.ggml.tokens": GgufArray(8, 248320)}
    path.write_bytes(gguf_file(metadata, TENSORS))
    assert path.stat().st_size > 4 * GGUF_INITIAL_FETCH_BYTES
    gguf = read_local_gguf([path])
    assert gguf.tensor_bytes == dict(TENSORS)
    assert gguf.metadata["tokenizer.ggml.tokens"] == GgufArray(8, 248320)


def test_a_split_model_adds_up_every_shards_tensors(tmp_path: Path) -> None:
    """The metadata is the first shard's: the others hold only `split.*` keys."""
    first, second = tmp_path / "model-00001-of-00002.gguf", tmp_path / "model-00002-of-00002.gguf"
    first.write_bytes(gguf_file({**METADATA, "split.count": 2}, TENSORS[:2]))
    second.write_bytes(gguf_file({"split.count": 2, "split.no": 1}, TENSORS[2:]))
    gguf = read_local_gguf([first, second])
    assert gguf.tensor_bytes == dict(TENSORS)
    assert gguf.metadata["general.architecture"] == "llama"


@pytest.mark.parametrize("keep", [3, 200])
def test_a_file_cut_off_in_its_header_is_an_error(tmp_path: Path, keep: int) -> None:
    path = tmp_path / "model.gguf"
    path.write_bytes(gguf_file(METADATA, TENSORS)[:keep])
    with pytest.raises(ValueError, match="truncated"):
        read_local_gguf([path])


def test_a_file_that_isnt_a_gguf_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "model.gguf"
    path.write_text('{"load_seconds": 0.1}')
    with pytest.raises(ValueError, match="not a GGUF"):
        read_local_gguf([path])

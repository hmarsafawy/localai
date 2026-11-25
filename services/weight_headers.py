"""Read a remote weight file's tensor names and shapes from its header alone,
via HTTP range requests - never downloading the weights themselves.

safetensors is handled by `HfApi.parse_safetensors_file_metadata`. GGUF has no
equivalent helper, so its header is parsed here: magic, version, tensor/kv
counts, the metadata key/values (skipped), then one entry per tensor holding
its name and dims. For a diffusion GGUF the whole header is a few tens of KB;
LLM GGUFs embed a tokenizer in the metadata and can run to several MB, so the
fetch grows until the header fits.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass

from huggingface_hub import HfApi, get_session, hf_hub_url
from huggingface_hub.utils import build_hf_headers, hf_raise_for_status


@dataclass(frozen=True)
class GgufArray:
    """An array whose items weren't kept (a tokenizer's strings, or numbers
    too many to be worth holding), or that the bytes read stop inside. Its
    length is known either way."""

    item_type: int
    length: int


TensorShapes = dict[str, tuple[int, ...]]
GgufMetadata = dict[str, str | int | float | bool | tuple[int | float | bool, ...] | GgufArray]

GGUF_MAGIC = b"GGUF"
GGUF_INITIAL_FETCH_BYTES = 256 * 1024
GGUF_MAX_FETCH_BYTES = 64 * 1024 * 1024
# Most `general.*` and architecture keys come before the tokenizer, so a small
# prefix holds them even when the full header runs to megabytes. The exception
# is `general.file_type`, which quantizing writes after the tokenizer: this only
# reaches it in files with no or a small tokenizer (mmproj, MTP heads).
GGUF_METADATA_FETCH_BYTES = 64 * 1024
# Numeric arrays up to this long are kept: per-layer values such as
# `head_count_kv` or the sliding-window pattern. Longer ones are the tokenizer's.
GGUF_KEPT_ARRAY_LENGTH = 1024

# GGUF metadata value types -> struct format. 8 (string) and 9 (array) are
# variable-length and handled separately.
_GGUF_SCALAR_FORMATS = {
    0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d",
}
_GGUF_STRING = 8
_GGUF_ARRAY = 9


class _NeedMoreBytes(Exception):
    pass


class _GgufCursor:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.offset = 0

    def take(self, size: int) -> bytes:
        end = self.offset + size
        if end > len(self.data):
            raise _NeedMoreBytes
        chunk = self.data[self.offset : end]
        self.offset = end
        return chunk

    def unpack(self, fmt: str) -> int | float | bool:
        return struct.unpack("<" + fmt, self.take(struct.calcsize("<" + fmt)))[0]

    def string(self) -> str:
        return self.take(int(self.unpack("Q"))).decode("utf-8", errors="replace")

    def numbers(self, item_type: int, count: int) -> tuple[int | float | bool, ...]:
        fmt = f"<{count}{_GGUF_SCALAR_FORMATS[item_type]}"
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))

    def skip_items(self, item_type: int, count: int) -> None:
        if item_type in _GGUF_SCALAR_FORMATS:
            self.take(count * struct.calcsize("<" + _GGUF_SCALAR_FORMATS[item_type]))
        else:
            for _ in range(count):
                self.skip_value(item_type)

    def skip_value(self, value_type: int) -> None:
        if value_type == _GGUF_STRING:
            self.take(int(self.unpack("Q")))
        elif value_type == _GGUF_ARRAY:
            item_type = int(self.unpack("I"))
            self.skip_items(item_type, int(self.unpack("Q")))
        elif value_type in _GGUF_SCALAR_FORMATS:
            self.take(struct.calcsize("<" + _GGUF_SCALAR_FORMATS[value_type]))
        else:
            raise ValueError(f"unknown GGUF metadata value type {value_type}")


def _read_gguf_preamble(cursor: _GgufCursor) -> tuple[int, int]:
    """Magic and version; returns (tensor count, metadata key/value count)."""
    if cursor.take(4) != GGUF_MAGIC:
        raise ValueError("not a GGUF file")
    version = cursor.unpack("I")
    if version < 2:
        # v1 used 32-bit counts/lengths throughout; it predates every GGUF on
        # the Hub worth fingerprinting.
        raise ValueError(f"unsupported GGUF version {version}")
    return int(cursor.unpack("Q")), int(cursor.unpack("Q"))


def parse_gguf_metadata(data: bytes) -> GgufMetadata:
    """Metadata values from the leading bytes of a GGUF file, as far as `data`
    reaches. Short numeric arrays are kept as tuples; any other array (the
    tokenizer) is a `GgufArray` giving only its length."""
    cursor = _GgufCursor(data)
    _, kv_count = _read_gguf_preamble(cursor)
    metadata: GgufMetadata = {}
    try:
        for _ in range(kv_count):
            key = cursor.string()
            value_type = int(cursor.unpack("I"))
            if value_type == _GGUF_STRING:
                metadata[key] = cursor.string()
            elif value_type in _GGUF_SCALAR_FORMATS:
                metadata[key] = cursor.unpack(_GGUF_SCALAR_FORMATS[value_type])
            elif value_type == _GGUF_ARRAY:
                item_type = int(cursor.unpack("I"))
                length = int(cursor.unpack("Q"))
                # Set before the items are read, so a prefix that stops inside
                # the tokenizer still says how many tokens there are.
                metadata[key] = GgufArray(item_type, length)
                if item_type in _GGUF_SCALAR_FORMATS and length <= GGUF_KEPT_ARRAY_LENGTH:
                    metadata[key] = cursor.numbers(item_type, length)
                else:
                    cursor.skip_items(item_type, length)
            else:
                raise ValueError(f"unknown GGUF metadata value type {value_type}")
    except _NeedMoreBytes:
        pass
    return metadata


def parse_gguf_tensor_shapes(data: bytes) -> TensorShapes:
    """Parse tensor names/shapes out of the leading bytes of a GGUF file.
    Raises `_NeedMoreBytes` if `data` stops before the tensor table does."""
    cursor = _GgufCursor(data)
    tensor_count, kv_count = _read_gguf_preamble(cursor)
    for _ in range(kv_count):
        cursor.string()
        cursor.skip_value(int(cursor.unpack("I")))

    shapes: TensorShapes = {}
    for _ in range(tensor_count):
        name = cursor.string()
        n_dims = int(cursor.unpack("I"))
        dims = [int(cursor.unpack("Q")) for _ in range(n_dims)]
        cursor.take(4 + 8)  # ggml type, data offset
        # GGUF stores dims innermost-first; reverse to match PyTorch/safetensors.
        shapes[name] = tuple(reversed(dims))
    return shapes


def read_tensor_shapes(api: HfApi, token: str | None, repo_id: str, filename: str) -> TensorShapes:
    extension = os.path.splitext(filename)[1].lower()
    if extension == ".safetensors":
        metadata = api.parse_safetensors_file_metadata(repo_id, filename)
        return {name: tuple(info.shape) for name, info in metadata.tensors.items()}
    if extension == ".gguf":
        return _read_gguf_tensor_shapes(token, repo_id, filename)
    raise ValueError(f"cannot read tensor names from {extension} files")


def read_gguf_metadata(token: str | None, repo_id: str, filename: str) -> GgufMetadata:
    """The leading metadata of a remote GGUF (`general.architecture`, the
    architecture's dimensions, ...) from one small range request - enough to
    tell which model a file holds without its multi-MB tensor table."""
    return parse_gguf_metadata(_fetch_prefix(token, repo_id, filename, GGUF_METADATA_FETCH_BYTES))


def _fetch_prefix(token: str | None, repo_id: str, filename: str, size: int) -> bytes:
    headers = build_hf_headers(token=token)
    headers["Range"] = f"bytes=0-{size - 1}"
    response = get_session().get(hf_hub_url(repo_id, filename), headers=headers)
    hf_raise_for_status(response)
    return response.content


def _read_gguf_tensor_shapes(token: str | None, repo_id: str, filename: str) -> TensorShapes:
    size = GGUF_INITIAL_FETCH_BYTES
    while True:
        data = _fetch_prefix(token, repo_id, filename, size)
        try:
            return parse_gguf_tensor_shapes(data)
        except _NeedMoreBytes:
            # A response shorter than requested means we already have the
            # whole file, so the header is genuinely truncated.
            if len(data) < size or size >= GGUF_MAX_FETCH_BYTES:
                raise ValueError("GGUF header is truncated or larger than the fetch limit") from None
            size = min(size * 4, GGUF_MAX_FETCH_BYTES)

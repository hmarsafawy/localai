"""A repo with only safetensors, for real: the app sets up llama.cpp's converter
(PyTorch included), converts the model to GGUF and quantizes it, and the result
is run and asked questions. A vision model's vision encoder is converted too,
and has to see what's in an image.

The converted model is the same one `test_e2e_model.py` downloads ready-made, so
it has to know the same things. A converter that mangles the weights or the
tokenizer gives garbage instead. Setting the converter up the first time takes
a few minutes and about 1 GB, so this runs on its own (`-m conversion`)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx
import pytest

from .helpers import (
    CONVERSION_SECONDS,
    CONVERTED_QUANTIZATION,
    SAFETENSORS_REPO,
    VISION_REPO,
    chat,
    chat_about_image,
    delete_model,
    launch_server,
    run_job,
    solid_png,
    stop_server,
)

pytestmark = [pytest.mark.e2e, pytest.mark.conversion]


@pytest.fixture(scope="module")
def converted(client: httpx.Client, runtime: dict[str, Any]) -> Iterator[dict[str, Any]]:
    body = {"repo_id": SAFETENSORS_REPO, "quantization": CONVERTED_QUANTIZATION}
    yield run_job(client, "/downloads", "/downloads", CONVERSION_SECONDS, body)
    delete_model(client, SAFETENSORS_REPO)


@pytest.fixture(scope="module")
def converted_server(client: httpx.Client, converted: dict[str, Any]) -> Iterator[dict[str, Any]]:
    server = launch_server(client, SAFETENSORS_REPO, CONVERTED_QUANTIZATION)
    yield server
    stop_server(client, server)


def test_the_converter_is_set_up_for_the_active_runtime(client: httpx.Client, converted: dict[str, Any]) -> None:
    converters = client.get("/runtimes/llama.cpp/converter").json()
    assert converters["ready"]
    assert "LlamaForCausalLM" in converters["converters"][0]["text_architectures"]


def test_only_the_converted_gguf_is_kept(client: httpx.Client, converted: dict[str, Any]) -> None:
    model = next(item for item in client.get("/library").json()["models"] if item["repo_id"] == SAFETENSORS_REPO)
    [option] = model["quantizations"]
    assert option["quantization"] == CONVERTED_QUANTIZATION
    assert option["converted"] and option["status"] == "complete"
    # The safetensors it was made from are deleted once it's done.
    assert all(name.endswith(".gguf") for name in option["files"])


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What is the capital of France?", "paris"),
        ("What color is the sky on a clear day?", "blue"),
    ],
)
def test_the_converted_model_answers_what_it_knows(
    client: httpx.Client, converted_server: dict[str, Any], question: str, expected: str
) -> None:
    assert expected in chat(client, converted_server["model_name"], question).lower()


@pytest.fixture(scope="module")
def converted_vision(client: httpx.Client, runtime: dict[str, Any]) -> Iterator[dict[str, Any]]:
    body = {"repo_id": VISION_REPO, "quantization": CONVERTED_QUANTIZATION}
    yield run_job(client, "/downloads", "/downloads", CONVERSION_SECONDS, body)
    delete_model(client, VISION_REPO)


@pytest.fixture(scope="module")
def vision_server(client: httpx.Client, converted_vision: dict[str, Any]) -> Iterator[dict[str, Any]]:
    server = launch_server(client, VISION_REPO, CONVERTED_QUANTIZATION)
    yield server
    stop_server(client, server)


def test_a_vision_models_encoder_is_converted_with_it(client: httpx.Client, converted_vision: dict[str, Any]) -> None:
    assert converted_vision["result"]["vision_encoder"] == "F16"
    model = next(item for item in client.get("/library").json()["models"] if item["repo_id"] == VISION_REPO)
    [vision] = model["vision_encoders"]
    assert vision["converted"] and vision["status"] == "complete"
    assert vision["files"] == ["mmproj-SmolVLM-256M-Instruct-F16.gguf"]


def test_the_vision_model_runs_with_its_vision_encoder(vision_server: dict[str, Any]) -> None:
    assert vision_server["vision_encoder"] == "F16"


@pytest.mark.parametrize(("colour", "rgb"), [("red", (220, 20, 20)), ("blue", (20, 40, 220))])
def test_the_converted_vision_encoder_sees_the_image(
    client: httpx.Client, vision_server: dict[str, Any], colour: str, rgb: tuple[int, int, int]
) -> None:
    # Two images, so an answer that ignored the image couldn't be right for both.
    question = "What color is this image? Answer in one word."
    assert colour in chat_about_image(client, vision_server["model_name"], solid_png(rgb), question).lower()

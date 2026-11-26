"""A user's first run, for real: llama.cpp is installed, a GGUF model is
downloaded, run and asked questions, then stopped and deleted (by the fixtures,
whose teardown fails the run if either goes wrong).

The questions don't grade the model. A 135M model knows little, but what it
knows it says the same way every time, so a wrong or garbled answer means
something between the download and the reply broke it: a corrupt file, the
wrong chat template, or the proxy changing what it passes on."""

from __future__ import annotations

import json
import sys
from typing import Any

import httpx
import pytest

from .helpers import AUTH, GGUF_QUANTIZATION, GGUF_REPO, chat, chat_body

pytestmark = pytest.mark.e2e


def test_the_machine_is_read_for_real(client: httpx.Client) -> None:
    hardware = client.get("/system/hardware").json()
    assert hardware["os"]["system"] == {"win32": "windows", "darwin": "macos"}.get(sys.platform, "linux")
    assert hardware["cpu"]["logical_cores"] and hardware["cpu"]["logical_cores"] > 0
    assert hardware["memory"]["total_bytes"] > 1024**3
    assert hardware["storage"], "no disks were found"


def test_the_runtime_installed_is_one_that_runs_here(runtime: dict[str, Any]) -> None:
    assert runtime["compatible"] and runtime["verified"], runtime.get("verification_error")
    assert runtime["server_path"]


def test_requirements_recommend_an_option_that_fits(client: httpx.Client) -> None:
    requirements = client.get(f"/models/{GGUF_REPO}/requirements").json()
    options = [option for role in requirements["roles"] for option in role["options"]]
    assert any(option["quantization"] == GGUF_QUANTIZATION for option in options)
    recommended = [option for option in options if option["recommended"]]
    assert len(recommended) == 1
    assert recommended[0]["fit"] != "too_large"
    # Sized from the model's own layers: SmolLM2 was trained for 8K, so that's
    # what it's rated at rather than the 16K default.
    assert recommended[0]["memory"]["context"] == 8192


def test_the_download_is_in_the_library_whole(client: httpx.Client, gguf_model: dict[str, Any]) -> None:
    model = next(item for item in client.get("/library").json()["models"] if item["repo_id"] == GGUF_REPO)
    option = next(item for item in model["quantizations"] if item["quantization"] == GGUF_QUANTIZATION)
    assert option["status"] == "complete"
    assert option["bytes_on_disk"] == option["size_bytes"] > 50 * 1024**2


def test_a_running_model_is_listed_by_v1(client: httpx.Client, gguf_server: dict[str, Any]) -> None:
    models = client.get("/v1/models", headers=AUTH).json()["data"]
    assert gguf_server["model_name"] in [model["id"] for model in models]


def test_v1_turns_away_a_request_without_the_key(client: httpx.Client, gguf_server: dict[str, Any]) -> None:
    response = client.post("/v1/chat/completions", json=chat_body(gguf_server["model_name"], "Hi"))
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_api_key"


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What is the capital of France?", "paris"),
        ("What color is the sky on a clear day?", "blue"),
    ],
)
def test_the_model_answers_what_it_knows(
    client: httpx.Client, gguf_server: dict[str, Any], question: str, expected: str
) -> None:
    assert expected in chat(client, gguf_server["model_name"], question).lower()


def test_the_same_question_gets_the_same_answer(client: httpx.Client, gguf_server: dict[str, Any]) -> None:
    question = "Name three fruits."
    answers = {chat(client, gguf_server["model_name"], question) for _ in range(3)}
    assert len(answers) == 1, answers


def test_a_streamed_answer_is_the_same_as_a_whole_one(client: httpx.Client, gguf_server: dict[str, Any]) -> None:
    question = "What is the capital of France?"
    whole = chat(client, gguf_server["model_name"], question)
    pieces: list[str] = []
    body = chat_body(gguf_server["model_name"], question, stream=True)
    with client.stream("POST", "/v1/chat/completions", headers=AUTH, json=body) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        for line in response.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                delta = json.loads(line.removeprefix("data: "))["choices"][0]["delta"]
                pieces.append(delta.get("content") or "")
    assert "".join(pieces) == whole


def test_the_app_passes_on_exactly_what_llama_server_says(
    client: httpx.Client, gguf_server: dict[str, Any]
) -> None:
    body = chat_body(gguf_server["model_name"], "Name three fruits.")
    through_app = client.post("/v1/chat/completions", headers=AUTH, json=body).json()
    direct = httpx.post(f"{gguf_server['base_url']}/v1/chat/completions", json=body, timeout=120).json()
    assert through_app["choices"][0]["message"] == direct["choices"][0]["message"]


def test_plain_completions_go_through_too(client: httpx.Client, gguf_server: dict[str, Any]) -> None:
    body = {"model": gguf_server["model_name"], "prompt": "The capital of France is", "max_tokens": 8, "temperature": 0}
    response = client.post("/v1/completions", headers=AUTH, json=body)
    assert response.status_code == 200, response.text
    assert "paris" in response.json()["choices"][0]["text"].lower()

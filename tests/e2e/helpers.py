"""What the end-to-end tests do through the app's API, the way a client would."""

from __future__ import annotations

import json
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

API_KEY = "e2e-test-key"
AUTH = {"Authorization": f"Bearer {API_KEY}"}
# A 135M-parameter model, about 100 MB at Q4_K_M: small enough to download and
# run on a CPU in seconds, big enough to answer simple questions.
GGUF_REPO = "bartowski/SmolLM2-135M-Instruct-GGUF"
GGUF_QUANTIZATION = "Q4_K_M"
# The same model as safetensors only, about 270 MB, which the app converts.
SAFETENSORS_REPO = "HuggingFaceTB/SmolLM2-135M-Instruct"
CONVERTED_QUANTIZATION = "Q8_0"
# A small context keeps the memory it needs, and so its load time, down.
CONTEXT_SIZE = 2048

APP_START_SECONDS = 60
INSTALL_SECONDS = 15 * 60
DOWNLOAD_SECONDS = 15 * 60
# Setting the converter up downloads PyTorch, about 1 GB, the first time.
CONVERSION_SECONDS = 40 * 60
SERVER_START_SECONDS = 5 * 60
FINISHED = {"completed", "failed", "cancelled"}


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_for_health(base_url: str, process: subprocess.Popen[bytes], log_path: Path) -> None:
    deadline = time.monotonic() + APP_START_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            pytest.fail(f"The app exited with {process.returncode}:\n{tail(log_path)}")
        try:
            if httpx.get(f"{base_url}/health", timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    pytest.fail(f"The app didn't answer within {APP_START_SECONDS}s:\n{tail(log_path)}")


def tail(path: Path, lines: int = 60) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return "(no log)"


def follow_job(client: httpx.Client, path: str, timeout: float) -> dict[str, Any]:
    """Follows a job's live progress (`.../events`) to its end, and returns how
    it finished. A job still going at `timeout` is cancelled and fails the test."""
    deadline = time.monotonic() + timeout
    last: dict[str, Any] | None = None
    # A message comes at least every 15 s, even while nothing changes.
    with client.stream("GET", f"{path}/events", timeout=httpx.Timeout(60, connect=10)) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line.startswith("data: "):
                last = json.loads(line.removeprefix("data: "))
            if time.monotonic() > deadline:
                client.delete(path)
                pytest.fail(f"{path} was still at {last and last.get('stage')} after {timeout}s: {last}")
    assert last is not None, f"{path}/events sent nothing"
    assert last["status"] in FINISHED, last
    return last


def run_job(client: httpx.Client, path: str, jobs_path: str, timeout: float, body: dict[str, Any]) -> dict[str, Any]:
    """Starts a job with a POST to `path`, and follows it under `jobs_path` until it's done."""
    response = client.post(path, json=body)
    assert response.status_code == 202, response.text
    job = follow_job(client, f"{jobs_path}/{response.json()['id']}", timeout)
    assert job["status"] == "completed", f"{job['kind']} {job['status']}: {job.get('error')}\n{job.get('notes')}"
    return job


def launch_server(client: httpx.Client, repo_id: str, quantization: str) -> dict[str, Any]:
    body = {"repo_id": repo_id, "quantization": quantization, "context_size": CONTEXT_SIZE}
    response = client.post("/servers", json=body)
    assert response.status_code == 202, response.text
    server = response.json()
    deadline = time.monotonic() + SERVER_START_SECONDS
    while server["status"] == "starting" and time.monotonic() < deadline:
        time.sleep(1)
        server = client.get(f"/servers/{server['id']}").json()
    if server["status"] != "ready":
        logs = client.get(f"/servers/{server['id']}/logs", params={"lines": 80}).json()["lines"]
        pytest.fail(f"llama-server is {server['status']}: {server.get('error')}\n" + "\n".join(logs))
    return server


def stop_server(client: httpx.Client, server: dict[str, Any]) -> None:
    stopped = client.delete(f"/servers/{server['id']}")
    assert stopped.status_code == 200, stopped.text
    assert stopped.json()["status"] == "stopped"


def delete_model(client: httpx.Client, repo_id: str) -> None:
    deleted = client.delete(f"/library/{repo_id}")
    assert deleted.status_code == 204, deleted.text
    assert all(model["repo_id"] != repo_id for model in client.get("/library").json()["models"])


def chat_body(model: str, question: str, **extra: Any) -> dict[str, Any]:
    """One question, answered greedily, so the same question always gets the same answer."""
    return {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "temperature": 0,
        "seed": 1,
        "max_tokens": 48,
        # llama-server reuses what it worked out for a prompt it has seen. That
        # changes the order of the GPU's sums, and on a near-tie a different
        # word wins, so a repeated question can get another answer.
        "cache_prompt": False,
        **extra,
    }


def chat(client: httpx.Client, model: str, question: str) -> str:
    response = client.post("/v1/chat/completions", headers=AUTH, json=chat_body(model, question))
    assert response.status_code == 200, response.text
    return response.json()["choices"][0]["message"]["content"]

"""End-to-end tests: the real app, run as its own process, doing for real what
a user would. It installs llama.cpp from GitHub, downloads a small model from
Hugging Face, runs it and talks to it. Nothing is faked, so these need the
internet and take minutes, and they only run when asked for (`-m e2e`).

Everything goes in a folder of its own: a temporary one, unless
LOCALAI_E2E_DATA_DIR names one to keep, which saves installing llama.cpp again
on the next run. Your own runtimes and models are never touched."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from .helpers import (
    API_KEY,
    DOWNLOAD_SECONDS,
    GGUF_QUANTIZATION,
    GGUF_REPO,
    INSTALL_SECONDS,
    delete_model,
    free_port,
    launch_server,
    run_job,
    stop_server,
    wait_for_health,
)

APP_DIR = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    configured = os.getenv("LOCALAI_E2E_DATA_DIR")
    if configured:
        path = Path(configured).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    return tmp_path_factory.mktemp("localai-e2e")


@pytest.fixture(scope="session")
def client(data_dir: Path) -> Iterator[httpx.Client]:
    """The app, started the way a user would start it, and a client for it."""
    port = free_port()
    env = {
        **os.environ,
        "LOCALAI_DATA_DIR": str(data_dir),
        # Set as well, so a models folder configured in .env isn't used instead.
        "LOCALAI_MODELS_DIR": str(data_dir / "model-files"),
        "LOCALAI_API_KEY": API_KEY,
    }
    log_path = data_dir / "app.log"
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", str(port)],
            cwd=APP_DIR,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    base_url = f"http://127.0.0.1:{port}"
    try:
        wait_for_health(base_url, process, log_path)
        with httpx.Client(base_url=base_url, timeout=httpx.Timeout(120, connect=10)) as http:
            yield http
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()


@pytest.fixture(scope="session")
def runtime(client: httpx.Client) -> dict[str, Any]:
    """llama.cpp's stable release, in the build the app picks for this machine."""
    run_job(client, "/runtimes/llama.cpp/install", "/runtimes/llama.cpp/jobs", INSTALL_SECONDS, {})
    installed = client.get("/runtimes/llama.cpp").json()
    return next(item for item in installed["runtimes"] if item["id"] == installed["active_id"])


@pytest.fixture(scope="session")
def gguf_model(client: httpx.Client, runtime: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """The model downloaded, and deleted again at the end."""
    body = {"repo_id": GGUF_REPO, "quantization": GGUF_QUANTIZATION}
    yield run_job(client, "/downloads", "/downloads", DOWNLOAD_SECONDS, body)
    delete_model(client, GGUF_REPO)


@pytest.fixture(scope="session")
def gguf_server(client: httpx.Client, gguf_model: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """The model running, and stopped again at the end."""
    server = launch_server(client, GGUF_REPO, GGUF_QUANTIZATION)
    yield server
    stop_server(client, server)

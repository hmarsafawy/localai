from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

APP_DIR = Path(__file__).resolve().parent.parent

# Hugging Face's name for the task. Each task gets its own folder under
# `models_dir()`, because each lays its files out differently.
TEXT_GENERATION_TASK = "text-generation"


def data_dir() -> Path:
    """Where runtimes and models live: the app's own folder unless configured otherwise."""
    configured = os.getenv("LOCALAI_DATA_DIR")
    return Path(configured).expanduser().resolve() if configured else APP_DIR


def llama_runtimes_dir() -> Path:
    return data_dir() / "runtimes" / "llama.cpp"


def models_dir() -> Path:
    """Downloaded models, which can be moved to a bigger drive on their own. The
    default isn't `models` because `<app>/models` is the Python package."""
    configured = os.getenv("LOCALAI_MODELS_DIR")
    return Path(configured).expanduser().resolve() if configured else data_dir() / "model-files"


def text_models_dir() -> Path:
    return models_dir() / TEXT_GENERATION_TASK


def llama_server_logs_dir() -> Path:
    return data_dir() / "logs" / "llama-server"


def github_token() -> str | None:
    return os.getenv("GITHUB_TOKEN") or None

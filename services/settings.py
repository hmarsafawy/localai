from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

APP_DIR = Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    """Where runtimes live: the app's own folder unless configured otherwise.
    Downloaded models can't go in `<this>/models`, which is the Python package."""
    configured = os.getenv("LOCALAI_DATA_DIR")
    return Path(configured).expanduser().resolve() if configured else APP_DIR


def llama_runtimes_dir() -> Path:
    return data_dir() / "runtimes" / "llama.cpp"


def github_token() -> str | None:
    return os.getenv("GITHUB_TOKEN") or None

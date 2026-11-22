from __future__ import annotations

from pathlib import Path

import pytest

from services.settings import APP_DIR, data_dir, llama_runtimes_dir


def test_runtimes_live_inside_the_app_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOCALAI_DATA_DIR", raising=False)
    assert (APP_DIR / "main.py").is_file()
    assert data_dir() == APP_DIR
    assert llama_runtimes_dir() == APP_DIR / "runtimes" / "llama.cpp"


def test_an_empty_setting_counts_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    # What copying `.env.example` as it stands gives you.
    monkeypatch.setenv("LOCALAI_DATA_DIR", "")
    assert data_dir() == APP_DIR


def test_the_location_can_be_configured(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LOCALAI_DATA_DIR", str(tmp_path))
    assert llama_runtimes_dir() == tmp_path.resolve() / "runtimes" / "llama.cpp"

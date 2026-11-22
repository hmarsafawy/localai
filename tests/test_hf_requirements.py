"""What `/models/{id}/requirements` says about a diffusers pipeline, which used
to need `/models/{id}/pipeline`: each part's library and, when asked for, its
config, plus the pipeline's diffusers version and other settings."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import routes.huggingface_router as huggingface_router
import services.huggingface as huggingface_module
from main import app
from services.huggingface import HuggingFaceService

REPO = "someone/tiny-sdxl"
INDEX = {
    "_class_name": "StableDiffusionXLPipeline",
    "_diffusers_version": "0.24.0",
    "force_zeros_for_empty_prompt": True,
    "scheduler": ["diffusers", "EulerDiscreteScheduler"],
    "unet": ["diffusers", "UNet2DConditionModel"],
    "vae": ["diffusers", "AutoencoderKL"],
    "image_encoder": [None, None],
}
CONFIGS = {
    "scheduler/scheduler_config.json": {"_class_name": "EulerDiscreteScheduler", "num_train_timesteps": 1000},
    "unet/config.json": {"_class_name": "UNet2DConditionModel", "sample_size": 64},
    "vae/config.json": {"_class_name": "AutoencoderKL", "latent_channels": 4},
}
FILES = {
    "model_index.json": INDEX,
    **CONFIGS,
    "unet/diffusion_pytorch_model.safetensors": None,
    "vae/diffusion_pytorch_model.safetensors": None,
}


class FakeApi:
    def model_info(self, repo_id: str, files_metadata: bool = False) -> SimpleNamespace:
        return SimpleNamespace(
            id=REPO,
            tags=[],
            card_data=None,
            siblings=[SimpleNamespace(rfilename=name, size=1000) for name in FILES],
        )


@pytest.fixture
def downloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    fetched: list[str] = []

    def fake_download(repo_id: str, filename: str, token: str | None = None) -> str:
        fetched.append(filename)
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(FILES[filename]), encoding="utf-8")
        return str(path)

    monkeypatch.setattr(huggingface_module, "hf_hub_download", fake_download)
    return fetched


@pytest.fixture
def service() -> HuggingFaceService:
    return HuggingFaceService(api=FakeApi())  # type: ignore[arg-type]


def test_a_pipeline_says_what_loads_each_part_and_its_own_settings(
    service: HuggingFaceService, downloads: list[str]
) -> None:
    requirements = service.get_model_requirements(REPO)
    assert requirements.pipeline_class == "StableDiffusionXLPipeline"
    assert requirements.diffusers_version == "0.24.0"
    assert requirements.pipeline_config == {"force_zeros_for_empty_prompt": True}
    roles = {role.role: role for role in requirements.roles}
    assert roles["unet"].library == "diffusers" and roles["unet"].architecture == "UNet2DConditionModel"
    assert roles["image_encoder"].library is None and not roles["image_encoder"].required
    # Configs cost a download each, so they're only fetched when asked for.
    assert all(role.config is None for role in requirements.roles)
    assert downloads == ["model_index.json"]


def test_asking_for_configs_adds_each_parts_own(service: HuggingFaceService, downloads: list[str]) -> None:
    requirements = service.get_model_requirements(REPO, include_config=True)
    roles = {role.role: role for role in requirements.roles}
    assert roles["scheduler"].config == CONFIGS["scheduler/scheduler_config.json"]
    assert roles["unet"].config == CONFIGS["unet/config.json"]
    assert roles["vae"].config == CONFIGS["vae/config.json"]
    assert roles["image_encoder"].config is None
    assert sorted(downloads) == sorted(["model_index.json", *CONFIGS])


def test_requirements_over_http_and_the_old_pipeline_route_is_gone(
    service: HuggingFaceService, downloads: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(huggingface_router, "service", service)
    client = TestClient(app)

    body = client.get(f"/models/{REPO}/requirements", params={"fit": False, "include_config": True}).json()
    assert body["diffusers_version"] == "0.24.0"
    assert body["pipeline_config"] == {"force_zeros_for_empty_prompt": True}
    unet = next(role for role in body["roles"] if role["role"] == "unet")
    assert unet["library"] == "diffusers" and unet["config"] == CONFIGS["unet/config.json"]

    assert not any(getattr(route, "path", "").endswith("/pipeline") for route in app.routes)

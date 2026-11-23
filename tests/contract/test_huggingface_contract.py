"""Checks that Hugging Face still answers the way the stand-ins in
`test_model_downloads.py` (FakeHf and FakeHub) pretend it does, using the small
model the end-to-end tests run.

Where a field is missing the app carries on with nothing rather than failing,
so a change on Hugging Face's side would pass every other test, the end-to-end
ones included: without `lfs.sha256` downloads would quietly stop being checked,
and without sizes every file would look like it fits. This needs the internet
and downloads a few kilobytes; it runs on its own (`-m contract`)."""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from huggingface_hub import HfApi, hf_hub_url

from schemas.hf_model import HfModelSearchRequest
from services.huggingface import HuggingFaceService
from services.model_conversion import text_architectures, vision_architecture
from services.model_downloads import _converter_weights, _gguf_files, _parameter_count
from services.tensor_signatures import is_vision_key
from services.weight_headers import read_gguf_metadata

from ..e2e.helpers import GGUF_QUANTIZATION, GGUF_REPO, SAFETENSORS_REPO
from ..test_model_downloads import FakeHf, FakeHub

pytestmark = pytest.mark.contract

GGUF_FILE = f"SmolLM2-135M-Instruct-{GGUF_QUANTIZATION}.gguf"
# Anyone can see it, but its files need a license accepted first.
GATED_REPO = "meta-llama/Llama-3.2-1B"
# A small vision model with only safetensors, whose vision encoder is converted too.
VISION_REPO = "HuggingFaceTB/SmolVLM-256M-Instruct"
SHA256 = re.compile(r"[0-9a-f]{64}")


@pytest.fixture(scope="module")
def api() -> HfApi:
    # Without a token, as CI is and as a new user is, so a token on this
    # machine can't hide a change.
    return HfApi(token=False)


@pytest.fixture(scope="module")
def gguf_info(api: HfApi) -> Any:
    return api.model_info(GGUF_REPO, files_metadata=True)


@pytest.fixture(scope="module")
def http() -> Iterator[httpx.Client]:
    with httpx.Client(timeout=httpx.Timeout(30, connect=10)) as client:
        yield client


def test_a_repo_has_every_field_the_stand_in_gives_it(gguf_info: Any) -> None:
    hub = FakeHub()
    hub.files = {GGUF_FILE: b"GGUF"}
    fake = FakeHf(hub, GGUF_REPO).model_info(GGUF_REPO, files_metadata=True)
    for name in vars(fake):
        assert getattr(gguf_info, name, None) is not None, f"model_info has no {name}"
    assert gguf_info.id == GGUF_REPO
    # Downloads pin this commit, so a repo changing halfway can't mix files.
    assert re.fullmatch(r"[0-9a-f]{40}", gguf_info.sha), gguf_info.sha

    [fake_file] = fake.siblings
    real_files = [sibling for sibling in gguf_info.siblings if sibling.rfilename.endswith(".gguf")]
    assert GGUF_FILE in [sibling.rfilename for sibling in real_files]
    for sibling in real_files:
        for name in vars(fake_file):
            assert getattr(sibling, name, None) is not None, f"{sibling.rfilename} has no {name}"
        assert isinstance(sibling.size, int) and sibling.size > 0, sibling.rfilename
        assert SHA256.fullmatch(sibling.lfs.sha256 or ""), sibling.rfilename


def test_the_app_finds_every_gguf_with_its_size_and_checksum(gguf_info: Any) -> None:
    files = _gguf_files(gguf_info)
    assert GGUF_FILE in files
    for file in files.values():
        assert file.size > 0, file.path
        assert SHA256.fullmatch(file.sha256 or ""), f"{file.path} would be downloaded without a check"


def config_of(http: httpx.Client, repo_id: str, revision: str) -> dict[str, Any]:
    response = http.get(hf_hub_url(repo_id, "config.json", revision=revision), follow_redirects=True)
    assert response.status_code == 200, response.status_code
    return response.json()


def test_a_safetensors_repo_says_what_converting_it_needs(api: HfApi, http: httpx.Client) -> None:
    info = api.model_info(SAFETENSORS_REPO, files_metadata=True)
    # What picks the converter, and what the size estimate before converting uses.
    config = config_of(http, SAFETENSORS_REPO, info.sha)
    assert text_architectures(config) == ["LlamaForCausalLM"]
    assert vision_architecture(config) is None
    assert (_parameter_count(info) or 0) > 100_000_000
    weights = _converter_weights(info)
    assert weights is not None and weights.files == ["model.safetensors"]
    assert weights.size_bytes > 100 * 1024**2
    [sibling] = [sibling for sibling in info.siblings if sibling.rfilename == "model.safetensors"]
    assert SHA256.fullmatch(sibling.lfs.sha256 or "")


def test_a_vision_model_says_what_its_vision_encoder_is(api: HfApi, http: httpx.Client) -> None:
    info = api.model_info(VISION_REPO, files_metadata=True)
    # The Hub's summary of config.json leaves out what the converter goes by.
    assert "text_config" not in (info.config or {}) and "vision_config" not in (info.config or {})
    config = config_of(http, VISION_REPO, info.sha)
    assert text_architectures(config) == ["VLlama3ForCausalLM", "Idefics3ForConditionalGeneration"]
    assert vision_architecture(config) == "Idefics3ForConditionalGeneration"
    # The vision encoder's size, and what's left for the text model, come from the headers.
    metadata = api.get_safetensors_metadata(VISION_REPO, revision=info.sha)
    vision = sum(
        tensor.parameter_count
        for file in metadata.files_metadata.values()
        for name, tensor in file.tensors.items()
        if is_vision_key(name)
    )
    assert 50_000_000 < vision < (_parameter_count(info) or 0) / 2


def test_a_download_can_resume_and_ends_at_the_listed_size(gguf_info: Any, http: httpx.Client) -> None:
    size = next(sibling.size for sibling in gguf_info.siblings if sibling.rfilename == GGUF_FILE)
    url = hf_hub_url(GGUF_REPO, GGUF_FILE, revision=gguf_info.sha)
    # The access check before a download: anything but 401 or 403 is let through.
    assert http.head(url, follow_redirects=False).status_code not in (401, 403)
    start = size - 16
    response = http.get(url, headers={"Range": f"bytes={start}-"}, follow_redirects=True)
    assert response.status_code == 206, response.status_code
    # A file that came out another size than listed would be thrown away.
    assert response.headers["Content-Range"] == f"bytes {start}-{size - 1}/{size}"
    assert len(response.content) == 16


def test_a_gated_repo_turns_away_a_request_without_a_token(api: HfApi, http: httpx.Client) -> None:
    assert api.model_info(GATED_REPO).gated in ("auto", "manual")
    response = http.head(hf_hub_url(GATED_REPO, "config.json"), follow_redirects=False)
    # What the app turns into "accept its license, then check HF_TOKEN".
    assert response.status_code in (401, 403), response.status_code


def test_the_app_reads_a_ggufs_header_from_a_slice_of_it() -> None:
    metadata = read_gguf_metadata(None, GGUF_REPO, GGUF_FILE)
    assert metadata["general.architecture"] == "llama"
    assert metadata["llama.context_length"] == 8192


def test_search_results_have_what_the_app_shows() -> None:
    service = HuggingFaceService(api=HfApi(token=False))
    results = service.search_models(HfModelSearchRequest(search="SmolLM2-135M-Instruct", limit=10))
    model = next(item for item in results if item.model_id == SAFETENSORS_REPO)
    # The app fills a missing field with 0 or nothing, so check they're real.
    assert model.author == "HuggingFaceTB"
    assert model.downloads > 0 and model.likes > 0
    assert model.pipeline_tag == "text-generation"
    assert "safetensors" in model.tags
    assert model.sha and model.last_modified and model.created_at

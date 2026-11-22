"""Filenames are real, from unsloth/Qwen3.8-27B-GGUF and
JonathanColetti/Qwen3.8-27B-Uncensored-GGUF. `FakeHub` copies how Hugging Face
behaved when tested: a 302 to its CDN, with past-the-end requests getting a 416
that has no Content-Range."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote

import httpx
import pytest
import requests
from fastapi.testclient import TestClient
from huggingface_hub.utils import HfHubHTTPError
from pydantic import ValidationError

import routes.library_router as library_router
import routes.model_download_router as model_download_router
import services.model_downloads as downloads_module
from main import app
from models.hf_model import HfModelRequirements, HfRequirementRole, HfRoleOption
from models.jobs import JobInfo
from schemas.downloads import ModelDownloadRequest
from services.jobs import DOWNLOAD_CHUNK_BYTES, DownloadIntegrityError, JobContext, JobStore, download_file
from services.model_downloads import (
    MANIFEST_NAME,
    HubUnavailableError,
    LocalModelBusyError,
    LocalModelChoiceError,
    LocalModelNotFoundError,
    ModelAccessError,
    ModelDownloadRequestError,
    ModelDownloadService,
    NotEnoughSpaceError,
    _is_safe_repo_path,
)
from services.model_options import (
    MemoryBudget,
    OptionChoiceError,
    attach_options,
    build_options,
    default_option,
    match_option,
)
from services.settings import APP_DIR, models_dir, text_models_dir

REPO = "unsloth/Qwen3.8-27B-GGUF"
Q4 = "Qwen3.8-27B-UD-Q4_K_M.gguf"
Q8 = "Qwen3.8-27B-Q8_0.gguf"
PLAIN_Q4_0 = "Qwen3.8-27B-Q4_0.gguf"
MTP = "MTP/mtp-Qwen3.8-27B-Q4_0.gguf"
SHARD_1 = "BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf"
SHARD_2 = "BF16/Qwen3.8-27B-BF16-00002-of-00002.gguf"
MMPROJ_F16 = "mmproj-F16.gguf"
MMPROJ_BF16 = "mmproj-BF16.gguf"

ROOMY = MemoryBudget(has_gpu=True, vram_free_bytes=16_000_000_000, ram_available_bytes=34_000_000_000)


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def content(name: str, size: int) -> bytes:
    return (name.encode() * (size // len(name) + 1))[:size]


class FakeHub:
    """The HTTP side: `huggingface.co` redirects to a CDN, which serves ranges."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.head_status = 302
        self.gets: list[tuple[str, int]] = []
        self.heads: list[str] = []
        self.auth_seen: dict[str, str | None] = {}
        # Lets a test take over a CDN response: (file, start of the range).
        self.hook: Callable[[str, int], httpx.Response | None] | None = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = unquote(request.url.path)
        self.auth_seen[request.url.host] = request.headers.get("authorization")
        if request.url.host == "huggingface.co":
            file = path.split("/resolve/", 1)[1].split("/", 1)[1]
            if request.method == "HEAD":
                self.heads.append(file)
                return httpx.Response(self.head_status)
            return httpx.Response(302, headers={"Location": f"https://cdn.test/{file}"})

        file = path.lstrip("/")
        requested = request.headers.get("range")
        start = int(requested.removeprefix("bytes=").removesuffix("-")) if requested else 0
        self.gets.append((file, start))
        if self.hook is not None:
            response = self.hook(file, start)
            if response is not None:
                return response
        body = self.files[file]
        if start >= len(body):
            return httpx.Response(416)
        if start:
            headers = {"Content-Range": f"bytes {start}-{len(body) - 1}/{len(body)}"}
            return httpx.Response(206, content=body[start:], headers=headers)
        return httpx.Response(200, content=body)


class FakeHf:
    """The `huggingface_hub` side, with options built by the real code."""

    token = "hf_test"

    def __init__(self, hub: FakeHub, repo_id: str = REPO) -> None:
        self.hub = hub
        self.repo_id = repo_id
        self.api = self
        self.commit = "c0ffee"
        self.requirement_calls = 0
        self.extra_language_files: list[str] = []
        self.roles: list[HfRequirementRole] | None = None
        self.checksums: dict[str, str | None] = {}
        self.info_error: Exception | None = None

    def model_info(self, repo_id: str, files_metadata: bool = False) -> SimpleNamespace:
        assert files_metadata
        if self.info_error is not None:
            raise self.info_error
        return SimpleNamespace(
            id=self.repo_id,
            sha=self.commit,
            siblings=[
                SimpleNamespace(
                    rfilename=name,
                    size=len(body),
                    lfs=SimpleNamespace(sha256=self.checksums.get(name, sha256_of(body))),
                )
                for name, body in self.hub.files.items()
            ],
        )

    def get_model_requirements(self, repo_id: str) -> HfModelRequirements:
        self.requirement_calls += 1
        ggufs = [name for name in self.hub.files if name.endswith(".gguf")]
        roles = self.roles
        if roles is None:
            language = [name for name in ggufs if "mmproj" not in name]
            vision = [name for name in ggufs if "mmproj" in name]
            roles = [HfRequirementRole(role="language_model", present=True, files=language + self.extra_language_files)]
            if vision:
                roles.append(HfRequirementRole(role="vision_encoder", required=False, present=True, files=vision))
        requirements = HfModelRequirements(model_id=repo_id, roles=[role.model_copy(deep=True) for role in roles])
        attach_options(requirements, {name: len(body) for name, body in self.hub.files.items()})
        return requirements


@pytest.fixture(autouse=True)
def no_retry_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(downloads_module, "RETRY_DELAY_SECONDS", 0)


@pytest.fixture
def hub() -> FakeHub:
    fake = FakeHub()
    fake.files = {
        Q4: content("q4", 3000),
        Q8: content("q8", 6000),
        PLAIN_Q4_0: content("q40", 2500),
        MTP: content("mtp", 500),
        SHARD_1: content("one", 9000),
        SHARD_2: content("two", 9000),
        MMPROJ_F16: content("f16", 1000),
        MMPROJ_BF16: content("bf16", 1000),
    }
    return fake


@pytest.fixture
def hf(hub: FakeHub) -> FakeHf:
    return FakeHf(hub)


@pytest.fixture
def service(tmp_path: Path, hub: FakeHub, hf: FakeHf) -> ModelDownloadService:
    return ModelDownloadService(
        hf=hf,  # type: ignore[arg-type]
        budget=lambda: ROOMY,
        client=httpx.Client(transport=httpx.MockTransport(hub), headers={"Authorization": "Bearer hf_test"}),
        root=tmp_path / "model-files" / "text-generation",
    )


def wait_for(service: ModelDownloadService, job: JobInfo, timeout: float = 10) -> JobInfo:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = service.jobs.get(job.id)
        assert current is not None
        if current.status not in ("queued", "running"):
            return current
        time.sleep(0.005)
    raise AssertionError(f"job didn't finish: {service.jobs.get(job.id)}")


def folder_of(service: ModelDownloadService, repo_id: str = REPO) -> Path:
    return service.root / repo_id.replace("/", "--")


def manifest_of(service: ModelDownloadService) -> dict:
    return json.loads((folder_of(service) / MANIFEST_NAME).read_text(encoding="utf-8"))


def downloaded(service: ModelDownloadService, **choice: str) -> JobInfo:
    job = wait_for(service, service.start(REPO, **choice))
    assert job.status == "completed", job.error
    return job


def option_named(service: ModelDownloadService, option_id: str):
    model = service.library().models[0]
    return next(option for option in [*model.quantizations, *model.vision_encoders] if option.id == option_id)


# Choosing an option


UNSLOTH_NAMES = [
    "BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf",
    "BF16/Qwen3.8-27B-BF16-00002-of-00002.gguf",
    "MTP/mtp-Qwen3.8-27B-Q4_0.gguf",
    "Qwen3.8-27B-Q4_0.gguf",
    "Qwen3.8-27B-Q8_0.gguf",
    "Qwen3.8-27B-UD-IQ3_S.gguf",
    "Qwen3.8-27B-UD-Q4_K_M.gguf",
    "Qwen3.8-27B-UD-Q4_K_S.gguf",
]
JONATHAN_NAMES = [
    "Qwen3.8-27B-Uncensored-Q4_K_M.gguf",
    "Qwen3.8-27B-Uncensored-Q8_0.gguf",
    "Qwen3.8-27B-Uncensored-draft-Q8_0.gguf",
    "Qwen3.8-27B-Uncensored-noMTP-Q4_K_M.gguf",
    "Qwen3.8-27B-Uncensored-noMTP-Q8_0.gguf",
]


def options_from(names: list[str]) -> list[HfRoleOption]:
    return build_options(names, dict.fromkeys(names, 100))


@pytest.mark.parametrize(
    ("wanted", "expected"),
    [
        ("Q4_K_M", "UD-Q4_K_M"),  # every file of this quantization is named UD-
        ("q4_k_m", "UD-Q4_K_M"),
        ("UD-Q4_K_M", "UD-Q4_K_M"),
        ("Q4_0", "Q4_0"),  # not mtp-Q4_0
        ("mtp-Q4_0", "mtp-Q4_0"),
        ("BF16", "BF16"),
    ],
)
def test_a_choice_matches_by_id_then_by_quantization_name(wanted: str, expected: str) -> None:
    assert match_option(options_from(UNSLOTH_NAMES), wanted).id == expected


def test_a_plain_name_means_the_untagged_option_when_there_is_one() -> None:
    options = options_from(JONATHAN_NAMES)
    assert match_option(options, "Q4_K_M").files == ["Qwen3.8-27B-Uncensored-Q4_K_M.gguf"]
    assert match_option(options, "noMTP-Q4_K_M").files == ["Qwen3.8-27B-Uncensored-noMTP-Q4_K_M.gguf"]


def test_a_name_matching_several_options_is_refused_and_lists_them() -> None:
    options = options_from(["Qwen3.8-27B-Uncensored-draft-Q8_0.gguf", "Qwen3.8-27B-Uncensored-noMTP-Q8_0.gguf"])
    with pytest.raises(OptionChoiceError, match=r"draft-Q8_0, noMTP-Q8_0"):
        match_option(options, "Q8_0")


def test_a_name_matching_nothing_lists_what_there_is() -> None:
    with pytest.raises(OptionChoiceError, match=r"isn't available.*UD-Q4_K_M"):
        match_option(options_from(UNSLOTH_NAMES), "Q7_K")


def test_the_default_is_the_recommended_option_or_an_only_option_that_fits() -> None:
    plain = HfRoleOption(id="a", label="a")
    recommended = HfRoleOption(id="b", label="b", recommended=True)
    too_big = HfRoleOption(id="c", label="c", fit="too_large")
    assert default_option([plain, recommended]) is recommended
    assert default_option([plain]) is plain
    assert default_option([too_big]) is None
    assert default_option([plain, too_big]) is None
    assert default_option([]) is None


@pytest.mark.parametrize("repo_id", ["unsloth/Qwen3.8-27B-GGUF", "gpt2", "a/b", "A_b/c-d.e"])
def test_real_repo_ids_are_accepted(repo_id: str) -> None:
    assert ModelDownloadRequest(repo_id=repo_id).quantization == "auto"


@pytest.mark.parametrize(
    "repo_id", ["../x", "a/../b", "a/..", "..", "a/b/c", "a/b.", "a\\b", "a/b ", "/a", "a/", "", "C:/x", "a/b:c"]
)
def test_repo_ids_that_could_leave_their_folder_are_refused(repo_id: str) -> None:
    with pytest.raises(ValidationError):
        ModelDownloadRequest(repo_id=repo_id)


@pytest.mark.parametrize("path", ["a.gguf", "BF16/a-00001-of-00002.gguf", "x/y/z.gguf", ".gguf", "a b.gguf"])
def test_normal_repo_paths_are_safe(path: str) -> None:
    assert _is_safe_repo_path(path)


@pytest.mark.parametrize(
    "path",
    ["../a.gguf", "a/../../b.gguf", "/etc/a.gguf", "..\\a.gguf", "C:/a.gguf", "a:b.gguf", ".. /a.gguf", ".../a.gguf", "a\x00.gguf", ""],
)
def test_repo_paths_that_could_leave_their_folder_are_refused(path: str) -> None:
    assert not _is_safe_repo_path(path)


# Where files go


def test_text_models_go_in_their_own_folder_inside_the_app(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOCALAI_MODELS_DIR", raising=False)
    monkeypatch.delenv("LOCALAI_DATA_DIR", raising=False)
    assert text_models_dir() == APP_DIR / "model-files" / "text-generation"


def test_models_can_move_without_runtimes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LOCALAI_MODELS_DIR", str(tmp_path))
    monkeypatch.delenv("LOCALAI_DATA_DIR", raising=False)
    assert models_dir() == tmp_path.resolve()
    assert text_models_dir() == tmp_path.resolve() / "text-generation"


def test_models_follow_the_data_folder_unless_told_otherwise(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LOCALAI_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LOCALAI_MODELS_DIR", "")
    assert models_dir() == tmp_path.resolve() / "model-files"


# Downloading


def test_a_bare_request_downloads_the_recommended_model_and_vision_encoder(
    service: ModelDownloadService, hub: FakeHub
) -> None:
    job = downloaded(service)
    folder = folder_of(service)
    assert job.result is not None
    assert job.result["quantization"] == "Q8_0"  # the largest ordinary one that fits
    assert job.result["vision_encoder"] == "F16"
    assert job.result["model_path"] == str(folder / Q8)
    assert job.result["vision_encoder_path"] == str(folder / MMPROJ_F16)
    assert job.result["already_downloaded"] is False
    assert (folder / Q8).read_bytes() == hub.files[Q8]
    assert (folder / MMPROJ_F16).read_bytes() == hub.files[MMPROJ_F16]
    assert job.bytes_done == job.bytes_total == len(hub.files[Q8]) + len(hub.files[MMPROJ_F16])


def test_only_the_chosen_files_are_fetched(service: ModelDownloadService, hub: FakeHub) -> None:
    downloaded(service, quantization="Q4_K_M", vision_encoder="BF16")
    assert sorted(file for file, _ in hub.gets) == sorted([Q4, MMPROJ_BF16])
    on_disk = {path.name for path in folder_of(service).iterdir()}
    assert on_disk == {Q4, MMPROJ_BF16, MANIFEST_NAME}


def test_the_vision_encoder_can_be_skipped(service: ModelDownloadService, hub: FakeHub) -> None:
    job = downloaded(service, vision_encoder="none")
    assert job.result is not None
    assert job.result["vision_encoder"] is None
    assert job.result["vision_encoder_path"] is None
    assert [file for file, _ in hub.gets] == [Q8]


def test_a_repo_without_a_vision_encoder_downloads_just_the_model(service: ModelDownloadService, hub: FakeHub) -> None:
    del hub.files[MMPROJ_F16], hub.files[MMPROJ_BF16]
    job = downloaded(service)
    assert job.result is not None
    assert job.result["vision_encoder"] is None


def test_asking_for_a_vision_encoder_a_repo_lacks_is_an_error(service: ModelDownloadService, hub: FakeHub) -> None:
    del hub.files[MMPROJ_F16], hub.files[MMPROJ_BF16]
    with pytest.raises(ModelDownloadRequestError, match="no vision encoder"):
        service.start(REPO, vision_encoder="F16")


def test_a_split_model_gets_every_shard_and_points_at_the_first(service: ModelDownloadService, hub: FakeHub) -> None:
    job = downloaded(service, quantization="BF16", vision_encoder="none")
    folder = folder_of(service)
    assert (folder / SHARD_1).read_bytes() == hub.files[SHARD_1]
    assert (folder / SHARD_2).read_bytes() == hub.files[SHARD_2]
    assert job.result is not None
    assert job.result["model_path"] == str(folder / SHARD_1)
    assert option_named(service, "BF16").shards == 2


def test_a_job_says_what_was_picked_before_it_starts(service: ModelDownloadService) -> None:
    job = service.start(REPO)
    assert job.key == f"{REPO}:Q8_0+F16"
    assert job.details["quantization"] == "Q8_0"
    assert job.details["vision_encoder"] == "F16"
    assert job.details["repo_id"] == REPO
    assert job.details["task"] == "text-generation"
    assert job.details["revision"] == "c0ffee"
    wait_for(service, job)


def test_files_other_than_gguf_are_never_fetched(service: ModelDownloadService, hub: FakeHub, hf: FakeHf) -> None:
    hub.files["config.json"] = b"{}"
    hub.files["modeling_custom.py"] = b"import os"
    hf.extra_language_files = ["config.json", "modeling_custom.py"]
    downloaded(service, quantization="Q4_K_M", vision_encoder="none")
    assert [file for file, _ in hub.gets] == [Q4]
    assert not list(folder_of(service).glob("*.py"))


def test_a_format_llama_cpp_cant_run_isnt_on_offer(service: ModelDownloadService, hub: FakeHub, hf: FakeHf) -> None:
    name = "Qwen3.8-27B-MLX-8bit.safetensors"
    hub.files[name] = b"x" * 100_000
    hf.extra_language_files = [name]
    with pytest.raises(ModelDownloadRequestError, match=r"'MLX-8bit' isn't available.*UD-Q4_K_M"):
        service.start(REPO, quantization="MLX-8bit")
    # Even as the biggest thing in the repo, it isn't what `auto` picks.
    assert service.start(REPO).details["quantization"] == "Q8_0"


def test_a_quantization_name_still_works_when_a_stray_file_lengthens_the_ids(
    service: ModelDownloadService, hub: FakeHub, hf: FakeHf
) -> None:
    # With nothing shared between all the names, ids keep the model's name.
    hub.files["model.safetensors"] = b"x"
    hf.extra_language_files = ["model.safetensors"]
    job = wait_for(service, service.start(REPO, quantization="Q8_0", vision_encoder="none"))
    assert job.status == "completed", job.error
    assert job.details["quantization"] == "Qwen3.8-27B-Q8_0"
    assert (folder_of(service) / Q8).read_bytes() == hub.files[Q8]


def test_the_token_from_the_environment_is_what_gets_sent(hub: FakeHub, tmp_path: Path) -> None:
    hf = FakeHf(hub)
    assert ModelDownloadService(hf=hf, root=tmp_path).client.headers["authorization"] == "Bearer hf_test"  # type: ignore[arg-type]
    hf.token = None
    assert "authorization" not in ModelDownloadService(hf=hf, root=tmp_path).client.headers  # type: ignore[arg-type]


def test_the_token_goes_to_hugging_face_but_not_its_cdn(service: ModelDownloadService, hub: FakeHub) -> None:
    downloaded(service, quantization="Q4_K_M", vision_encoder="none")
    assert hub.auth_seen["huggingface.co"] == "Bearer hf_test"
    assert hub.auth_seen["cdn.test"] is None


def test_a_repo_filename_trying_to_escape_is_refused(
    service: ModelDownloadService, hub: FakeHub, hf: FakeHf, tmp_path: Path
) -> None:
    hub.files["../../evil-Q4_K_M.gguf"] = b"x"
    hf.roles = [HfRequirementRole(role="language_model", present=True, files=["../../evil-Q4_K_M.gguf"])]
    with pytest.raises(ModelDownloadRequestError, match="unsafe path"):
        service.start(REPO, quantization="Q4_K_M", vision_encoder="none")
    assert not hub.gets
    assert not list(tmp_path.rglob("evil*"))


# What gets refused


def test_a_repo_with_neither_gguf_nor_safetensors_is_refused(service: ModelDownloadService, hub: FakeHub) -> None:
    # Repos without GGUF are converted, which tests/test_model_conversion.py covers.
    hub.files = {"pytorch_model.bin": b"x"}
    with pytest.raises(ModelDownloadRequestError, match="neither GGUF files to run nor safetensors weights"):
        service.start("Qwen/Qwen3-0.6B")
    assert not service.jobs.all_jobs()


@pytest.mark.parametrize("status", [401, 403])
def test_a_gated_repo_is_reported_before_its_files_are_read(
    service: ModelDownloadService, hub: FakeHub, hf: FakeHf, status: int
) -> None:
    hub.head_status = status
    with pytest.raises(ModelAccessError, match=rf"Accept its license at https://huggingface.co/{REPO}"):
        service.start(REPO)
    assert hf.requirement_calls == 0
    assert not service.jobs.all_jobs()


def test_an_image_repo_is_refused_and_named(service: ModelDownloadService, hub: FakeHub, hf: FakeHf) -> None:
    hub.files = {"qwen-image-2.1-UC-Q4_K_M.gguf": b"x", "vae.safetensors": b"y"}
    hf.roles = [
        HfRequirementRole(role="transformer", present=True, files=["qwen-image-2.1-UC-Q4_K_M.gguf"]),
        HfRequirementRole(role="vae", present=True, files=["vae.safetensors"]),
    ]
    with pytest.raises(ModelDownloadRequestError, match=r"isn't a text model.*transformer, vae"):
        service.start("abenzerps/Qwen-Image-2.1-Uncensored-Quantized")


def test_an_unknown_quantization_lists_the_real_ones(service: ModelDownloadService) -> None:
    with pytest.raises(ModelDownloadRequestError, match=r"'Q7_K' isn't available.*UD-Q4_K_M"):
        service.start(REPO, quantization="Q7_K")
    assert not service.jobs.all_jobs()


def test_an_unknown_vision_encoder_lists_the_real_ones(service: ModelDownloadService) -> None:
    with pytest.raises(ModelDownloadRequestError, match=r"Vision encoder 'X9' isn't available.*F16"):
        service.start(REPO, vision_encoder="X9")


def test_auto_refuses_when_nothing_fits(service: ModelDownloadService) -> None:
    service._budget = lambda: MemoryBudget(has_gpu=False, vram_free_bytes=None, ram_available_bytes=10)
    with pytest.raises(ModelDownloadRequestError, match="pick one"):
        service.start(REPO)


def test_a_choice_that_wont_fit_downloads_with_a_warning(service: ModelDownloadService) -> None:
    service._budget = lambda: MemoryBudget(has_gpu=False, vram_free_bytes=None, ram_available_bytes=10)
    job = downloaded(service, quantization="Q8_0", vision_encoder="none")
    assert any("not to fit" in note for note in job.notes)


def test_without_hardware_information_auto_still_picks_q4_k_m(service: ModelDownloadService) -> None:
    service._budget = lambda: None
    assert service.start(REPO).details["quantization"] == "UD-Q4_K_M"


def test_not_enough_disk_space_is_refused(service: ModelDownloadService, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(downloads_module, "_free_bytes", lambda path: 1000)
    with pytest.raises(NotEnoughSpaceError, match="LOCALAI_MODELS_DIR"):
        service.start(REPO)
    assert not service.jobs.all_jobs()


def test_a_download_already_on_disk_doesnt_need_the_space_again(
    service: ModelDownloadService, monkeypatch: pytest.MonkeyPatch
) -> None:
    downloaded(service, quantization="Q8_0", vision_encoder="none")
    monkeypatch.setattr(downloads_module, "_free_bytes", lambda path: 0)
    downloaded(service, quantization="Q8_0", vision_encoder="none")


def test_only_what_is_left_of_a_partial_needs_space(
    service: ModelDownloadService, hub: FakeHub, monkeypatch: pytest.MonkeyPatch
) -> None:
    partial = folder_of(service) / f"{Q8}.partial"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(hub.files[Q8][:5000])
    remaining = len(hub.files[Q8]) - 5000
    monkeypatch.setattr(downloads_module, "_free_bytes", lambda path: downloads_module.SPACE_MARGIN_BYTES + remaining)
    downloaded(service, quantization="Q8_0", vision_encoder="none")
    monkeypatch.setattr(downloads_module, "_free_bytes", lambda path: downloads_module.SPACE_MARGIN_BYTES + remaining - 1)
    (folder_of(service) / Q8).unlink()
    partial.write_bytes(hub.files[Q8][:5000])
    with pytest.raises(NotEnoughSpaceError):
        service.start(REPO, quantization="Q8_0", vision_encoder="none")


def test_space_is_checked_again_once_a_queued_job_gets_its_turn(
    service: ModelDownloadService, monkeypatch: pytest.MonkeyPatch
) -> None:
    free = iter([10**15, 0])
    monkeypatch.setattr(downloads_module, "_free_bytes", lambda path: next(free))
    job = wait_for(service, service.start(REPO, quantization="Q8_0", vision_encoder="none"))
    assert job.status == "failed"
    assert "free space" in (job.error or "")


@pytest.mark.parametrize("failure", [OSError("network is down"), httpx.ConnectError("no route")])
def test_being_offline_is_its_own_error(service: ModelDownloadService, hf: FakeHf, failure: Exception) -> None:
    hf.info_error = failure
    with pytest.raises(HubUnavailableError, match="Could not reach"):
        service.start(REPO)


def test_hugging_face_errors_arent_mistaken_for_being_offline(service: ModelDownloadService, hf: FakeHf) -> None:
    response = requests.Response()
    response.status_code = 404
    hf.info_error = HfHubHTTPError("not found", response=response)
    with pytest.raises(HfHubHTTPError):
        service.start(REPO)


# Repeats, resuming and failures


def test_repeating_a_finished_download_fetches_nothing(service: ModelDownloadService, hub: FakeHub) -> None:
    downloaded(service)
    hub.gets.clear()
    job = downloaded(service)
    assert hub.gets == []
    assert job.result is not None
    assert job.result["already_downloaded"] is True
    assert job.bytes_done == job.bytes_total


def test_a_second_quantization_reuses_the_vision_encoder(service: ModelDownloadService, hub: FakeHub) -> None:
    downloaded(service, quantization="Q8_0")
    hub.gets.clear()
    downloaded(service, quantization="Q4_K_M")
    assert [file for file, _ in hub.gets] == [Q4]
    assert {option.id for option in service.library().models[0].quantizations} == {"Q8_0", "UD-Q4_K_M"}


def test_a_partial_resumes_from_where_it_stopped(service: ModelDownloadService, hub: FakeHub) -> None:
    partial = folder_of(service) / f"{Q8}.partial"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(hub.files[Q8][:2000])
    downloaded(service, quantization="Q8_0", vision_encoder="none")
    assert hub.gets == [(Q8, 2000)]
    assert (folder_of(service) / Q8).read_bytes() == hub.files[Q8]
    assert not partial.exists()


def test_a_finished_partial_is_checked_and_kept_not_fetched_again(service: ModelDownloadService, hub: FakeHub) -> None:
    # Hugging Face's bare 416 for this request looks just like a stale partial.
    partial = folder_of(service) / f"{Q8}.partial"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(hub.files[Q8])
    downloaded(service, quantization="Q8_0", vision_encoder="none")
    assert hub.gets == []
    assert (folder_of(service) / Q8).read_bytes() == hub.files[Q8]


def test_a_partial_longer_than_the_file_is_thrown_away(service: ModelDownloadService, hub: FakeHub) -> None:
    partial = folder_of(service) / f"{Q8}.partial"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(hub.files[Q8] + b"extra")
    downloaded(service, quantization="Q8_0", vision_encoder="none")
    assert hub.gets == [(Q8, 0)]
    assert (folder_of(service) / Q8).read_bytes() == hub.files[Q8]


def test_a_bad_checksum_never_gets_the_real_name_and_the_retry_works(service: ModelDownloadService, hub: FakeHub) -> None:
    good = hub.files[Q8]
    hub.hook = lambda file, start: httpx.Response(200, content=b"x" * len(good)) if file == Q8 else None
    job = wait_for(service, service.start(REPO, quantization="Q8_0", vision_encoder="none"))
    assert job.status == "failed"
    assert "SHA-256" in (job.error or "")
    folder = folder_of(service)
    assert not (folder / Q8).exists()
    assert not (folder / f"{Q8}.partial").exists()
    assert option_named(service, "Q8_0").status == "incomplete"

    hub.hook = None
    downloaded(service, quantization="Q8_0", vision_encoder="none")
    assert (folder / Q8).read_bytes() == good
    assert option_named(service, "Q8_0").status == "complete"


class _DroppingStream(httpx.SyncByteStream):
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __iter__(self):
        yield self.body
        raise httpx.ReadError("connection reset")


def test_a_dropped_connection_resumes_by_itself(service: ModelDownloadService, hub: FakeHub) -> None:
    body = content("big", DOWNLOAD_CHUNK_BYTES + 5000)
    hub.files[Q8] = body
    # A whole chunk first, as anything smaller wouldn't reach the disk, then nothing.
    replies = iter([body[:DOWNLOAD_CHUNK_BYTES], b""])

    def drop_twice(file: str, start: int) -> httpx.Response | None:
        piece = next(replies, None) if file == Q8 else None
        if piece is None:
            return None
        return httpx.Response(206 if start else 200, stream=_DroppingStream(piece))

    hub.hook = drop_twice
    job = downloaded(service, quantization="Q8_0", vision_encoder="none")
    assert (folder_of(service) / Q8).read_bytes() == body
    assert [start for file, start in hub.gets if file == Q8] == [0, DOWNLOAD_CHUNK_BYTES, DOWNLOAD_CHUNK_BYTES]
    assert sum("resuming automatically" in note for note in job.notes) == 1


def test_a_file_that_ends_early_resumes_instead_of_failing_its_checksum(service: ModelDownloadService, hub: FakeHub) -> None:
    body = hub.files[Q8]
    short = iter([True])
    hub.hook = lambda file, start: httpx.Response(200, content=body[:2000]) if file == Q8 and next(short, False) else None
    downloaded(service, quantization="Q8_0", vision_encoder="none")
    assert hub.gets == [(Q8, 0), (Q8, 2000)]
    assert (folder_of(service) / Q8).read_bytes() == body


def test_headway_resets_the_count_of_failed_tries(service: ModelDownloadService, hub: FakeHub) -> None:
    body = content("big", DOWNLOAD_CHUNK_BYTES * 8)
    hub.files[Q8] = body
    tries = downloads_module.MAX_ATTEMPTS_WITHOUT_PROGRESS + 2

    def one_chunk_at_a_time(file: str, start: int) -> httpx.Response | None:
        if file != Q8 or start >= DOWNLOAD_CHUNK_BYTES * tries:
            return None
        return httpx.Response(206 if start else 200, stream=_DroppingStream(body[start : start + DOWNLOAD_CHUNK_BYTES]))

    hub.hook = one_chunk_at_a_time
    downloaded(service, quantization="Q8_0", vision_encoder="none")
    assert (folder_of(service) / Q8).read_bytes() == body


def test_it_gives_up_after_tries_with_no_new_data_and_keeps_the_partial(service: ModelDownloadService, hub: FakeHub) -> None:
    partial = folder_of(service) / f"{Q8}.partial"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(hub.files[Q8][:2000])

    def always_down(file: str, start: int) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    hub.hook = always_down
    job = wait_for(service, service.start(REPO, quantization="Q8_0", vision_encoder="none"))
    assert job.status == "failed"
    assert "with no new data" in (job.error or "")
    assert "resume" in (job.error or "")
    assert len(hub.gets) == downloads_module.MAX_ATTEMPTS_WITHOUT_PROGRESS
    assert partial.read_bytes() == hub.files[Q8][:2000]


@pytest.mark.parametrize("status", [429, 500, 503])
def test_passing_server_errors_are_retried(service: ModelDownloadService, hub: FakeHub, status: int) -> None:
    errors = iter([status])

    def fail_once(file: str, start: int) -> httpx.Response | None:
        code = next(errors, None)
        return httpx.Response(code) if code else None

    hub.hook = fail_once
    downloaded(service, quantization="Q8_0", vision_encoder="none")
    assert len(hub.gets) == 2


def test_a_missing_file_is_not_retried(service: ModelDownloadService, hub: FakeHub) -> None:
    hub.hook = lambda file, start: httpx.Response(404)
    job = wait_for(service, service.start(REPO, quantization="Q8_0", vision_encoder="none"))
    assert job.status == "failed"
    assert "404" in (job.error or "")
    assert len(hub.gets) == 1


def test_losing_access_partway_explains_itself(service: ModelDownloadService, hub: FakeHub) -> None:
    hub.hook = lambda file, start: httpx.Response(403)
    job = wait_for(service, service.start(REPO, quantization="Q8_0", vision_encoder="none"))
    assert job.status == "failed"
    assert "HF_TOKEN" in (job.error or "")
    assert len(hub.gets) == 1


class _HeldStream(httpx.SyncByteStream):
    """Sends one chunk, then waits to be told to carry on."""

    def __init__(self, body: bytes, started: threading.Event, release: threading.Event) -> None:
        self.body = body
        self.started = started
        self.release = release

    def __iter__(self):
        yield self.body[:DOWNLOAD_CHUNK_BYTES]
        self.started.set()
        self.release.wait(10)
        yield self.body[DOWNLOAD_CHUNK_BYTES:]


@pytest.fixture
def held(hub: FakeHub):
    """Makes the first fetch of `Q8` hang after its first chunk."""
    body = content("big", DOWNLOAD_CHUNK_BYTES + 5000)
    hub.files[Q8] = body
    started, release = threading.Event(), threading.Event()
    first = iter([True])

    def hold(file: str, start: int) -> httpx.Response | None:
        if file == Q8 and next(first, False):
            return httpx.Response(200, stream=_HeldStream(body, started, release))
        return None

    hub.hook = hold
    yield SimpleNamespace(body=body, started=started, release=release)
    release.set()


def test_cancelling_keeps_the_data_shows_it_in_the_library_and_resumes(
    service: ModelDownloadService, hub: FakeHub, held: SimpleNamespace
) -> None:
    job = service.start(REPO, quantization="Q8_0", vision_encoder="none")
    assert held.started.wait(10)
    assert service.jobs.get(job.id).current_file == Q8
    assert service.jobs.cancel(job.id).cancel_requested
    held.release.set()
    assert wait_for(service, job).status == "cancelled"

    partial = folder_of(service) / f"{Q8}.partial"
    assert partial.stat().st_size == DOWNLOAD_CHUNK_BYTES
    assert not (folder_of(service) / Q8).exists()
    listed = option_named(service, "Q8_0")
    assert listed.status == "incomplete"
    assert listed.bytes_on_disk == DOWNLOAD_CHUNK_BYTES
    assert listed.size_bytes == len(held.body)

    downloaded(service, quantization="Q8_0", vision_encoder="none")
    assert hub.gets[-1] == (Q8, DOWNLOAD_CHUNK_BYTES)
    assert (folder_of(service) / Q8).read_bytes() == held.body
    assert option_named(service, "Q8_0").status == "complete"


def test_repeating_a_running_request_joins_it(service: ModelDownloadService, held: SimpleNamespace) -> None:
    job = service.start(REPO, quantization="Q8_0", vision_encoder="none")
    assert held.started.wait(10)
    assert service.start(REPO, quantization="q8_0", vision_encoder="NONE").id == job.id
    held.release.set()
    wait_for(service, job)


def test_downloads_run_one_at_a_time_and_share_files_safely(
    service: ModelDownloadService, hub: FakeHub, held: SimpleNamespace
) -> None:
    first = service.start(REPO, quantization="Q8_0", vision_encoder="F16")
    assert held.started.wait(10)
    second = service.start(REPO, quantization="Q4_K_M", vision_encoder="F16")
    assert second.id != first.id
    time.sleep(0.05)
    assert service.jobs.get(second.id).status == "queued"
    held.release.set()
    assert wait_for(service, first).status == "completed"
    assert wait_for(service, second).status == "completed"
    assert sum(file == MMPROJ_F16 for file, _ in hub.gets) == 1


def test_all_downloads_are_listed_newest_first(service: ModelDownloadService) -> None:
    first = service.start(REPO, quantization="Q8_0", vision_encoder="none")
    wait_for(service, first)
    second = service.start(REPO, quantization="Q4_K_M", vision_encoder="none")
    wait_for(service, second)
    assert [job.id for job in service.jobs.all_jobs()] == [second.id, first.id]


# When the repo changes


def test_an_updated_file_is_replaced_only_once_the_new_one_is_verified(
    service: ModelDownloadService, hub: FakeHub, hf: FakeHf
) -> None:
    downloaded(service, quantization="Q8_0", vision_encoder="none")
    old = hub.files[Q8]
    new = content("new", len(old) + 10)
    hub.files[Q8] = new
    hf.commit = "beef"

    seen_during_download: list[bytes] = []

    def look(file: str, start: int) -> None:
        seen_during_download.append((folder_of(service) / Q8).read_bytes())

    hub.hook = look
    job = downloaded(service, quantization="Q8_0", vision_encoder="none")
    assert seen_during_download == [old]
    assert (folder_of(service) / Q8).read_bytes() == new
    assert job.result is not None
    assert job.result["already_downloaded"] is False
    assert manifest_of(service)["files"][Q8]["revision"] == "beef"


def test_a_partial_of_an_older_version_is_not_resumed(service: ModelDownloadService, hub: FakeHub, hf: FakeHf) -> None:
    partial = folder_of(service) / f"{Q8}.partial"
    downloaded(service, quantization="Q8_0", vision_encoder="none")
    partial.write_bytes(b"old" * 100)  # left behind by an older version
    new = content("new", len(hub.files[Q8]) + 10)
    hub.files[Q8] = new
    hub.gets.clear()
    downloaded(service, quantization="Q8_0", vision_encoder="none")
    assert hub.gets == [(Q8, 0)]
    assert (folder_of(service) / Q8).read_bytes() == new


def test_a_downloaded_file_deleted_by_hand_is_noticed_and_fetched_again(service: ModelDownloadService, hub: FakeHub) -> None:
    downloaded(service, quantization="Q8_0", vision_encoder="none")
    (folder_of(service) / Q8).unlink()
    listed = option_named(service, "Q8_0")
    assert listed.status == "incomplete"
    assert listed.bytes_on_disk == 0
    hub.gets.clear()
    job = downloaded(service, quantization="Q8_0", vision_encoder="none")
    assert hub.gets == [(Q8, 0)]
    assert job.result is not None
    assert job.result["already_downloaded"] is False
    assert option_named(service, "Q8_0").status == "complete"


def test_a_downloaded_file_changed_by_hand_is_noticed(service: ModelDownloadService) -> None:
    downloaded(service, quantization="Q8_0", vision_encoder="none")
    (folder_of(service) / Q8).write_bytes(b"truncated")
    assert option_named(service, "Q8_0").status == "incomplete"


# The library


def test_an_empty_library_is_fine_and_creates_nothing(service: ModelDownloadService) -> None:
    library = service.library()
    assert library.models == []
    assert library.size_on_disk_bytes == 0
    assert library.free_bytes is not None
    assert library.models_dir == str(service.root)
    assert not service.root.exists()


def test_the_library_gives_paths_and_sizes_for_each_role(service: ModelDownloadService, hub: FakeHub) -> None:
    downloaded(service, quantization="Q4_K_M", vision_encoder="F16")
    downloaded(service, quantization="BF16", vision_encoder="none")
    library = service.library()
    assert len(library.models) == 1
    model = library.models[0]
    folder = folder_of(service)
    assert model.repo_id == REPO
    assert model.task == "text-generation"
    assert model.directory == str(folder)
    assert [option.id for option in model.quantizations] == ["UD-Q4_K_M", "BF16"]
    assert [option.id for option in model.vision_encoders] == ["F16"]
    q4, bf16 = model.quantizations
    assert q4.path == str(folder / Q4)
    assert q4.status == "complete"
    assert q4.size_bytes == q4.bytes_on_disk == len(hub.files[Q4])
    assert bf16.path == str(folder / SHARD_1)
    expected = sum(len(hub.files[name]) for name in (Q4, SHARD_1, SHARD_2, MMPROJ_F16))
    assert model.size_on_disk_bytes == library.size_on_disk_bytes == expected


def test_an_unreadable_manifest_doesnt_break_the_library(service: ModelDownloadService) -> None:
    downloaded(service, quantization="Q8_0", vision_encoder="none")
    for broken_name, text in (("a--broken", "{not json"), ("b--wrong", '["a list"]')):
        broken = service.root / broken_name
        broken.mkdir()
        (broken / MANIFEST_NAME).write_text(text, encoding="utf-8")
    (service.root / "c--no-manifest").mkdir()
    assert [model.repo_id for model in service.library().models] == [REPO]


def test_a_broken_manifest_is_replaced_by_the_next_download(service: ModelDownloadService, hub: FakeHub) -> None:
    downloaded(service, quantization="Q8_0", vision_encoder="none")
    (folder_of(service) / MANIFEST_NAME).write_text("{not json", encoding="utf-8")
    hub.gets.clear()
    downloaded(service, quantization="Q8_0", vision_encoder="none")
    assert hub.gets == [(Q8, 0)]  # with no record of it, the file can't be trusted
    assert option_named(service, "Q8_0").status == "complete"


# Deleting


def test_deleting_a_repo_removes_everything_including_partials(service: ModelDownloadService) -> None:
    downloaded(service, quantization="Q8_0")
    (folder_of(service) / f"{Q4}.partial").write_bytes(b"half")
    service.delete(REPO)
    assert not folder_of(service).exists()
    assert service.library().models == []


def test_deleting_one_quantization_leaves_the_rest(service: ModelDownloadService) -> None:
    downloaded(service, quantization="Q8_0", vision_encoder="F16")
    downloaded(service, quantization="Q4_K_M", vision_encoder="F16")
    service.delete(REPO, quantization="Q8_0")
    folder = folder_of(service)
    assert not (folder / Q8).exists()
    assert (folder / Q4).exists()
    assert (folder / MMPROJ_F16).exists()
    assert Q8 not in manifest_of(service)["files"]
    model = service.library().models[0]
    assert [option.id for option in model.quantizations] == ["UD-Q4_K_M"]
    assert [option.id for option in model.vision_encoders] == ["F16"]


def test_deleting_a_vision_encoder_leaves_the_model(service: ModelDownloadService) -> None:
    downloaded(service, quantization="Q4_K_M", vision_encoder="F16")
    service.delete(REPO, vision_encoder="F16")
    assert not (folder_of(service) / MMPROJ_F16).exists()
    assert (folder_of(service) / Q4).exists()
    assert service.library().models[0].vision_encoders == []


def test_deleting_accepts_the_same_names_as_downloading(service: ModelDownloadService) -> None:
    downloaded(service, quantization="Q4_K_M", vision_encoder="none")
    downloaded(service, quantization="Q8_0", vision_encoder="none")
    service.delete(REPO, quantization="q4_k_m")  # really `UD-Q4_K_M`
    assert not (folder_of(service) / Q4).exists()


def test_deleting_an_unfinished_download_removes_its_partial(
    service: ModelDownloadService, held: SimpleNamespace
) -> None:
    downloaded(service, quantization="Q4_K_M", vision_encoder="none")
    job = service.start(REPO, quantization="Q8_0", vision_encoder="none")
    assert held.started.wait(10)
    service.jobs.cancel(job.id)
    held.release.set()
    wait_for(service, job)
    partial = folder_of(service) / f"{Q8}.partial"
    assert partial.exists()
    service.delete(REPO, quantization="Q8_0")
    assert not partial.exists()
    assert (folder_of(service) / Q4).exists()


def test_deleting_a_split_model_removes_its_folder(service: ModelDownloadService) -> None:
    downloaded(service, quantization="BF16", vision_encoder="none")
    downloaded(service, quantization="Q4_K_M", vision_encoder="none")
    service.delete(REPO, quantization="BF16")
    assert not (folder_of(service) / "BF16").exists()
    assert (folder_of(service) / Q4).exists()


def test_deleting_the_last_option_removes_the_repo(service: ModelDownloadService) -> None:
    downloaded(service, quantization="Q4_K_M", vision_encoder="none")
    service.delete(REPO, quantization="Q4_K_M")
    assert not folder_of(service).exists()


def test_a_file_another_option_needs_is_kept(service: ModelDownloadService) -> None:
    downloaded(service, quantization="Q4_K_M", vision_encoder="none")
    downloaded(service, quantization="Q8_0", vision_encoder="none")
    manifest = manifest_of(service)
    for record in manifest["options"]:
        if record["id"] == "Q8_0":
            record["files"] = [Q8, Q4]
    (folder_of(service) / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    service.delete(REPO, quantization="UD-Q4_K_M")
    assert (folder_of(service) / Q4).exists()


def test_deleting_what_isnt_there_is_not_found(service: ModelDownloadService) -> None:
    with pytest.raises(LocalModelNotFoundError):
        service.delete("nobody/nothing")
    downloaded(service, quantization="Q4_K_M", vision_encoder="none")
    with pytest.raises(LocalModelNotFoundError, match="isn't available"):
        service.delete(REPO, quantization="Q8_0")
    assert (folder_of(service) / Q4).exists()


def test_a_repo_that_is_downloading_cant_be_deleted(service: ModelDownloadService, held: SimpleNamespace) -> None:
    downloaded(service, quantization="Q4_K_M", vision_encoder="none")
    job = service.start(REPO, quantization="Q8_0", vision_encoder="none")
    assert held.started.wait(10)
    with pytest.raises(LocalModelBusyError, match="Cancel that first"):
        service.delete(REPO)
    with pytest.raises(LocalModelBusyError):
        service.delete(REPO, quantization="Q4_K_M")
    held.release.set()
    wait_for(service, job)
    service.delete(REPO)
    assert not folder_of(service).exists()


def test_files_in_use_are_reported_as_busy(service: ModelDownloadService, monkeypatch: pytest.MonkeyPatch) -> None:
    downloaded(service, quantization="Q4_K_M", vision_encoder="none")

    def locked(path: Path) -> None:
        raise PermissionError("in use")

    monkeypatch.setattr(downloads_module.shutil, "rmtree", locked)
    with pytest.raises(LocalModelBusyError, match="open"):
        service.delete(REPO)


# `download_file` and the job store


def _context(store: JobStore) -> tuple[JobContext, threading.Event]:
    captured: dict = {}
    done = threading.Event()

    def work(context: JobContext) -> None:
        captured["context"] = context
        done.wait(10)

    store.submit("test", "key", work)
    while "context" not in captured:
        time.sleep(0.001)
    return captured["context"], done


def test_a_known_size_means_a_finished_partial_needs_no_request(tmp_path: Path) -> None:
    def no_requests(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request should be made")

    context, done = _context(JobStore())
    target = tmp_path / "x.partial"
    target.write_bytes(b"12345")
    with httpx.Client(transport=httpx.MockTransport(no_requests)) as client:
        download_file(client, "https://x.test/f", target, context, sha256=sha256_of(b"12345"), size=5)
    done.set()
    assert target.read_bytes() == b"12345"


def test_more_data_than_expected_is_rejected(tmp_path: Path) -> None:
    context, done = _context(JobStore())
    target = tmp_path / "x.partial"
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"123456"))) as client:
        with pytest.raises(DownloadIntegrityError, match="instead of 5"):
            download_file(client, "https://x.test/f", target, context, size=5)
    done.set()
    assert not target.exists()


def test_waiting_is_cut_short_by_a_cancel() -> None:
    store = JobStore()
    captured: dict = {}

    def work(context: JobContext) -> None:
        captured["started"] = time.monotonic()
        context.wait(30)

    job = store.submit("test", "key", work)
    while "started" not in captured:
        time.sleep(0.001)
    store.cancel(job.id)
    deadline = time.monotonic() + 5
    while store.get(job.id).status == "running" and time.monotonic() < deadline:
        time.sleep(0.005)
    assert store.get(job.id).status == "cancelled"
    assert time.monotonic() - captured["started"] < 5


def _follow(
    store: JobStore, job_id: str, on_each: Callable[[JobInfo], None] = lambda job: None, **options: float
) -> list[JobInfo]:
    async def collect() -> list[JobInfo]:
        seen = []
        async for job in store.follow(job_id, interval=0.001, **options):
            seen.append(job)
            on_each(job)
        return seen

    return asyncio.run(asyncio.wait_for(collect(), timeout=10))


def test_following_a_job_gives_each_change_until_it_finishes() -> None:
    store = JobStore()
    started, halfway, finish = threading.Event(), threading.Event(), threading.Event()

    def work(context: JobContext) -> dict:
        context.begin_transfer(100)
        started.set()
        assert halfway.wait(5)
        context.progress(50)
        assert finish.wait(5)
        return {"done": True}

    job = store.submit("test", "key", work)
    assert started.wait(5)

    def step(seen: JobInfo) -> None:
        (finish if seen.bytes_done == 50 else halfway).set()

    seen = _follow(store, job.id, step)
    assert [(job.status, job.bytes_done) for job in seen] == [("running", 0), ("running", 50), ("completed", 50)]
    assert seen[-1].result == {"done": True}
    assert seen[-1] == store.get(job.id)


def test_a_job_that_hasnt_changed_is_given_again() -> None:
    store = JobStore()
    started, release = threading.Event(), threading.Event()

    def work(context: JobContext) -> None:
        started.set()
        assert release.wait(5)

    job = store.submit("test", "key", work)
    assert started.wait(5)
    count = 0

    def step(seen: JobInfo) -> None:
        nonlocal count
        count += 1
        if count == 3:
            release.set()

    seen = _follow(store, job.id, step, keepalive=0)
    assert seen[0] == seen[1] == seen[2]
    assert seen[0].status == "running"
    assert seen[-1].status == "completed"


def test_following_a_finished_job_gives_it_once() -> None:
    store = JobStore()
    job = store.submit("test", "key", lambda context: None)
    while store.get(job.id).status != "completed":
        time.sleep(0.001)
    assert _follow(store, job.id) == [store.get(job.id)]


def test_following_an_unknown_job_gives_nothing() -> None:
    assert _follow(JobStore(), "nope") == []


# Routes


@pytest.fixture
def client(service: ModelDownloadService, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(model_download_router, "service", service)
    monkeypatch.setattr(library_router, "service", service)
    return TestClient(app)


def _finish(client: TestClient, job_id: str) -> dict:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        body = client.get(f"/downloads/{job_id}").json()
        if body["status"] not in ("queued", "running"):
            return body
        time.sleep(0.005)
    raise AssertionError("job didn't finish")


def test_the_whole_flow_over_http(client: TestClient, service: ModelDownloadService) -> None:
    started = client.post("/downloads", json={"repo_id": REPO, "quantization": "Q4_K_M", "vision_encoder": "F16"})
    assert started.status_code == 202
    body = started.json()
    assert body["details"]["quantization"] == "UD-Q4_K_M"
    assert body["details"]["vision_encoder"] == "F16"

    finished = _finish(client, body["id"])
    assert finished["status"] == "completed", finished["error"]
    assert finished["result"]["model_path"].endswith(Q4)

    assert [job["id"] for job in client.get("/downloads").json()] == [body["id"]]

    library = client.get("/library").json()
    assert library["models"][0]["repo_id"] == REPO
    assert library["models"][0]["quantizations"][0]["status"] == "complete"

    assert client.delete(f"/library/{REPO}", params={"vision_encoder": "F16"}).status_code == 204
    assert client.get("/library").json()["models"][0]["vision_encoders"] == []
    assert client.delete(f"/library/{REPO}").status_code == 204
    assert client.get("/library").json()["models"] == []
    assert client.delete(f"/library/{REPO}").status_code == 404


def test_a_repo_alone_is_enough_over_http(client: TestClient) -> None:
    started = client.post("/downloads", json={"repo_id": REPO})
    assert started.status_code == 202
    assert started.json()["details"]["quantization"] == "Q8_0"
    assert _finish(client, started.json()["id"])["status"] == "completed"


def test_cancelling_over_http(client: TestClient, held: SimpleNamespace) -> None:
    job_id = client.post("/downloads", json={"repo_id": REPO, "quantization": "Q8_0", "vision_encoder": "none"}).json()["id"]
    assert held.started.wait(10)
    assert client.get(f"/downloads/{job_id}").json()["current_file"] == Q8
    assert client.delete(f"/library/{REPO}").status_code == 409
    cancelled = client.delete(f"/downloads/{job_id}")
    assert cancelled.status_code == 200
    assert cancelled.json()["cancel_requested"] is True
    held.release.set()
    assert _finish(client, job_id)["status"] == "cancelled"


def test_a_download_can_be_followed_over_http(client: TestClient) -> None:
    job_id = client.post("/downloads", json={"repo_id": REPO}).json()["id"]
    with client.stream("GET", f"/downloads/{job_id}/events") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["cache-control"] == "no-cache"
        lines = list(response.iter_lines())
    events = [json.loads(line.removeprefix("data: ")) for line in lines if line.startswith("data: ")]
    assert {event["id"] for event in events} == {job_id}
    assert all(event["status"] in ("queued", "running") for event in events[:-1])
    assert events[-1]["status"] == "completed", events[-1]["error"]
    assert events[-1] == client.get(f"/downloads/{job_id}").json()


def test_unknown_jobs_are_404(client: TestClient) -> None:
    assert client.get("/downloads/nope").status_code == 404
    assert client.get("/downloads/nope/events").status_code == 404
    assert client.delete("/downloads/nope").status_code == 404


@pytest.mark.parametrize("status", [401, 403])
def test_gated_repos_are_403_over_http(client: TestClient, hub: FakeHub, status: int) -> None:
    hub.head_status = status
    response = client.post("/downloads", json={"repo_id": REPO})
    assert response.status_code == 403
    assert "license" in response.json()["detail"]


def test_bad_choices_are_422_over_http(client: TestClient, hub: FakeHub) -> None:
    response = client.post("/downloads", json={"repo_id": REPO, "quantization": "Q7_K"})
    assert response.status_code == 422
    assert "UD-Q4_K_M" in response.json()["detail"]
    hub.files = {"model.safetensors": b"x"}
    response = client.post("/downloads", json={"repo_id": "Qwen/Qwen3-0.6B"})
    assert response.status_code == 422
    assert "GGUF" in response.json()["detail"]


@pytest.mark.parametrize("repo_id", ["../evil", "a/../../b", "a/b/c"])
def test_bad_repo_ids_are_422_over_http(client: TestClient, repo_id: str) -> None:
    assert client.post("/downloads", json={"repo_id": repo_id}).status_code == 422


def test_a_missing_repo_id_is_422_over_http(client: TestClient) -> None:
    assert client.post("/downloads", json={}).status_code == 422


def test_no_space_is_507_over_http(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(downloads_module, "_free_bytes", lambda path: 0)
    response = client.post("/downloads", json={"repo_id": REPO})
    assert response.status_code == 507
    assert "free space" in response.json()["detail"]


def test_being_offline_is_502_over_http(client: TestClient, hf: FakeHf) -> None:
    hf.info_error = OSError("network is down")
    assert client.post("/downloads", json={"repo_id": REPO}).status_code == 502


@pytest.mark.parametrize(("status", "expected"), [(404, 404), (429, 429), (401, 403)])
def test_hugging_face_errors_map_over_http(client: TestClient, hf: FakeHf, status: int, expected: int) -> None:
    response = requests.Response()
    response.status_code = status
    hf.info_error = HfHubHTTPError("no", response=response)
    assert client.post("/downloads", json={"repo_id": REPO}).status_code == expected


@pytest.mark.parametrize("path", ["%2E%2E/canary", "..%2Fcanary", "a/%2E%2E/%2E%2E/canary", "a/b/c"])
def test_deleting_cant_reach_outside_the_library_over_http(
    client: TestClient, service: ModelDownloadService, path: str
) -> None:
    canary = service.root.parent / "canary"
    canary.mkdir(parents=True)
    assert client.delete(f"/library/{path}").status_code == 422
    assert canary.is_dir()


# Choosing downloaded files to run


def test_the_only_downloaded_quantization_runs_with_its_vision_encoder(service: ModelDownloadService) -> None:
    downloaded(service)
    selection = service.local_selection(REPO)
    assert (selection.repo_id, selection.model.id, selection.vision.id) == (REPO, "Q8_0", "F16")
    assert selection.model.path == str(folder_of(service) / Q8)
    assert Path(selection.vision.path).is_file()


def test_a_repo_is_found_however_its_capitalised(service: ModelDownloadService) -> None:
    downloaded(service)
    assert service.local_selection(REPO.upper()).repo_id == REPO


def test_a_repo_is_deleted_however_its_capitalised(service: ModelDownloadService) -> None:
    """Windows ignores case in paths anyway, Linux doesn't."""
    downloaded(service)
    service.delete(REPO.upper())
    assert not folder_of(service).exists()


def test_several_downloaded_quantizations_need_a_choice(service: ModelDownloadService) -> None:
    downloaded(service, quantization="Q4_K_M")
    downloaded(service, quantization="Q8_0")
    with pytest.raises(LocalModelChoiceError, match="several quantizations downloaded, so pick one: UD-Q4_K_M, Q8_0"):
        service.local_selection(REPO)
    assert service.local_selection(REPO, "q4_k_m").model.id == "UD-Q4_K_M"
    assert service.local_selection(REPO, "Q8_0").model.id == "Q8_0"


def test_a_quantization_that_isnt_downloaded_lists_the_ones_that_are(service: ModelDownloadService) -> None:
    downloaded(service, quantization="Q4_K_M")
    with pytest.raises(LocalModelChoiceError, match="'Q8_0' isn't available. Available: UD-Q4_K_M"):
        service.local_selection(REPO, "Q8_0")


def test_the_vision_encoder_prefers_f16_and_can_be_left_out(service: ModelDownloadService) -> None:
    downloaded(service, vision_encoder="BF16")
    assert service.local_selection(REPO).vision.id == "BF16"
    downloaded(service, vision_encoder="F16")
    assert service.local_selection(REPO).vision.id == "F16"
    assert service.local_selection(REPO, vision_encoder="bf16").vision.id == "BF16"
    assert service.local_selection(REPO, vision_encoder="none").vision is None


def test_without_a_vision_encoder_downloaded_it_runs_text_only(service: ModelDownloadService) -> None:
    downloaded(service, vision_encoder="none")
    assert service.local_selection(REPO).vision is None
    with pytest.raises(LocalModelChoiceError, match="no vision encoder downloaded"):
        service.local_selection(REPO, vision_encoder="F16")


def test_an_unfinished_download_cant_be_run(service: ModelDownloadService) -> None:
    downloaded(service, quantization="Q4_K_M")
    (folder_of(service) / Q4).unlink()
    with pytest.raises(LocalModelBusyError, match=r"UD-Q4_K_M is only partly downloaded \(0.0 of 0.0 GB\)"):
        service.local_selection(REPO)
    with pytest.raises(LocalModelBusyError, match="only partly downloaded"):
        service.local_selection(REPO, "Q4_K_M")


def test_an_unfinished_vision_encoder_is_skipped_unless_asked_for(service: ModelDownloadService) -> None:
    downloaded(service)
    (folder_of(service) / MMPROJ_F16).unlink()
    assert service.local_selection(REPO).vision is None
    with pytest.raises(LocalModelBusyError, match="F16 is only partly downloaded"):
        service.local_selection(REPO, vision_encoder="F16")


def test_what_is_still_downloading_cant_be_run(service: ModelDownloadService) -> None:
    downloaded(service, quantization="Q4_K_M")
    release = threading.Event()
    job = service.jobs.submit(
        "model_download",
        "busy",
        lambda context: release.wait(10) and None,
        details={"repo_id": REPO, "quantization": "UD-Q4_K_M", "vision_encoder": None},
    )
    try:
        with pytest.raises(LocalModelBusyError, match="UD-Q4_K_M is still downloading"):
            service.local_selection(REPO, "Q4_K_M")
    finally:
        release.set()
        wait_for(service, job)
    assert service.local_selection(REPO, "Q4_K_M").model.id == "UD-Q4_K_M"


def test_a_model_not_in_the_library_cant_be_run(service: ModelDownloadService) -> None:
    with pytest.raises(LocalModelNotFoundError, match="isn't in the library"):
        service.local_selection(REPO)


def test_files_in_use_cant_be_deleted(service: ModelDownloadService) -> None:
    downloaded(service)
    in_use = {REPO.casefold()}
    service.add_busy_check(lambda repo_id, options: "it's running" if repo_id.casefold() in in_use else None)
    with pytest.raises(LocalModelBusyError, match="it's running"):
        service.delete(REPO)
    with pytest.raises(LocalModelBusyError, match="it's running"):
        service.delete(REPO, vision_encoder="F16")
    assert (folder_of(service) / MMPROJ_F16).is_file()
    in_use.clear()
    service.delete(REPO)
    assert not folder_of(service).exists()


def test_the_busy_check_is_told_which_files_are_going(service: ModelDownloadService) -> None:
    """So a running model only blocks deleting its own files, not the others."""
    downloaded(service, quantization="Q4_K_M")
    downloaded(service, quantization="Q8_0", vision_encoder="none")
    asked: list = []
    service.add_busy_check(lambda repo_id, options: asked.append((repo_id, options)))
    service.delete(REPO.upper(), quantization="Q8_0")
    service.delete(REPO, vision_encoder="f16")
    service.delete(REPO)
    assert asked == [
        (REPO.upper(), {("language_model", "Q8_0")}),
        (REPO, {("vision_encoder", "F16")}),
        (REPO, None),
    ]


def test_an_unknown_option_is_reported_before_asking_whether_its_busy(service: ModelDownloadService) -> None:
    downloaded(service)
    service.add_busy_check(lambda repo_id, options: "it's running")
    with pytest.raises(LocalModelNotFoundError):
        service.delete(REPO, quantization="Q2_K")

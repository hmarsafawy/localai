"""Asset names and `--list-devices` output here are real, from llama.cpp's
releases and from running its builds on a PC with a dGPU and an iGPU."""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import sys
import tarfile
import threading
import time
import zipfile
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

import routes.runtime_router as runtime_router
import services.llama_runtime as runtime_module
from main import app
from models.hardware import CpuInfo, GpuApis, GpuInfo, HardwareInfo, MemoryInfo, OsInfo
from models.jobs import JobInfo
from models.runtime import RuntimeDevice
from services.jobs import DOWNLOAD_CHUNK_BYTES, JobContext, JobStore
from services.llama_runtime import (
    GITHUB_API,
    GitHubRateLimitError,
    GitHubUnavailableError,
    LaunchableRuntime,
    LlamaRuntimeService,
    ParsedBuild,
    RuntimeConflictError,
    RuntimeNotFoundError,
    UnsafeArchiveError,
    _single_root,
    _Verification,
    backend_preferences,
    match_gpu_device,
    parse_build_name,
    parse_device_free_memory,
    parse_list_devices,
    rank_builds,
    safe_extract,
    same_gpu_name,
    server_binary_name,
    server_dir,
    verify_runtime,
)

GIB = 1024**3

RX_9070 = GpuInfo(
    name="AMD Radeon RX 9070 XT",
    vendor="amd",
    kind="discrete",
    vram_total_bytes=16 * GIB,
    apis=GpuApis(rocm=True, vulkan=True),
)
RADEON_IGPU = GpuInfo(
    name="AMD Radeon(TM) Graphics",
    vendor="amd",
    kind="integrated",
    vram_total_bytes=GIB // 2,
    apis=GpuApis(rocm=True, vulkan=True),
)


def nvidia(cuda: str | None = "12.8") -> GpuInfo:
    return GpuInfo(
        name="NVIDIA GeForce RTX 3080",
        vendor="nvidia",
        kind="discrete",
        vram_total_bytes=10 * GIB,
        apis=GpuApis(cuda=cuda, vulkan=True),
    )


def hardware(gpus: list[GpuInfo], system: str = "windows", arch: str = "x64") -> HardwareInfo:
    return HardwareInfo(
        os=OsInfo(system=system, arch=arch),
        cpu=CpuInfo(),
        memory=MemoryInfo(total_bytes=32 * GIB, available_bytes=32 * GIB, used_bytes=0, percent_used=0),
        gpus=gpus,
        data_dir=".",
    )


@pytest.mark.parametrize(
    ("name", "os_name", "arch", "backend", "version"),
    [
        ("llama-b12047-bin-win-vulkan-x64.zip", "windows", "x64", "vulkan", None),
        ("llama-b12047-bin-win-rocm-10.0-x64.zip", "windows", "x64", "rocm", "10.0"),
        ("llama-b12047-bin-win-cuda-13.4-x64.zip", "windows", "x64", "cuda", "13.4"),
        ("llama-b12047-bin-win-cuda-13.4-arm64.zip", "windows", "arm64", "cuda", "13.4"),
        ("llama-b12047-bin-win-cpu-x64.zip", "windows", "x64", "cpu", None),
        ("llama-b12047-bin-win-sycl-x64.zip", "windows", "x64", "sycl", None),
        ("llama-b12047-bin-win-openvino-2026.4.1-x64.zip", "windows", "x64", "openvino", "2026.4.1"),
        ("llama-b12047-bin-win-opencl-adreno-arm64.zip", "windows", "arm64", "opencl", "adreno"),
        ("llama-b12047-bin-ubuntu-x64.tar.gz", "linux", "x64", "cpu", None),
        ("llama-b12047-bin-ubuntu-arm64.tar.gz", "linux", "arm64", "cpu", None),
        ("llama-b12047-bin-ubuntu-vulkan-x64.tar.gz", "linux", "x64", "vulkan", None),
        ("llama-b12047-bin-ubuntu-rocm-10.0-x64.tar.gz", "linux", "x64", "rocm", "10.0"),
        ("llama-b12047-bin-ubuntu-cuda-12.8-x64.tar.gz", "linux", "x64", "cuda", "12.8"),
        ("llama-b12047-bin-ubuntu-sycl-fp16-x64.tar.gz", "linux", "x64", "sycl", "fp16"),
        ("llama-b12047-bin-ubuntu-26.04-s390x.tar.gz", "linux", "s390x", "cpu", None),
        ("llama-b12047-bin-ubuntu-zdnn-s390x.tar.gz", "linux", "s390x", "zdnn", None),
        ("llama-b12047-bin-linux-arm64-snapdragon.tar.gz", "linux", "arm64", "snapdragon", None),
        ("llama-b12047-bin-macos-arm64.tar.gz", "macos", "arm64", "metal", None),
        ("llama-b12047-bin-macos-x64.tar.gz", "macos", "x64", "cpu", None),
        ("llama-b12047-bin-android-arm64.tar.gz", "android", "arm64", "cpu", None),
    ],
)
def test_parse_build_name(name: str, os_name: str, arch: str, backend: str, version: str | None) -> None:
    build = parse_build_name(name)
    assert build is not None
    assert (build.os, build.arch, build.backend, build.backend_version) == (os_name, arch, backend, version)
    assert build.tag == "b12047"
    assert build.build == 12047
    assert not build.id.endswith((".zip", ".gz"))


def test_parse_build_name_accepts_a_folder_name() -> None:
    build = parse_build_name("llama-b10766-bin-win-rocm-7.14-x64")
    assert build is not None
    assert (build.backend, build.backend_version, build.id) == ("rocm", "7.14", "llama-b10766-bin-win-rocm-7.14-x64")


@pytest.mark.parametrize(
    "name",
    [
        "cudart-llama-bin-win-cuda-13.4-x64.zip",
        "cudart-llama-b12047-bin-ubuntu-cuda-13.4-x64.tar.gz",
        "llama-b12047-xcframework.zip",
        "llama-b12047-ui.tar.gz",
        "nightly-tag.txt",
        ".downloads",
        "state.json",
    ],
)
def test_parse_build_name_ignores_everything_else(name: str) -> None:
    assert parse_build_name(name) is None


WINDOWS_ASSETS = [
    "llama-b1-bin-win-cpu-x64.zip",
    "llama-b1-bin-win-cuda-12.4-x64.zip",
    "llama-b1-bin-win-cuda-13.4-x64.zip",
    "llama-b1-bin-win-openvino-2026.4.1-x64.zip",
    "llama-b1-bin-win-rocm-10.0-x64.zip",
    "llama-b1-bin-win-sycl-x64.zip",
    "llama-b1-bin-win-vulkan-x64.zip",
]


def order_for(gpus: list[GpuInfo], assets: list[str] | None = None, **host: str) -> list[str]:
    builds = [parse_build_name(name) for name in (assets or WINDOWS_ASSETS)]
    ranked, _ = rank_builds([build for build in builds if build], hardware(gpus, **host))
    return [f"{build.backend}-{build.backend_version}" if build.backend_version else build.backend for build, _ in ranked]


def test_ranking_amd_with_rocm_runtime() -> None:
    assert order_for([RX_9070, RADEON_IGPU]) == ["rocm-10.0", "vulkan", "cpu"]


def test_ranking_amd_without_rocm_runtime_is_vulkan_first() -> None:
    no_sdk = RX_9070.model_copy(update={"apis": GpuApis(rocm=False, vulkan=True)})
    assert order_for([no_sdk]) == ["vulkan", "cpu"]


def test_ranking_nvidia_never_exceeds_what_the_driver_supports() -> None:
    assert order_for([nvidia("12.8")]) == ["cuda-12.4", "vulkan", "cpu"]
    assert order_for([nvidia("13.4")]) == ["cuda-13.4", "cuda-12.4", "vulkan", "cpu"]


def test_ranking_nvidia_with_a_driver_too_old_for_any_build() -> None:
    assert order_for([nvidia("11.8")]) == ["vulkan", "cpu"]


def test_ranking_nvidia_without_nvidia_smi() -> None:
    assert order_for([nvidia(None)]) == ["vulkan", "cpu"]


def test_ranking_intel_discrete() -> None:
    arc = GpuInfo(name="Intel Arc A770", vendor="intel", kind="discrete", apis=GpuApis(vulkan=True))
    assert order_for([arc]) == ["vulkan", "cpu"]


def test_ranking_integrated_gpu_only_prefers_cpu() -> None:
    assert order_for([RADEON_IGPU]) == ["cpu", "vulkan"]


def test_ranking_no_gpu() -> None:
    assert order_for([]) == ["cpu"]


def test_ranking_without_vulkan_drivers() -> None:
    no_vulkan = RX_9070.model_copy(update={"apis": GpuApis(rocm=False, vulkan=False)})
    assert order_for([no_vulkan]) == ["cpu"]


def test_ranking_apple_silicon() -> None:
    apple = GpuInfo(name="Apple M3", vendor="apple", kind="unified", apis=GpuApis(metal=True))
    assert order_for([apple], ["llama-b1-bin-macos-arm64.tar.gz"], system="macos", arch="arm64") == ["metal"]


def test_ranking_picks_the_largest_gpu_when_there_are_two() -> None:
    assert backend_preferences(hardware([RX_9070, nvidia("13.4")]))[0].backend == "rocm"
    bigger = nvidia("13.4").model_copy(update={"vram_total_bytes": 24 * GIB})
    assert backend_preferences(hardware([RX_9070, bigger]))[0].backend == "cuda"


def test_builds_that_auto_wont_pick_still_say_why() -> None:
    builds = [parse_build_name(name) for name in WINDOWS_ASSETS]
    _, others = rank_builds([build for build in builds if build], hardware([RX_9070]))
    reasons = {build.backend: reason for build, reason in others}
    assert reasons["cuda"] == "Needs an NVIDIA GPU"
    assert "oneAPI" in reasons["sycl"]


VULKAN_OUTPUT = """version: 0.4.0-dev (build 10878, commit 4850c7727)
built with Clang 20.1.8 for Windows x86_64

Available devices:
  Vulkan0: AMD Radeon RX 9070 XT (16304 MiB, 15416 MiB free)
  Vulkan1: AMD Radeon(TM) Graphics (16208 MiB, 15397 MiB free)
"""
ROCM_OUTPUT = """Available devices:
  ROCm0: AMD Radeon(TM) Graphics (12476 MiB, 12193 MiB free)
  ROCm1: AMD Radeon RX 9070 XT (16304 MiB, 16153 MiB free)
"""


def test_parse_list_devices() -> None:
    devices = parse_list_devices(VULKAN_OUTPUT)
    assert [device.id for device in devices] == ["Vulkan0", "Vulkan1"]
    assert devices[0].name == "AMD Radeon RX 9070 XT"
    assert devices[0].memory_total_bytes == 16304 * 1024**2


def test_parse_list_devices_with_no_gpu() -> None:
    assert parse_list_devices("Available devices:\n  (none)\n") == []


def test_gpu_device_follows_the_name_not_the_number() -> None:
    assert match_gpu_device(parse_list_devices(VULKAN_OUTPUT), [RX_9070, RADEON_IGPU]) == "Vulkan0"
    assert match_gpu_device(parse_list_devices(ROCM_OUTPUT), [RX_9070, RADEON_IGPU]) == "ROCm1"


def test_gpu_device_isnt_fooled_by_an_igpu_reporting_more_memory() -> None:
    devices = [
        RuntimeDevice(id="Vulkan0", name="AMD Radeon(TM) Graphics", memory_total_bytes=64 * GIB),
        RuntimeDevice(id="Vulkan1", name="AMD Radeon RX 9070 XT", memory_total_bytes=16 * GIB),
    ]
    assert match_gpu_device(devices, [RX_9070, RADEON_IGPU]) == "Vulkan1"


def test_gpu_device_uses_the_igpu_if_its_all_there_is() -> None:
    assert match_gpu_device(parse_list_devices(ROCM_OUTPUT)[:1], [RADEON_IGPU]) == "ROCm0"


def test_gpu_device_prefers_an_unknown_device_over_a_known_igpu() -> None:
    devices = [
        RuntimeDevice(id="D0", name="AMD Radeon(TM) Graphics"),
        RuntimeDevice(id="D1", name="Some Card"),
    ]
    assert match_gpu_device(devices, [RADEON_IGPU]) == "D1"


def test_gpu_device_without_devices() -> None:
    assert match_gpu_device([], [RX_9070]) is None


def test_the_free_memory_a_build_sees_is_read_per_device() -> None:
    assert parse_device_free_memory(ROCM_OUTPUT) == {"ROCm0": 12193 * 1024**2, "ROCm1": 16153 * 1024**2}
    assert parse_device_free_memory("Available devices:\n  (none)\n") == {}


@pytest.mark.parametrize(
    ("device", "gpu", "same"),
    [
        ("AMD Radeon RX 9070 XT", "AMD Radeon RX 9070 XT", True),
        ("AMD Radeon(TM) Graphics", "AMD Radeon(TM) Graphics", True),
        ("NVIDIA GeForce RTX 4090", "GeForce RTX 4090", True),
        ("AMD Radeon RX 9070 XT", "AMD Radeon(TM) Graphics", False),
        ("", "AMD Radeon RX 9070 XT", False),
    ],
)
def test_devices_and_gpus_are_matched_by_name(device: str, gpu: str, same: bool) -> None:
    assert same_gpu_name(device, gpu) is same


def test_a_build_is_asked_what_it_sees_now(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    asked: list = []

    def answer(directory: Path, backend: str, flag: str) -> tuple[int, str]:
        asked.append((directory, backend, flag))
        return 0, ROCM_OUTPUT

    monkeypatch.setattr(runtime_module, "_run_server_command", answer)
    runtime = LaunchableRuntime(id="r", directory=tmp_path, backend="rocm", gpu_device="ROCm1")
    devices, free = LlamaRuntimeService.current_devices(runtime)
    assert [device.id for device in devices] == ["ROCm0", "ROCm1"]
    assert free["ROCm1"] == 16153 * 1024**2
    assert asked == [(tmp_path, "rocm", "--list-devices")]
    monkeypatch.setattr(runtime_module, "_run_server_command", lambda *args: (1, "boom"))
    assert LlamaRuntimeService.current_devices(runtime) is None


def make_zip(path: Path, entries: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as bundle:
        for name, content in entries.items():
            bundle.writestr(name, content)
    return path


def make_tar(path: Path, entries: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as bundle:
        for name, content in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            bundle.addfile(info, io.BytesIO(content))
    return path


def test_safe_extract_zip(tmp_path: Path) -> None:
    archive = make_zip(tmp_path / "a.zip", {"llama-server.exe": b"x", "sub/ggml.dll": b"y"})
    safe_extract(archive, tmp_path / "out")
    assert (tmp_path / "out" / "llama-server.exe").read_bytes() == b"x"
    assert (tmp_path / "out" / "sub" / "ggml.dll").read_bytes() == b"y"


def test_safe_extract_tar(tmp_path: Path) -> None:
    archive = make_tar(tmp_path / "a.tar.gz", {"llama-b1/llama-server": b"x"})
    safe_extract(archive, tmp_path / "out")
    assert (tmp_path / "out" / "llama-b1" / "llama-server").read_bytes() == b"x"


@pytest.mark.parametrize("name", ["../evil.dll", "a/../../evil.dll", "/abs/evil.dll"] + (["C:/evil.dll"] if sys.platform == "win32" else []))
def test_safe_extract_rejects_zip_paths_that_escape(tmp_path: Path, name: str) -> None:
    archive = make_zip(tmp_path / "a.zip", {name: b"x"})
    with pytest.raises(UnsafeArchiveError):
        safe_extract(archive, tmp_path / "deep" / "out")
    assert not (tmp_path / "deep" / "evil.dll").exists()


def test_safe_extract_writes_nothing_if_any_entry_is_bad(tmp_path: Path) -> None:
    archive = make_zip(tmp_path / "a.zip", {"good.dll": b"x", "../evil.dll": b"y"})
    with pytest.raises(UnsafeArchiveError):
        safe_extract(archive, tmp_path / "out")
    assert not (tmp_path / "out" / "good.dll").exists()


def test_safe_extract_rejects_zip_symlinks(tmp_path: Path) -> None:
    info = zipfile.ZipInfo("link")
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(tmp_path / "a.zip", "w") as bundle:
        bundle.writestr(info, "/etc/passwd")
    with pytest.raises(UnsafeArchiveError):
        safe_extract(tmp_path / "a.zip", tmp_path / "out")


def test_safe_extract_rejects_tar_paths_that_escape(tmp_path: Path) -> None:
    archive = make_tar(tmp_path / "a.tar.gz", {"../evil.so": b"x"})
    with pytest.raises(UnsafeArchiveError):
        safe_extract(archive, tmp_path / "deep" / "out")
    assert not (tmp_path / "deep" / "evil.so").exists()


def test_single_root_descends_through_wrapper_folders(tmp_path: Path) -> None:
    nested = tmp_path / "llama-b11960"
    nested.mkdir()
    (nested / "llama-server").write_bytes(b"x")
    (nested / "lib.so").write_bytes(b"y")
    assert _single_root(tmp_path) == nested


def test_single_root_leaves_a_flat_folder_alone(tmp_path: Path) -> None:
    (tmp_path / "llama-server.exe").write_bytes(b"x")
    assert _single_root(tmp_path) == tmp_path


class FakeHardware:
    def __init__(self, info: HardwareInfo) -> None:
        self.info = info

    def get_hardware(self, refresh: bool = False) -> HardwareInfo:
        return self.info


def asset(name: str, content: bytes = b"", state: str = "uploaded", digest: bool = True) -> dict:
    return {
        "name": name,
        "size": len(content),
        "state": state,
        "browser_download_url": f"https://downloads.test/{name}",
        "digest": f"sha256:{hashlib.sha256(content).hexdigest()}" if digest else None,
    }


class FakeGitHub:
    """Stands in for GitHub and its download CDN, with `Range` support."""

    def __init__(self) -> None:
        self.releases: dict[str, dict] = {}
        self.listing: list[dict] = []
        self.latest: dict | None = None
        self.files: dict[str, bytes] = {}
        self.requests: list[httpx.Request] = []
        self.api_override: Callable[[httpx.Request], httpx.Response | None] | None = None

    def add_release(self, tag: str, assets: dict[str, bytes], **extra: object) -> dict:
        release = {
            "tag_name": tag,
            "target_commitish": "abc",
            "published_at": "2026-10-28T12:45:39Z",
            "draft": False,
            "body": "",
            "assets": [asset(name, content) for name, content in assets.items()],
            **extra,
        }
        self.releases[tag] = release
        self.listing.insert(0, release)
        for name, content in assets.items():
            self.files[name] = content
        return release

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url = str(request.url)
        if url.startswith("https://downloads.test/"):
            return self._download(request, url.rsplit("/", 1)[1])
        if self.api_override is not None:
            overridden = self.api_override(request)
            if overridden is not None:
                return overridden
        path = url.removeprefix(GITHUB_API)
        if path == "/releases/latest":
            return httpx.Response(200, json=self.latest) if self.latest else httpx.Response(404)
        if path.startswith("/releases?"):
            return httpx.Response(200, json=self.listing)
        if path.startswith("/releases/tags/"):
            release = self.releases.get(path.rsplit("/", 1)[1])
            return httpx.Response(200, json=release) if release else httpx.Response(404)
        return httpx.Response(404)

    def _download(self, request: httpx.Request, name: str) -> httpx.Response:
        content = self.files.get(name)
        if content is None:
            return httpx.Response(404)
        range_header = request.headers.get("Range")
        if not range_header:
            return httpx.Response(200, content=content)
        start = int(range_header.removeprefix("bytes=").rstrip("-"))
        if start >= len(content):
            return httpx.Response(416, headers={"Content-Range": f"bytes */{len(content)}"})
        return httpx.Response(206, content=content[start:])


def server_zip(extra: dict[str, bytes] | None = None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr(server_binary_name(), b"binary")
        for name, content in (extra or {}).items():
            bundle.writestr(name, content)
    return buffer.getvalue()


@pytest.fixture
def github() -> FakeGitHub:
    return FakeGitHub()


@pytest.fixture
def verifier(monkeypatch: pytest.MonkeyPatch) -> dict[str, _Verification]:
    """Canned test results per backend, since the fake binaries can't be run."""
    results: dict[str, _Verification] = {}

    def fake_verify(path: Path, backend: str, gpus: list[GpuInfo]) -> _Verification:
        assert (path / server_binary_name()).is_file()
        return results.get(backend, _Verification(True, version="1.0", commit="abc", gpu_device=f"{backend}0"))

    monkeypatch.setattr(runtime_module, "verify_runtime", fake_verify)
    return results


@pytest.fixture
def service(tmp_path: Path, github: FakeGitHub) -> LlamaRuntimeService:
    return LlamaRuntimeService(
        hardware=FakeHardware(hardware([RX_9070, RADEON_IGPU])),  # type: ignore[arg-type]
        client=httpx.Client(transport=httpx.MockTransport(github)),
        root=tmp_path / "runtimes",
        jobs=JobStore(),
    )


def wait_for(service: LlamaRuntimeService, job: JobInfo, timeout: float = 10) -> JobInfo:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = service.jobs.get(job.id)
        assert current is not None
        if current.status not in ("queued", "running"):
            return current
        time.sleep(0.01)
    raise AssertionError(f"job didn't finish: {service.jobs.get(job.id)}")


def stable_release(github: FakeGitHub, build_tag: str = "b11960", version: str = "v0.8.0") -> dict:
    release = github.add_release(version, {"nightly-tag.txt": f"{build_tag}\n".encode()})
    github.latest = release
    return release


def test_stable_resolves_to_the_build_it_names(service: LlamaRuntimeService, github: FakeGitHub) -> None:
    github.add_release("b11960", {"llama-b11960-bin-win-vulkan-x64.zip": b"x"})
    stable_release(github)
    available = service.available("stable")
    assert (available.tag, available.stable_version) == ("b11960", "v0.8.0")
    assert [build.id for build in available.builds] == ["llama-b11960-bin-win-vulkan-x64"]


def test_stable_falls_back_to_the_release_notes(service: LlamaRuntimeService, github: FakeGitHub) -> None:
    github.add_release("b11669", {"llama-b11669-bin-win-vulkan-x64.zip": b"x"})
    github.latest = github.add_release("v0.7.0", {}, body="Based on build b11669 of the project.")
    assert service.available("stable").tag == "b11669"


def test_stable_with_no_way_to_find_its_build(service: LlamaRuntimeService, github: FakeGitHub) -> None:
    github.latest = github.add_release("v0.6.0", {})
    with pytest.raises(RuntimeNotFoundError, match="v0.6.0"):
        service.available("stable")


def test_specific_stable_version(service: LlamaRuntimeService, github: FakeGitHub) -> None:
    github.add_release("b11669", {"llama-b11669-bin-win-vulkan-x64.zip": b"x"})
    github.add_release("v0.7.0", {"nightly-tag.txt": b"b11669\n"})
    available = service.available("v0.7.0")
    assert (available.tag, available.stable_version, available.requested) == ("b11669", "v0.7.0", "v0.7.0")


def test_a_version_with_no_release_is_a_clear_error(service: LlamaRuntimeService) -> None:
    with pytest.raises(RuntimeNotFoundError, match="no release called v0.1.0"):
        service.available("v0.1.0")


def test_latest_skips_a_release_thats_still_uploading(service: LlamaRuntimeService, github: FakeGitHub) -> None:
    github.add_release("b100", {"llama-b100-bin-win-rocm-10.0-x64.zip": b"x", "llama-b100-bin-win-cpu-x64.zip": b"x"})
    # Published, but CI hasn't uploaded the ROCm build this machine would want.
    github.add_release("b101", {"llama-b101-bin-win-cpu-x64.zip": b"x"})
    assert service.available("latest").tag == "b100"


def test_latest_skips_assets_with_unfinished_uploads(service: LlamaRuntimeService, github: FakeGitHub) -> None:
    github.add_release("b100", {"llama-b100-bin-win-rocm-10.0-x64.zip": b"x"})
    newest = github.add_release("b101", {"llama-b101-bin-win-rocm-10.0-x64.zip": b"x"})
    newest["assets"][0]["state"] = "open"
    assert service.available("latest").tag == "b100"


def test_latest_ignores_stable_tags(service: LlamaRuntimeService, github: FakeGitHub) -> None:
    github.add_release("b100", {"llama-b100-bin-win-rocm-10.0-x64.zip": b"x"})
    github.add_release("v0.8.0", {"nightly-tag.txt": b"b100\n"})
    assert service.available("latest").tag == "b100"


def test_available_marks_only_the_best_as_recommended(service: LlamaRuntimeService, github: FakeGitHub) -> None:
    github.add_release("b1", {name: b"x" for name in WINDOWS_ASSETS})
    builds = service.available("b1").builds
    assert builds[0].backend == "rocm" and builds[0].recommended
    assert sum(build.recommended for build in builds) == 1
    assert [build.backend for build in builds if build.rank is not None] == ["rocm", "vulkan", "cpu"]


def test_available_includes_cuda_dlls_in_the_size(service: LlamaRuntimeService, github: FakeGitHub) -> None:
    github.add_release(
        "b1",
        {
            "llama-b1-bin-win-cuda-13.4-x64.zip": b"12345",
            "cudart-llama-bin-win-cuda-13.4-x64.zip": b"1234567",
            "cudart-llama-bin-win-cuda-12.4-x64.zip": b"1",
        },
    )
    build = next(item for item in service.available("b1").builds if item.backend == "cuda")
    assert build.companion_asset == "cudart-llama-bin-win-cuda-13.4-x64.zip"
    assert build.size_bytes == 12


def test_linux_cuda_companion_is_matched_by_its_tagged_name(tmp_path: Path, github: FakeGitHub) -> None:
    service = LlamaRuntimeService(
        hardware=FakeHardware(hardware([nvidia("13.4")], system="linux")),  # type: ignore[arg-type]
        client=httpx.Client(transport=httpx.MockTransport(github)),
        root=tmp_path,
    )
    github.add_release(
        "b1",
        {
            "llama-b1-bin-ubuntu-cuda-13.4-x64.tar.gz": b"x",
            "cudart-llama-b1-bin-ubuntu-cuda-13.4-x64.tar.gz": b"y",
            "cudart-llama-bin-win-cuda-13.4-x64.zip": b"z",
        },
    )
    build = next(item for item in service.available("b1").builds if item.backend == "cuda")
    assert build.companion_asset == "cudart-llama-b1-bin-ubuntu-cuda-13.4-x64.tar.gz"


def test_github_responses_are_cached(service: LlamaRuntimeService, github: FakeGitHub) -> None:
    github.add_release("b1", {"llama-b1-bin-win-vulkan-x64.zip": b"x"})
    service.available("b1")
    service.available("b1")
    assert len(github.requests) == 1


def test_rate_limit_is_a_clear_error(service: LlamaRuntimeService, github: FakeGitHub) -> None:
    github.api_override = lambda request: httpx.Response(
        403, headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1800000000"}
    )
    with pytest.raises(GitHubRateLimitError, match="GITHUB_TOKEN"):
        service.available("b1")


def test_stale_cache_is_used_when_github_goes_away(
    service: LlamaRuntimeService, github: FakeGitHub, monkeypatch: pytest.MonkeyPatch
) -> None:
    github.add_release("b1", {"llama-b1-bin-win-vulkan-x64.zip": b"x"})
    service.available("b1")
    monkeypatch.setattr(runtime_module, "RELEASE_CACHE_SECONDS", 0)
    github.api_override = lambda request: httpx.Response(500)
    assert service.available("b1").tag == "b1"


def test_github_down_with_nothing_cached(service: LlamaRuntimeService, github: FakeGitHub) -> None:
    github.api_override = lambda request: httpx.Response(500)
    with pytest.raises(GitHubUnavailableError):
        service.available("b1")


def test_auto_install_activates_the_best_build(
    service: LlamaRuntimeService, github: FakeGitHub, verifier: dict
) -> None:
    github.add_release("b1", {"llama-b1-bin-win-vulkan-x64.zip": server_zip()})
    stable_release(github, "b1")
    job = wait_for(service, service.start_install())
    assert job.status == "completed", job.error
    assert job.result["runtime_id"] == "llama-b1-bin-win-vulkan-x64"
    assert job.result["activated"] is True
    assert job.result["stable_version"] == "v0.8.0"
    assert job.bytes_done == job.bytes_total

    installed = service.list_installed()
    assert installed.active_id == "llama-b1-bin-win-vulkan-x64"
    assert installed.runtimes[0].verified is True
    assert installed.runtimes[0].stable_version == "v0.8.0"
    assert installed.runtimes[0].installed_at is not None
    # Nothing is left behind but the build and the record of which is active.
    assert sorted(path.name for path in service.root.iterdir()) == [
        ".downloads",
        "llama-b1-bin-win-vulkan-x64",
        "state.json",
    ]
    assert list((service.root / ".downloads").iterdir()) == []


def test_auto_install_falls_back_when_a_build_finds_no_gpu(
    service: LlamaRuntimeService, github: FakeGitHub, verifier: dict
) -> None:
    verifier["rocm"] = _Verification(False, error="This rocm build found no GPU on this machine.")
    github.add_release(
        "b1",
        {
            "llama-b1-bin-win-rocm-10.0-x64.zip": server_zip(),
            "llama-b1-bin-win-vulkan-x64.zip": server_zip(),
        },
    )
    stable_release(github, "b1")
    job = wait_for(service, service.start_install())
    assert job.status == "completed", job.error
    assert job.result["runtime_id"] == "llama-b1-bin-win-vulkan-x64"
    assert any("found no GPU" in note for note in job.notes)
    # The build that failed is gone, not left around taking up gigabytes.
    assert [runtime.id for runtime in service.list_installed().runtimes] == ["llama-b1-bin-win-vulkan-x64"]


def test_auto_install_fails_clearly_if_nothing_works(
    service: LlamaRuntimeService, github: FakeGitHub, verifier: dict
) -> None:
    failure = _Verification(False, error="nope")
    verifier.update(vulkan=failure, cpu=failure)
    github.add_release(
        "b1",
        {"llama-b1-bin-win-vulkan-x64.zip": server_zip(), "llama-b1-bin-win-cpu-x64.zip": server_zip()},
    )
    stable_release(github, "b1")
    job = wait_for(service, service.start_install())
    assert job.status == "failed"
    assert "No build worked" in job.error
    assert service.list_installed().active_id is None


def test_explicit_backend_is_kept_but_not_activated_when_it_finds_no_gpu(
    service: LlamaRuntimeService, github: FakeGitHub, verifier: dict
) -> None:
    verifier["rocm"] = _Verification(False, error="found no GPU")
    github.add_release(
        "b1",
        {
            "llama-b1-bin-win-rocm-10.0-x64.zip": server_zip(),
            "llama-b1-bin-win-vulkan-x64.zip": server_zip(),
        },
    )
    job = wait_for(service, service.start_install("b1", "rocm"))
    assert job.status == "completed", job.error
    assert job.result["activated"] is False
    assert job.result["verified"] is False
    installed = service.list_installed()
    assert installed.active_id is None
    assert [(runtime.id, runtime.verified) for runtime in installed.runtimes] == [
        ("llama-b1-bin-win-rocm-10.0-x64", False)
    ]


def test_explicit_backend_with_version(service: LlamaRuntimeService, github: FakeGitHub, verifier: dict) -> None:
    github.add_release(
        "b1",
        {
            "llama-b1-bin-win-cuda-12.4-x64.zip": server_zip(),
            "llama-b1-bin-win-cuda-13.4-x64.zip": server_zip(),
        },
    )
    job = wait_for(service, service.start_install("b1", "cuda-12.4"))
    assert job.result["runtime_id"] == "llama-b1-bin-win-cuda-12.4-x64"


def test_unknown_backend_lists_whats_on_offer(service: LlamaRuntimeService, github: FakeGitHub) -> None:
    github.add_release("b1", {"llama-b1-bin-win-vulkan-x64.zip": server_zip()})
    with pytest.raises(RuntimeNotFoundError, match="Available: vulkan"):
        service.start_install("b1", "rocm")


def test_cuda_install_merges_its_dlls_into_the_build(
    service: LlamaRuntimeService, github: FakeGitHub, verifier: dict
) -> None:
    github.add_release(
        "b1",
        {
            "llama-b1-bin-win-cuda-13.4-x64.zip": server_zip(),
            "cudart-llama-bin-win-cuda-13.4-x64.zip": server_zip({"cudart64_13.dll": b"dll"}),
        },
    )
    job = wait_for(service, service.start_install("b1", "cuda"))
    assert job.status == "completed", job.error
    assert (service.root / "llama-b1-bin-win-cuda-13.4-x64" / "cudart64_13.dll").read_bytes() == b"dll"


def test_nested_archive_is_flattened(tmp_path: Path, github: FakeGitHub, verifier: dict) -> None:
    """Linux tarballs wrap everything in a `llama-b1/` folder."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        info = tarfile.TarInfo(f"llama-b1/{server_binary_name()}")
        info.size = 1
        bundle.addfile(info, io.BytesIO(b"x"))
    system = "windows" if sys.platform == "win32" else "linux"
    service = LlamaRuntimeService(
        hardware=FakeHardware(hardware([], system=system)),  # type: ignore[arg-type]
        client=httpx.Client(transport=httpx.MockTransport(github)),
        root=tmp_path / "runtimes",
    )
    # A `.zip` name would change nothing: what's in the file decides how it's read.
    name = "llama-b1-bin-win-cpu-x64.tar.gz" if system == "windows" else "llama-b1-bin-ubuntu-x64.tar.gz"
    github.add_release("b1", {name: buffer.getvalue()})
    job = wait_for(service, service.start_install("b1"))
    assert job.status == "completed", job.error
    assert (service.root / job.result["runtime_id"] / server_binary_name()).is_file()


def test_archive_without_a_server_fails_clearly(
    service: LlamaRuntimeService, github: FakeGitHub, verifier: dict
) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr("readme.txt", b"x")
    github.add_release("b1", {"llama-b1-bin-win-vulkan-x64.zip": buffer.getvalue()})
    job = wait_for(service, service.start_install("b1", "vulkan"))
    assert job.status == "failed"
    assert server_binary_name() in job.error
    assert not (service.root / "llama-b1-bin-win-vulkan-x64").exists()


def test_a_corrupted_download_fails_and_is_removed(
    service: LlamaRuntimeService, github: FakeGitHub, verifier: dict
) -> None:
    github.add_release("b1", {"llama-b1-bin-win-vulkan-x64.zip": server_zip()})
    github.files["llama-b1-bin-win-vulkan-x64.zip"] = b"tampered"
    job = wait_for(service, service.start_install("b1", "vulkan"))
    assert job.status == "failed"
    assert "SHA-256" in job.error
    assert list((service.root / ".downloads").iterdir()) == []
    assert not (service.root / "llama-b1-bin-win-vulkan-x64").exists()


def test_malicious_archive_is_refused_and_leaves_nothing(
    service: LlamaRuntimeService, github: FakeGitHub, verifier: dict, tmp_path: Path
) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr(server_binary_name(), b"x")
        bundle.writestr("../../evil.dll", b"y")
    github.add_release("b1", {"llama-b1-bin-win-vulkan-x64.zip": buffer.getvalue()})
    job = wait_for(service, service.start_install("b1", "vulkan"))
    assert job.status == "failed"
    assert "escapes" in job.error
    assert not (service.root / "llama-b1-bin-win-vulkan-x64").exists()
    assert not list(tmp_path.rglob("evil.dll"))
    assert not list(service.root.glob(".tmp-*"))


def test_download_resumes_from_a_partial_file(
    service: LlamaRuntimeService, github: FakeGitHub, verifier: dict
) -> None:
    content = server_zip({"padding.bin": b"z" * 5000})
    github.add_release("b1", {"llama-b1-bin-win-vulkan-x64.zip": content})
    partial = service.root / ".downloads" / "b1-llama-b1-bin-win-vulkan-x64.zip.partial"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(content[:100])

    job = wait_for(service, service.start_install("b1", "vulkan"))
    assert job.status == "completed", job.error
    download = next(request for request in github.requests if "downloads.test" in str(request.url))
    assert download.headers["Range"] == "bytes=100-"


def test_a_finished_partial_file_isnt_downloaded_again(
    service: LlamaRuntimeService, github: FakeGitHub, verifier: dict
) -> None:
    content = server_zip()
    github.add_release("b1", {"llama-b1-bin-win-vulkan-x64.zip": content})
    partial = service.root / ".downloads" / "b1-llama-b1-bin-win-vulkan-x64.zip.partial"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(content)
    job = wait_for(service, service.start_install("b1", "vulkan"))
    assert job.status == "completed", job.error


def test_a_partial_longer_than_the_file_starts_over(
    service: LlamaRuntimeService, github: FakeGitHub, verifier: dict
) -> None:
    content = server_zip()
    github.add_release("b1", {"llama-b1-bin-win-vulkan-x64.zip": content})
    partial = service.root / ".downloads" / "b1-llama-b1-bin-win-vulkan-x64.zip.partial"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(content + b"stale leftovers")
    job = wait_for(service, service.start_install("b1", "vulkan"))
    assert job.status == "completed", job.error


def test_not_enough_disk_space_fails_before_downloading(
    service: LlamaRuntimeService, github: FakeGitHub, verifier: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    github.add_release("b1", {"llama-b1-bin-win-vulkan-x64.zip": server_zip()})

    class Usage:
        free = 1

    monkeypatch.setattr(runtime_module.shutil, "disk_usage", lambda path: Usage)
    job = wait_for(service, service.start_install("b1", "vulkan"))
    assert job.status == "failed"
    assert "Not enough free space" in job.error
    assert not any("downloads.test" in str(request.url) for request in github.requests)


def test_installing_whats_already_there_just_activates_it(
    service: LlamaRuntimeService, github: FakeGitHub, verifier: dict
) -> None:
    github.add_release("b1", {"llama-b1-bin-win-vulkan-x64.zip": server_zip()})
    wait_for(service, service.start_install("b1", "vulkan", activate=False))
    github.requests.clear()
    job = wait_for(service, service.start_install("b1", "vulkan"))
    assert job.result["already_installed"] is True
    assert job.result["activated"] is True
    assert not any("downloads.test" in str(request.url) for request in github.requests)


def test_auto_never_deletes_an_install_that_was_already_there(
    service: LlamaRuntimeService, github: FakeGitHub, verifier: dict
) -> None:
    """Falling back throws away what it just downloaded, and nothing older."""
    existing = service.root / "llama-b1-bin-win-rocm-10.0-x64"
    existing.mkdir(parents=True)
    (existing / server_binary_name()).write_bytes(b"x")
    verifier["rocm"] = _Verification(False, error="found no GPU")
    github.add_release(
        "b1",
        {
            "llama-b1-bin-win-rocm-10.0-x64.zip": server_zip(),
            "llama-b1-bin-win-vulkan-x64.zip": server_zip(),
        },
    )
    stable_release(github, "b1")
    job = wait_for(service, service.start_install())
    assert job.result["runtime_id"] == "llama-b1-bin-win-vulkan-x64"
    assert existing.is_dir()


def test_repeat_install_request_joins_the_running_job(
    service: LlamaRuntimeService, github: FakeGitHub, verifier: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    github.add_release("b1", {"llama-b1-bin-win-vulkan-x64.zip": server_zip()})
    release = threading.Event()
    monkeypatch.setattr(
        service, "_install", lambda *args: (release.wait(5), {"runtime_id": "x"})[1]
    )
    first = service.start_install("b1", "vulkan")
    second = service.start_install("b1", "vulkan")
    assert first.id == second.id
    release.set()
    wait_for(service, first)


def test_cancelling_keeps_the_partial_download_and_installing_again_resumes(
    service: LlamaRuntimeService, github: FakeGitHub, verifier: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Stored uncompressed and bigger than a chunk, so it arrives in several.
    content = io.BytesIO()
    with zipfile.ZipFile(content, "w", zipfile.ZIP_STORED) as bundle:
        bundle.writestr(server_binary_name(), b"binary")
        bundle.writestr("pad.bin", os.urandom(3 * DOWNLOAD_CHUNK_BYTES))
    github.add_release("b1", {"llama-b1-bin-win-vulkan-x64.zip": content.getvalue()})
    partial = service.root / ".downloads" / "b1-llama-b1-bin-win-vulkan-x64.zip.partial"

    original_progress = JobContext.progress
    cancelled = threading.Event()

    def cancel_after_first_chunk(context: JobContext, bytes_done: int) -> None:
        original_progress(context, bytes_done)
        if bytes_done > 0 and not cancelled.is_set():
            cancelled.set()
            context._record.cancel.set()

    monkeypatch.setattr(JobContext, "progress", cancel_after_first_chunk)
    job = wait_for(service, service.start_install("b1", "vulkan"))
    assert job.status == "cancelled"
    assert not (service.root / "llama-b1-bin-win-vulkan-x64").exists()
    assert service.list_installed().active_id is None
    kept = partial.stat().st_size
    assert 0 < kept < len(content.getvalue())

    monkeypatch.setattr(JobContext, "progress", original_progress)
    github.requests.clear()
    job = wait_for(service, service.start_install("b1", "vulkan"))
    assert job.status == "completed", job.error
    download = next(request for request in github.requests if "downloads.test" in str(request.url))
    assert download.headers["Range"] == f"bytes={kept}-"
    # The checksum in `add_release` covers the whole file, so this proves the
    # resumed download joined up correctly.
    assert (service.root / "llama-b1-bin-win-vulkan-x64" / "pad.bin").stat().st_size == 3 * DOWNLOAD_CHUNK_BYTES


def test_cancelling_a_finished_job_changes_nothing(
    service: LlamaRuntimeService, github: FakeGitHub, verifier: dict
) -> None:
    github.add_release("b1", {"llama-b1-bin-win-vulkan-x64.zip": server_zip()})
    job = wait_for(service, service.start_install("b1", "vulkan"))
    after = service.jobs.cancel(job.id)
    assert after.status == "completed"
    assert after.cancel_requested is False


def make_installed(service: LlamaRuntimeService, name: str, manifest: dict | None = None) -> Path:
    path = service.root / name
    path.mkdir(parents=True)
    (path / server_binary_name()).write_bytes(b"x")
    if manifest is not None:
        (path / "localai-runtime.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_existing_folders_with_no_manifest_are_tested_once(
    service: LlamaRuntimeService, verifier: dict, github: FakeGitHub
) -> None:
    make_installed(service, "llama-b10878-bin-win-vulkan-x64")
    first = service.list_installed().runtimes[0]
    assert first.verified is True
    assert first.gpu_device == "vulkan0"
    assert (service.root / "llama-b10878-bin-win-vulkan-x64" / "localai-runtime.json").is_file()

    verifier["vulkan"] = _Verification(False, error="would only show if tested again")
    assert service.list_installed().runtimes[0].verified is True
    assert service.list_installed(refresh=True).runtimes[0].verified is False


def test_build_for_another_os_is_listed_but_never_run(
    service: LlamaRuntimeService, verifier: dict, github: FakeGitHub
) -> None:
    make_installed(service, "llama-b11024-bin-ubuntu-rocm-10.0-x64")
    runtime = service.list_installed().runtimes[0]
    assert runtime.compatible is False
    assert runtime.verified is None


def test_listing_ignores_other_folders_and_sorts_newest_first(
    service: LlamaRuntimeService, verifier: dict, github: FakeGitHub
) -> None:
    make_installed(service, "llama-b10-bin-win-vulkan-x64")
    make_installed(service, "llama-b20-bin-win-vulkan-x64")
    (service.root / ".downloads").mkdir()
    (service.root / ".tmp-llama-b30-bin-win-vulkan-x64").mkdir()
    (service.root / "something-else").mkdir()
    assert [runtime.build for runtime in service.list_installed().runtimes] == [20, 10]


def test_listing_works_with_no_folder_yet(service: LlamaRuntimeService, github: FakeGitHub) -> None:
    assert service.list_installed().runtimes == []


def test_listing_works_offline(service: LlamaRuntimeService, verifier: dict, github: FakeGitHub) -> None:
    make_installed(service, "llama-b10-bin-win-vulkan-x64")
    github.api_override = lambda request: httpx.Response(500)
    installed = service.list_installed()
    assert len(installed.runtimes) == 1
    assert installed.update_available is None


def test_updates_are_for_the_backend_in_use(
    service: LlamaRuntimeService, verifier: dict, github: FakeGitHub
) -> None:
    make_installed(service, "llama-b10-bin-win-vulkan-x64")
    service.set_active("llama-b10-bin-win-vulkan-x64")
    github.add_release("b20", {"llama-b20-bin-win-vulkan-x64.zip": b"x"})
    stable_release(github, "b20")
    # Newer, but with no Vulkan build yet: it can't be offered as an update.
    github.add_release("b30", {"llama-b30-bin-win-cpu-x64.zip": b"x"})
    updates = service.list_installed().update_available
    assert (updates.stable_tag, updates.stable_version, updates.stable_is_newer) == ("b20", "v0.8.0", True)
    assert (updates.latest_tag, updates.latest_is_newer) == ("b20", True)


def test_set_active_and_delete_rules(service: LlamaRuntimeService, verifier: dict, github: FakeGitHub) -> None:
    first = make_installed(service, "llama-b10-bin-win-vulkan-x64")
    make_installed(service, "llama-b20-bin-win-vulkan-x64")
    assert service.set_active(first.name).active_id == first.name

    with pytest.raises(RuntimeConflictError, match="active"):
        service.delete(first.name)
    assert first.is_dir()

    service.set_active("llama-b20-bin-win-vulkan-x64")
    service.delete(first.name)
    assert not first.exists()


def test_cant_activate_a_build_for_another_os(service: LlamaRuntimeService, verifier: dict) -> None:
    make_installed(service, "llama-b1-bin-ubuntu-rocm-10.0-x64")
    with pytest.raises(RuntimeConflictError, match="linux-x64"):
        service.set_active("llama-b1-bin-ubuntu-rocm-10.0-x64")


def test_cant_activate_a_build_that_cant_see_the_gpu(service: LlamaRuntimeService, verifier: dict) -> None:
    make_installed(service, "llama-b1-bin-win-rocm-10.0-x64")
    verifier["rocm"] = _Verification(False, error="found no GPU")
    with pytest.raises(RuntimeConflictError, match="found no GPU"):
        service.set_active("llama-b1-bin-win-rocm-10.0-x64")


def test_a_build_that_failed_is_tested_again_before_refusing(service: LlamaRuntimeService, verifier: dict) -> None:
    """The SDK may have been installed since it last failed."""
    make_installed(service, "llama-b1-bin-win-rocm-10.0-x64", {"verified": False, "verification_error": "old"})
    assert service.set_active("llama-b1-bin-win-rocm-10.0-x64").active_id == "llama-b1-bin-win-rocm-10.0-x64"


@pytest.mark.parametrize(
    "runtime_id",
    [
        "..",
        "../..",
        "..\\..",
        "llama-b1-bin-win-x64/../../outside",
        "llama-b1-bin-win-x64-/../../outside",
        "llama-b1-bin-win-x64-..",
        "llama-b1-bin-win-vulkan-x64.zip",
        "state.json",
        "something-else",
        "C:\\Windows",
        "/etc",
        "",
    ],
)
def test_delete_cant_reach_anything_but_a_runtime(
    service: LlamaRuntimeService, tmp_path: Path, runtime_id: str
) -> None:
    service.root.mkdir(parents=True)
    (service.root / "something-else").mkdir()
    (service.root / "state.json").write_text("{}")
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(RuntimeNotFoundError):
        service.delete(runtime_id)
    assert outside.is_dir()
    assert (service.root / "something-else").is_dir()


def test_delete_missing_runtime(service: LlamaRuntimeService) -> None:
    with pytest.raises(RuntimeNotFoundError):
        service.delete("llama-b1-bin-win-vulkan-x64")


def test_parsed_build_is_frozen_and_comparable() -> None:
    assert parse_build_name("llama-b1-bin-win-cpu-x64.zip") == ParsedBuild(
        id="llama-b1-bin-win-cpu-x64", tag="b1", build=1, os="windows", arch="x64", backend="cpu"
    )


def test_wrapper_folder_of_a_hand_extracted_tarball_is_seen_through(
    service: LlamaRuntimeService, verifier: dict, github: FakeGitHub
) -> None:
    path = service.root / "llama-b1-bin-win-vulkan-x64"
    (path / "llama-b1").mkdir(parents=True)
    (path / "llama-b1" / server_binary_name()).write_bytes(b"x")
    assert server_dir(path) == path / "llama-b1"
    runtime = service.list_installed().runtimes[0]
    assert runtime.verified is True
    assert runtime.server_path == str(path / "llama-b1" / server_binary_name())
    # The manifest written beside the wrapper folder mustn't stop it being found.
    assert (path / "localai-runtime.json").is_file()
    assert service.list_installed().runtimes[0].server_path == runtime.server_path
    assert service.set_active(path.name).active_id == path.name


def test_folder_with_no_server_cant_be_activated_and_isnt_a_crash(
    service: LlamaRuntimeService, verifier: dict, github: FakeGitHub
) -> None:
    path = service.root / "llama-b1-bin-win-vulkan-x64"
    path.mkdir(parents=True)
    (path / "readme.txt").write_bytes(b"x")
    assert server_dir(path) is None
    runtime = service.list_installed().runtimes[0]
    assert (runtime.verified, runtime.server_path) == (None, None)
    with pytest.raises(RuntimeConflictError, match=server_binary_name()):
        service.set_active(path.name)


def test_a_failure_message_keeps_its_end_where_the_cause_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real loader error: a long path, then the library that's actually missing."""
    message = (
        "/mnt/c/" + "very-long-folder/" * 30 + "llama-server: error while loading shared libraries: "
        "libgomp.so.1: cannot open shared object file: No such file or directory"
    )
    monkeypatch.setattr(runtime_module, "_run_server_command", lambda *args: (127, message))
    result = verify_runtime(tmp_path, "cpu", [])
    assert result.verified is False
    assert "exit code 127" in result.error
    assert "libgomp.so.1: cannot open shared object file" in result.error


def test_verify_runtime_needs_a_gpu_build_to_find_a_gpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    answers = {
        "--version": (0, "version: 0.8.0-dev (build 11960, commit 8642541b2)"),
        "--list-devices": (0, "Available devices:\n  (none)\n"),
    }
    monkeypatch.setattr(runtime_module, "_run_server_command", lambda path, backend, flag: answers[flag])
    no_gpu = verify_runtime(tmp_path, "rocm", [RX_9070])
    assert no_gpu.verified is False
    assert "found no GPU" in no_gpu.error and "ROCm SDK" in no_gpu.error
    assert (no_gpu.version, no_gpu.commit) == ("0.8.0-dev", "8642541b2")
    # A CPU build has no GPU to find.
    assert verify_runtime(tmp_path, "cpu", []).verified is True

    answers["--list-devices"] = (0, ROCM_OUTPUT)
    found = verify_runtime(tmp_path, "rocm", [RX_9070, RADEON_IGPU])
    assert (found.verified, found.gpu_device) == (True, "ROCm1")


# Running models


def test_the_build_to_run_with_is_the_active_one_and_its_gpu(service: LlamaRuntimeService, verifier: dict) -> None:
    path = make_installed(service, "llama-b10-bin-win-vulkan-x64")
    service.set_active(path.name)
    runtime = service.launchable()
    assert (runtime.id, runtime.directory, runtime.backend, runtime.gpu_device) == (
        path.name,
        path,
        "vulkan",
        "vulkan0",
    )


def test_a_build_can_be_named_without_being_activated(service: LlamaRuntimeService, verifier: dict) -> None:
    path = make_installed(service, "llama-b10-bin-win-rocm-10.0-x64")
    with pytest.raises(RuntimeNotFoundError, match="No llama.cpp runtime is active"):
        service.launchable()
    assert service.launchable(path.name).id == path.name
    assert service.list_installed().active_id is None


def test_a_build_that_cant_run_here_isnt_launched(service: LlamaRuntimeService, verifier: dict) -> None:
    make_installed(service, "llama-b1-bin-ubuntu-rocm-10.0-x64")
    with pytest.raises(RuntimeConflictError, match="linux-x64"):
        service.launchable("llama-b1-bin-ubuntu-rocm-10.0-x64")
    make_installed(service, "llama-b1-bin-win-rocm-10.0-x64")
    verifier["rocm"] = _Verification(False, error="found no GPU")
    with pytest.raises(RuntimeConflictError, match="found no GPU"):
        service.launchable("llama-b1-bin-win-rocm-10.0-x64")
    with pytest.raises(RuntimeNotFoundError):
        service.launchable("llama-b2-bin-win-rocm-10.0-x64")


def test_a_build_in_use_cant_be_deleted(service: LlamaRuntimeService, verifier: dict) -> None:
    path = make_installed(service, "llama-b10-bin-win-vulkan-x64")
    in_use = {path.name}
    service.add_busy_check(lambda runtime_id: f"{runtime_id} is running a model" if runtime_id in in_use else None)
    with pytest.raises(RuntimeConflictError, match="is running a model"):
        service.delete(path.name)
    assert (path / server_binary_name()).is_file()
    in_use.clear()
    service.delete(path.name)
    assert not path.exists()


# Routes


@pytest.fixture
def client(service: LlamaRuntimeService, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(runtime_router, "service", service)
    return TestClient(app)


def events(client: TestClient, path: str) -> list[dict]:
    with client.stream("GET", path) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        return [json.loads(line.removeprefix("data: ")) for line in response.iter_lines() if line.startswith("data: ")]


def test_an_install_can_be_followed_and_is_listed_over_http(
    client: TestClient, github: FakeGitHub, verifier: dict
) -> None:
    github.add_release("b1", {"llama-b1-bin-win-vulkan-x64.zip": server_zip()})
    stable_release(github, "b1")
    started = client.post("/runtimes/llama.cpp/install")
    assert started.status_code == 202
    job_id = started.json()["id"]

    followed = events(client, f"/runtimes/llama.cpp/jobs/{job_id}/events")
    assert {event["id"] for event in followed} == {job_id}
    assert followed[-1]["status"] == "completed", followed[-1]["error"]
    assert followed[-1]["result"]["runtime_id"] == "llama-b1-bin-win-vulkan-x64"
    assert all(event["status"] in ("queued", "running") for event in followed[:-1])
    assert followed[-1] == client.get(f"/runtimes/llama.cpp/jobs/{job_id}").json()

    # Once it's done, following it gives it as it finished, and nothing else.
    assert events(client, f"/runtimes/llama.cpp/jobs/{job_id}/events") == [followed[-1]]
    assert client.get("/runtimes/llama.cpp/jobs").json() == [followed[-1]]


def test_install_jobs_are_listed_newest_first_over_http(client: TestClient, service: LlamaRuntimeService) -> None:
    assert client.get("/runtimes/llama.cpp/jobs").json() == []
    first = service.jobs.submit("test", "first", lambda context: None)
    second = service.jobs.submit("test", "second", lambda context: None)
    assert [job["id"] for job in client.get("/runtimes/llama.cpp/jobs").json()] == [second.id, first.id]


def test_unknown_install_jobs_are_404_over_http(client: TestClient) -> None:
    assert client.get("/runtimes/llama.cpp/jobs/nope").status_code == 404
    assert client.get("/runtimes/llama.cpp/jobs/nope/events").status_code == 404
    assert client.delete("/runtimes/llama.cpp/jobs/nope").status_code == 404

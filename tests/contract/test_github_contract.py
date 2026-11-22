"""Checks that GitHub still answers the way the stand-in in `test_llama_runtime.py`
pretends it does, using llama.cpp's real stable release.

The regular tests can only be as right as their stand-ins. And where a field
or file is missing the app carries on with nothing rather than failing: a
renamed CUDA build would just stop being offered, and an asset without a
checksum would be installed unchecked. So a change on GitHub's side would pass
every other test. This needs the internet, makes a few API calls and downloads
a few bytes; it runs on its own (`-m contract`)."""

from __future__ import annotations

import hashlib
import re
from typing import Any

import pytest

from models.hardware import GpuApis, GpuInfo, HardwareInfo
from services.llama_runtime import STABLE_TAG_ASSET, LlamaRuntimeService, _digest_of, parse_build_name

from ..test_llama_runtime import FakeGitHub, FakeHardware, asset, hardware, nvidia

pytestmark = pytest.mark.contract

DOWNLOADS = "https://github.com/ggml-org/llama.cpp/releases/download/"
# The builds the app picks for the machines it supports. Losing one means a
# machine like that silently gets a worse build, or none.
EXPECTED_BUILDS = {
    ("windows", "x64", "cpu"),
    ("windows", "x64", "vulkan"),
    ("windows", "x64", "cuda"),
    ("linux", "x64", "cpu"),
    ("linux", "x64", "vulkan"),
    ("linux", "x64", "cuda"),
    ("macos", "arm64", "metal"),
}
APPLE = GpuInfo(name="Apple M3", vendor="apple", kind="unified", apis=GpuApis(metal=True))


@pytest.fixture(scope="module")
def service(tmp_path_factory: pytest.TempPathFactory) -> LlamaRuntimeService:
    """The real service with its real client, using GITHUB_TOKEN when it's set.
    Its cache means each release is fetched once for the whole module."""
    return LlamaRuntimeService(hardware=FakeHardware(hardware([])), root=tmp_path_factory.mktemp("runtimes"))


@pytest.fixture(scope="module")
def stable(service: LlamaRuntimeService) -> dict[str, Any]:
    return service._github_json("/releases/latest")


@pytest.fixture(scope="module")
def build(service: LlamaRuntimeService, stable: dict[str, Any]) -> dict[str, Any]:
    """The build the stable release is made from, which holds the downloads."""
    return service._github_json(f"/releases/tags/{service._stable_build_tag(stable)}")


def test_the_stable_release_names_the_build_it_is(service: LlamaRuntimeService, stable: dict[str, Any]) -> None:
    assert re.fullmatch(r"v\d+\.\d+\.\d+", stable["tag_name"]), stable["tag_name"]
    assert re.fullmatch(r"b\d+", service._stable_build_tag(stable))


def test_releases_have_every_field_the_stand_in_gives_them(stable: dict[str, Any], build: dict[str, Any]) -> None:
    fake_release = FakeGitHub().add_release("b1", {})
    for release in (stable, build):
        assert set(fake_release) <= set(release), set(fake_release) - set(release)
        assert isinstance(release["draft"], bool)
        assert isinstance(release["published_at"], str)
    # Shown as the commit an installed build was made from.
    assert re.fullmatch(r"[0-9a-f]{40}", build["target_commitish"]), build["target_commitish"]


def test_every_file_has_a_size_a_link_and_a_checksum(stable: dict[str, Any], build: dict[str, Any]) -> None:
    fake_asset = asset("llama-b1-bin-win-cpu-x64.zip", b"x")
    files = stable["assets"] + build["assets"]
    assert files
    for item in files:
        assert set(fake_asset) <= set(item), (item["name"], set(fake_asset) - set(item))
        assert isinstance(item["size"], int) and item["size"] > 0, item["name"]
        assert item["state"] == "uploaded", item["name"]
        assert item["browser_download_url"].startswith(DOWNLOADS), item["browser_download_url"]
        # Without one, an install isn't checked for corruption.
        assert _digest_of(item) and re.fullmatch(r"[0-9a-f]{64}", _digest_of(item) or ""), item["name"]


def test_a_files_checksum_is_the_one_of_its_contents(service: LlamaRuntimeService, stable: dict[str, Any]) -> None:
    item = next(item for item in stable["assets"] if item["name"] == STABLE_TAG_ASSET)
    response = service.client.get(item["browser_download_url"], follow_redirects=True)
    response.raise_for_status()
    assert len(response.content) == item["size"]
    assert hashlib.sha256(response.content).hexdigest() == _digest_of(item)


def test_a_download_can_resume_where_it_stopped(service: LlamaRuntimeService, build: dict[str, Any]) -> None:
    item = min(build["assets"], key=lambda candidate: candidate["size"])
    headers = {"Range": "bytes=10-19"}
    response = service.client.get(item["browser_download_url"], headers=headers, follow_redirects=True)
    assert response.status_code == 206, response.status_code
    assert response.headers["Content-Range"] == f"bytes 10-19/{item['size']}"
    assert len(response.content) == 10


def test_the_builds_the_app_offers_are_all_there(build: dict[str, Any]) -> None:
    names = [item["name"] for item in build["assets"]]
    archives = [name for name in names if name.startswith("llama-b") and "-bin-" in name]
    # A build named in a way the app can't read would just be left out.
    unreadable = [name for name in archives if parse_build_name(name) is None]
    assert not unreadable, unreadable
    found = {(parsed.os, parsed.arch, parsed.backend) for parsed in map(parse_build_name, archives) if parsed}
    assert not EXPECTED_BUILDS - found, EXPECTED_BUILDS - found


def test_every_cuda_build_comes_with_its_cuda_libraries(service: LlamaRuntimeService, build: dict[str, Any]) -> None:
    release = service._to_resolved(build, "stable", None)
    for os_name, arch in (("windows", "x64"), ("linux", "x64")):
        cuda = [parsed for parsed in release.builds_for(os_name, arch) if parsed.backend == "cuda"]
        assert cuda, f"no CUDA build for {os_name} {arch}"
        alone = [parsed.id for parsed in cuda if release.companion_for(parsed) is None]
        assert not alone, f"no cudart download next to {alone}"


@pytest.mark.parametrize(
    ("machine", "backend"),
    [
        (hardware([nvidia("12.8")], "windows", "x64"), "cuda"),
        (hardware([nvidia("12.8")], "linux", "x64"), "cuda"),
        (hardware([], "windows", "x64"), "cpu"),
        # What CI's own machines are.
        (hardware([], "linux", "x64"), "cpu"),
        (hardware([APPLE], "macos", "arm64"), "metal"),
    ],
    ids=["windows-nvidia", "linux-nvidia", "windows-no-gpu", "linux-no-gpu", "mac"],
)
def test_the_app_recommends_the_right_build_for_a_machine(
    service: LlamaRuntimeService,
    monkeypatch: pytest.MonkeyPatch,
    machine: HardwareInfo,
    backend: str,
) -> None:
    monkeypatch.setattr(service, "hardware", FakeHardware(machine))
    offered = service.available("stable")
    recommended = [item for item in offered.builds if item.recommended]
    assert [item.backend for item in recommended] == [backend], [(item.id, item.reason) for item in offered.builds]
    if backend == "cuda":
        # The driver supports CUDA 12.8, so the build has to be for that or older.
        version = tuple(int(part) for part in (recommended[0].backend_version or "").split("."))
        assert version <= (12, 8), recommended[0].id
        assert recommended[0].companion_asset

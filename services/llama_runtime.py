"""Finds, installs and switches between llama.cpp builds.

Three things about llama.cpp's releases shape this:

- Every tag ships dozens of builds, so a runtime's id is its release asset name
  (`llama-b11960-bin-win-vulkan-x64`), which is also its folder name.
- Stable releases (`v0.8.0`) contain no binaries, just `nightly-tag.txt` naming
  the build (`b11960`) they correspond to.
- Having the right GPU isn't enough for a build to use it: on Windows a ROCm
  build reports no devices without the ROCm SDK. So every install is run and
  asked what it can see, and that answer is what counts.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import threading
import time
import zipfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from models.hardware import GpuInfo, HardwareInfo
from models.jobs import JobInfo
from models.runtime import (
    AvailableBuild,
    AvailableBuilds,
    InstalledRuntime,
    InstalledRuntimes,
    RuntimeDevice,
    RuntimeUpdates,
)
from services.hardware import HardwareService, hardware_service, rocm_sdk_directories
from services.jobs import JobContext, JobStore, download_file
from services.settings import github_token, llama_runtimes_dir

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com/repos/ggml-org/llama.cpp"
RELEASE_CACHE_SECONDS = 600
RECENT_RELEASES = 30
STABLE_TAG_ASSET = "nightly-tag.txt"
MANIFEST_NAME = "localai-runtime.json"
STATE_NAME = "state.json"
DOWNLOADS_DIR = ".downloads"
# Room to ask for on top of the download: a ROCm archive grows 4x when extracted
# (266 MB to 1.1 GB), and the download itself sits alongside until that's done.
ARCHIVE_SPACE_FACTOR = 5
VERIFY_TIMEOUT_SECONDS = 60
VERIFY_WORKERS = 4

_BUILD_NAME = re.compile(
    r"^llama-(?P<tag>b(?P<build>\d+))-bin-(?P<os>win|ubuntu|linux|macos|android)-(?P<rest>.+)$"
)
_COMPANION_NAME = re.compile(
    r"^cudart-llama(?:-b\d+)?-bin-(?P<os>win|ubuntu|linux)-cuda-(?P<version>[\d.]+)-(?P<arch>[a-z0-9]+)"
    r"\.(?:zip|tar\.gz)$"
)
_ARCHIVE_SUFFIX = re.compile(r"\.(?:zip|tar\.gz)$")
_SAFE_RUNTIME_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_DEVICE_LINE = re.compile(r"^\s*(?P<id>[^\s:]+):\s+(?P<name>.+?)\s+\((?P<total>\d+) MiB,\s*\d+ MiB free\)\s*$")
_VERSION_LINE = re.compile(r"version:\s*(?P<version>\S+)\s*\(build\s*(?P<build>\d+),\s*commit\s*(?P<commit>\w+)\)")
_STABLE_BUILD_TAG = re.compile(r"\bb\d{4,6}\b")

ARCHES = {"x64", "arm64", "s390x"}
OS_BY_TOKEN = {"win": "windows", "ubuntu": "linux", "linux": "linux", "macos": "macos", "android": "android"}

# Why `auto` never picks these, for builds that aren't in the ranked order.
_UNRANKED_REASONS = {
    "cuda": "Needs an NVIDIA GPU",
    "rocm": "Needs an AMD GPU and the ROCm runtime",
    "vulkan": "No usable Vulkan GPU was found",
    "sycl": "For Intel GPUs with oneAPI; not chosen automatically",
    "openvino": "For Intel hardware with OpenVINO; not chosen automatically",
    "opencl": "For Adreno GPUs; not chosen automatically",
    "snapdragon": "For Snapdragon devices; not chosen automatically",
    "zdnn": "For IBM Z accelerators; not chosen automatically",
}


class RuntimeNotFoundError(Exception):
    """A version, build or installed runtime that doesn't exist."""


class RuntimeConflictError(Exception):
    """The request is valid but can't be done in the current state."""


class GitHubRateLimitError(Exception):
    """GitHub is refusing calls until the rate limit resets."""


class GitHubUnavailableError(Exception):
    """GitHub couldn't be reached or answered with an error."""


class RuntimeInstallError(Exception):
    """An install that can't proceed, with a reason a user can act on."""


class UnsafeArchiveError(RuntimeInstallError):
    """An archive tried to write outside its destination."""


@dataclass(frozen=True)
class ParsedBuild:
    id: str
    tag: str
    build: int
    os: str
    arch: str
    backend: str
    backend_version: str | None = None


def parse_build_name(name: str) -> ParsedBuild | None:
    """Reads a release asset or folder name. Parsed rather than matched against
    a list, because backend versions keep changing (`rocm-7.14` -> `rocm-10.0`)."""
    identifier = _ARCHIVE_SUFFIX.sub("", name)
    match = _BUILD_NAME.match(identifier)
    if match is None:
        return None
    tokens = match["rest"].split("-")
    arch = next((token for token in tokens if token in ARCHES), None)
    if arch is None:
        return None
    others = [token for token in tokens if token != arch]
    # `ubuntu-26.04-s390x`: a distro version in the place a backend would be.
    if others and re.fullmatch(r"\d+\.\d+", others[0]):
        others = others[1:]
    os_name = OS_BY_TOKEN[match["os"]]
    default_backend = "metal" if os_name == "macos" and arch == "arm64" else "cpu"
    return ParsedBuild(
        id=identifier,
        tag=match["tag"],
        build=int(match["build"]),
        os=os_name,
        arch=arch,
        backend=others[0] if others else default_backend,
        backend_version="-".join(others[1:]) or None,
    )


def _version_key(version: str | None) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", version or ""))


def _primary_gpu(hardware: HardwareInfo) -> GpuInfo | None:
    discrete = [gpu for gpu in hardware.gpus if gpu.kind == "discrete"]
    return max(discrete, key=lambda gpu: gpu.vram_total_bytes or 0, default=None)


@dataclass(frozen=True)
class BackendPreference:
    backend: str
    reason: str
    # Only builds at or below this version qualify (a driver's CUDA limit).
    max_version: str | None = None


def backend_preferences(hardware: HardwareInfo) -> list[BackendPreference]:
    """The order to try backends in on this machine, best first."""
    if any(gpu.kind == "unified" for gpu in hardware.gpus):
        return [BackendPreference("metal", "Uses the Apple GPU through Metal")]

    cpu_reason = "Runs on the CPU. Works everywhere, but far slower than a GPU"
    primary = _primary_gpu(hardware)
    if primary is None:
        integrated = next((gpu for gpu in hardware.gpus if gpu.kind == "integrated"), None)
        if integrated is None:
            return [BackendPreference("cpu", "No GPU was found, so this runs on the CPU")]
        preferences = [
            BackendPreference(
                "cpu", f"{integrated.name} is an integrated GPU, so the CPU is the dependable choice"
            )
        ]
        if integrated.apis.vulkan:
            preferences.append(
                BackendPreference(
                    "vulkan",
                    f"Uses {integrated.name} through Vulkan. It shares system memory, and may or may not "
                    "beat the CPU",
                )
            )
        return preferences

    preferences = []
    if primary.vendor == "nvidia" and primary.apis.cuda:
        preferences.append(
            BackendPreference(
                "cuda",
                f"Best for {primary.name}: your driver supports up to CUDA {primary.apis.cuda}",
                max_version=primary.apis.cuda,
            )
        )
    elif primary.vendor == "amd" and primary.apis.rocm:
        preferences.append(BackendPreference("rocm", f"Best for {primary.name}: the ROCm runtime was found"))
    if primary.apis.vulkan:
        preferences.append(BackendPreference("vulkan", f"Uses {primary.name} through Vulkan, with no extra setup"))
    preferences.append(BackendPreference("cpu", cpu_reason))
    return preferences


def rank_builds(
    builds: list[ParsedBuild], hardware: HardwareInfo
) -> tuple[list[tuple[ParsedBuild, str]], list[tuple[ParsedBuild, str]]]:
    """Splits builds for this OS and CPU into (the order `auto` tries them in,
    the rest), each with the reason to show for it."""
    ranked: list[tuple[ParsedBuild, str]] = []
    placed: set[str] = set()
    for preference in backend_preferences(hardware):
        matches = [build for build in builds if build.backend == preference.backend]
        if preference.max_version is not None:
            limit = _version_key(preference.max_version)
            matches = [build for build in matches if _version_key(build.backend_version) <= limit]
        for build in sorted(matches, key=lambda build: _version_key(build.backend_version), reverse=True):
            ranked.append((build, preference.reason))
            placed.add(build.id)

    others = [
        (build, _UNRANKED_REASONS.get(build.backend, "Not chosen automatically"))
        for build in builds
        if build.id not in placed
    ]
    others.sort(key=lambda entry: (entry[0].backend, entry[0].backend_version or ""))
    return ranked, others


def parse_list_devices(output: str) -> list[RuntimeDevice]:
    """Device lines from `llama-server --list-devices`. Anything else it prints
    (load logs, the `(none)` placeholder) doesn't match and is skipped."""
    devices: list[RuntimeDevice] = []
    for line in output.splitlines():
        match = _DEVICE_LINE.match(line)
        if match:
            devices.append(
                RuntimeDevice(
                    id=match["id"], name=match["name"], memory_total_bytes=int(match["total"]) * 1024**2
                )
            )
    return devices


def _normalized_gpu_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def match_gpu_device(devices: list[RuntimeDevice], gpus: list[GpuInfo]) -> str | None:
    """Which of a runtime's devices to run on. Matched by name because numbering
    differs per backend, and never by memory because a backend can report an
    integrated GPU as owning most of the system's RAM."""
    if not devices:
        return None
    by_kind: dict[str, set[str]] = {}
    for gpu in gpus:
        by_kind.setdefault(gpu.kind, set()).add(_normalized_gpu_name(gpu.name))

    def is_kind(device: RuntimeDevice, kind: str) -> bool:
        name = _normalized_gpu_name(device.name)
        return any(known and (known in name or name in known) for known in by_kind.get(kind, ()))

    for kind in ("discrete", "unified"):
        match = next((device for device in devices if is_kind(device, kind)), None)
        if match is not None:
            return match.id
    not_integrated = next((device for device in devices if not is_kind(device, "integrated")), None)
    return (not_integrated or devices[0]).id


def runtime_environment(runtime_dir: Path, backend: str) -> dict[str, str]:
    """The environment to run a build in. The ROCm SDK is only added for ROCm
    builds, so its DLLs can't be picked up by anything else."""
    environment = os.environ.copy()
    directories = [str(runtime_dir)]
    if backend == "rocm":
        directories.extend(rocm_sdk_directories())
    environment["PATH"] = os.pathsep.join([*directories, environment.get("PATH", "")])
    if sys.platform != "win32":
        environment["LD_LIBRARY_PATH"] = os.pathsep.join([str(runtime_dir), environment.get("LD_LIBRARY_PATH", "")])
    return environment


def server_binary_name() -> str:
    return "llama-server.exe" if sys.platform == "win32" else "llama-server"


def server_dir(runtime_dir: Path) -> Path | None:
    """The folder holding the executable. Installs put it at the top, but a Linux
    tarball extracted by hand still has its `llama-bNNNN/` wrapper folder."""
    name = server_binary_name()
    if (runtime_dir / name).is_file():
        return runtime_dir
    nested = next(runtime_dir.glob(f"*/{name}"), None)
    return nested.parent if nested is not None else None


def _tail(output: str, limit: int = 300) -> str:
    """The end of a failure message, which is where the cause is: a loader error
    starts with a long path and only names the missing library after it."""
    text = output.strip()
    return text if len(text) <= limit else "..." + text[-limit:]


@dataclass
class _Verification:
    verified: bool
    version: str | None = None
    commit: str | None = None
    devices: list[RuntimeDevice] = field(default_factory=list)
    gpu_device: str | None = None
    error: str | None = None


def _run_server_command(runtime_dir: Path, backend: str, flag: str) -> tuple[int | None, str]:
    binary = runtime_dir / server_binary_name()
    try:
        completed = subprocess.run(
            [str(binary), flag],
            capture_output=True,
            timeout=VERIFY_TIMEOUT_SECONDS,
            cwd=runtime_dir,
            env=runtime_environment(runtime_dir, backend),
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
    except subprocess.TimeoutExpired:
        return None, f"timed out after {VERIFY_TIMEOUT_SECONDS}s"
    except OSError as error:
        return None, str(error)
    output = completed.stdout + b"\n" + completed.stderr
    return completed.returncode, output.decode("utf-8", errors="replace")


def verify_runtime(runtime_dir: Path, backend: str, gpus: list[GpuInfo]) -> _Verification:
    """Runs the build. It has to start, and a GPU build has to find a GPU - a
    build that silently falls back to the CPU is the failure this exists for."""
    code, output = _run_server_command(runtime_dir, backend, "--version")
    if code != 0:
        return _Verification(False, error=f"llama-server wouldn't start (exit code {code}): {_tail(output)}")
    match = _VERSION_LINE.search(output)
    version = match["version"] if match else None
    commit = match["commit"] if match else None

    code, output = _run_server_command(runtime_dir, backend, "--list-devices")
    if code != 0:
        return _Verification(
            False,
            version=version,
            commit=commit,
            error=f"Listing devices failed (exit code {code}): {_tail(output)}",
        )
    devices = parse_list_devices(output)
    gpu_device = match_gpu_device(devices, gpus)
    if backend != "cpu" and not devices:
        hint = " The ROCm SDK may be missing." if backend == "rocm" else ""
        return _Verification(
            False, version=version, commit=commit, error=f"This {backend} build found no GPU on this machine.{hint}"
        )
    return _Verification(True, version=version, commit=commit, devices=devices, gpu_device=gpu_device)


def safe_extract(archive: Path, destination: Path, check: Callable[[], None] | None = None) -> None:
    """Extracts an archive, refusing any entry that would land outside
    `destination`. `check` runs between entries, so extraction can be cancelled."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            members = bundle.infolist()
            for member in members:
                if not (root / member.filename).resolve().is_relative_to(root):
                    raise UnsafeArchiveError(f"Archive entry escapes its folder: {member.filename}")
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise UnsafeArchiveError(f"Archive entry is a symbolic link: {member.filename}")
            for member in members:
                if check:
                    check()
                bundle.extract(member, root)
        return
    try:
        with tarfile.open(archive, "r:gz") as bundle:
            for member in bundle:
                if check:
                    check()
                bundle.extract(member, root, filter="data")
    except tarfile.FilterError as error:
        raise UnsafeArchiveError(f"Archive entry refused: {error}") from error
    except tarfile.ReadError as error:
        raise RuntimeInstallError(f"{archive.name} isn't a readable archive: {error}") from error


def _single_root(folder: Path) -> Path:
    """Descends through wrapper folders: Linux archives nest everything under
    `llama-b11960/`, while Windows archives have no such folder."""
    entries = list(folder.iterdir())
    while len(entries) == 1 and entries[0].is_dir():
        folder = entries[0]
        entries = list(folder.iterdir())
    return folder


def _digest_of(asset: dict[str, Any]) -> str | None:
    digest = asset.get("digest")
    if isinstance(digest, str) and digest.startswith("sha256:"):
        return digest.removeprefix("sha256:")
    return None


def _uploaded_assets(release: dict[str, Any]) -> list[dict[str, Any]]:
    # A release is published before CI has finished uploading to it.
    return [asset for asset in release.get("assets") or [] if asset.get("state", "uploaded") == "uploaded"]


@dataclass
class _ResolvedRelease:
    requested: str
    tag: str
    stable_version: str | None
    commit: str | None
    published_at: str | None
    assets: list[dict[str, Any]]

    def builds_for(self, os_name: str, arch: str) -> list[ParsedBuild]:
        builds = (parse_build_name(asset["name"]) for asset in self.assets)
        return [build for build in builds if build and build.os == os_name and build.arch == arch]

    def asset_for(self, build: ParsedBuild) -> dict[str, Any]:
        return next(asset for asset in self.assets if _ARCHIVE_SUFFIX.sub("", asset["name"]) == build.id)

    def companion_for(self, build: ParsedBuild) -> dict[str, Any] | None:
        """The CUDA DLLs, which ship separately from the build that needs them."""
        if build.backend != "cuda":
            return None
        for asset in self.assets:
            match = _COMPANION_NAME.match(asset["name"])
            if (
                match
                and OS_BY_TOKEN[match["os"]] == build.os
                and match["version"] == build.backend_version
                and match["arch"] == build.arch
            ):
                return asset
        return None


class LlamaRuntimeService:
    def __init__(
        self,
        hardware: HardwareService = hardware_service,
        client: httpx.Client | None = None,
        root: Path | None = None,
        jobs: JobStore | None = None,
    ) -> None:
        self.hardware = hardware
        self.client = client or httpx.Client(timeout=httpx.Timeout(15.0, connect=5.0))
        self._root = root
        self.jobs = jobs or JobStore()
        self._cache: dict[str, tuple[float, Any]] = {}
        self._cache_lock = threading.Lock()
        self._state_lock = threading.Lock()

    @property
    def root(self) -> Path:
        return self._root or llama_runtimes_dir()

    # GitHub

    def _github_json(self, path: str) -> Any:
        """A cached API response. If GitHub can't answer, a stale copy is better
        than nothing: unauthenticated calls are limited to 60 an hour."""
        with self._cache_lock:
            cached = self._cache.get(path)
        if cached is not None and time.monotonic() - cached[0] < RELEASE_CACHE_SECONDS:
            return cached[1]

        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        token = github_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            response = self.client.get(f"{GITHUB_API}{path}", headers=headers, follow_redirects=True)
        except httpx.HTTPError as error:
            if cached is not None:
                return cached[1]
            raise GitHubUnavailableError(f"Could not reach GitHub: {error}") from error

        if response.status_code == 404:
            raise RuntimeNotFoundError(f"GitHub has no llama.cpp release at {path}")
        if response.status_code in (403, 429) and response.headers.get("X-RateLimit-Remaining") == "0":
            if cached is not None:
                return cached[1]
            raise GitHubRateLimitError(self._rate_limit_message(response))
        if response.status_code >= 400:
            if cached is not None:
                return cached[1]
            raise GitHubUnavailableError(f"GitHub answered with HTTP {response.status_code}")

        try:
            data = response.json()
        except ValueError as error:
            if cached is not None:
                return cached[1]
            raise GitHubUnavailableError("GitHub sent a response that couldn't be read") from error
        with self._cache_lock:
            self._cache[path] = (time.monotonic(), data)
        return data

    @staticmethod
    def _rate_limit_message(response: httpx.Response) -> str:
        reset = response.headers.get("X-RateLimit-Reset")
        when = ""
        if reset and reset.isdigit():
            when = f" It resets at {datetime.fromtimestamp(int(reset), UTC):%H:%M} UTC."
        return f"GitHub's rate limit was reached.{when} Setting GITHUB_TOKEN raises the limit."

    def _stable_build_tag(self, release: dict[str, Any]) -> str:
        """The build a stable release corresponds to, from the file it ships
        for exactly this, or failing that from its release notes."""
        tag = release["tag_name"]
        asset = next((item for item in release.get("assets") or [] if item["name"] == STABLE_TAG_ASSET), None)
        if asset is not None:
            url = asset["browser_download_url"]
            with self._cache_lock:
                cached = self._cache.get(url)
            if cached is not None:
                return cached[1]
            try:
                response = self.client.get(url, follow_redirects=True)
                response.raise_for_status()
            except httpx.HTTPError as error:
                raise GitHubUnavailableError(f"Could not read {STABLE_TAG_ASSET} for {tag}: {error}") from error
            content = response.text.strip()
            if re.fullmatch(r"b\d+", content):
                with self._cache_lock:
                    # A published release's build doesn't change, so no expiry.
                    self._cache[url] = (float("inf"), content)
                return content
        mentioned = _STABLE_BUILD_TAG.search(release.get("body") or "")
        if mentioned:
            return mentioned.group(0)
        raise RuntimeNotFoundError(f"Could not work out which build stable release {tag} corresponds to")

    def _resolve_stable_release(self, release: dict[str, Any], requested: str) -> _ResolvedRelease:
        stable_version = release["tag_name"]
        build_tag = self._stable_build_tag(release)
        build = self._github_json(f"/releases/tags/{build_tag}")
        return self._to_resolved(build, requested, stable_version)

    @staticmethod
    def _to_resolved(release: dict[str, Any], requested: str, stable_version: str | None) -> _ResolvedRelease:
        return _ResolvedRelease(
            requested=requested,
            tag=release["tag_name"],
            stable_version=stable_version,
            commit=release.get("target_commitish"),
            published_at=release.get("published_at"),
            assets=_uploaded_assets(release),
        )

    def _resolve_version(self, version: str, hardware: HardwareInfo, wanted_backend: str) -> _ResolvedRelease:
        """`stable`, `latest`, a stable version (`v0.8.0`) or a build (`b11960`).
        `latest` is the newest build carrying `wanted_backend` for this machine:
        the very newest release may not have finished uploading."""
        if version == "stable":
            release = self._github_json("/releases/latest")
            if release["tag_name"].startswith("v"):
                return self._resolve_stable_release(release, version)
            return self._to_resolved(release, version, None)

        if version == "latest":
            releases = self._github_json(f"/releases?per_page={RECENT_RELEASES}")
            nightlies = [
                self._to_resolved(release, version, None)
                for release in releases
                if release["tag_name"].startswith("b") and not release.get("draft")
            ]
            if not nightlies:
                raise RuntimeNotFoundError("GitHub listed no recent llama.cpp builds")
            for candidate in nightlies:
                builds = candidate.builds_for(hardware.os.system, hardware.os.arch)
                if any(build.backend == wanted_backend for build in builds):
                    return candidate
            return nightlies[0]

        try:
            release = self._github_json(f"/releases/tags/{version}")
        except RuntimeNotFoundError as error:
            raise RuntimeNotFoundError(f"llama.cpp has no release called {version}") from error
        if version.startswith("v"):
            return self._resolve_stable_release(release, version)
        return self._to_resolved(release, version, None)

    # What's on offer

    def available(self, version: str = "stable") -> AvailableBuilds:
        hardware = self.hardware.get_hardware()
        preferred = backend_preferences(hardware)[0].backend
        release = self._resolve_version(version, hardware, preferred)
        ranked, others = rank_builds(release.builds_for(hardware.os.system, hardware.os.arch), hardware)

        builds = [
            self._available_build(release, build, reason, rank)
            for rank, (build, reason) in enumerate(ranked, start=1)
        ]
        builds.extend(self._available_build(release, build, reason, None) for build, reason in others)
        if builds and builds[0].rank is not None:
            builds[0].recommended = True
        return AvailableBuilds(
            requested=version,
            tag=release.tag,
            stable_version=release.stable_version,
            published_at=release.published_at,
            os=hardware.os.system,
            arch=hardware.os.arch,
            builds=builds,
        )

    def _available_build(
        self, release: _ResolvedRelease, build: ParsedBuild, reason: str, rank: int | None
    ) -> AvailableBuild:
        companion = release.companion_for(build)
        return AvailableBuild(
            id=build.id,
            tag=build.tag,
            backend=build.backend,
            backend_version=build.backend_version,
            arch=build.arch,
            size_bytes=release.asset_for(build)["size"] + (companion["size"] if companion else 0),
            companion_asset=companion["name"] if companion else None,
            rank=rank,
            reason=reason,
            installed=server_dir(self.root / build.id) is not None,
        )

    # Installing

    def start_install(self, version: str = "stable", backend: str = "auto", activate: bool = True) -> JobInfo:
        """Looks everything up first, so a bad version or backend is an immediate
        error rather than a failed job. Then the work runs in the background."""
        hardware = self.hardware.get_hardware()
        is_auto = backend == "auto"
        wanted = backend_preferences(hardware)[0].backend if is_auto else backend.split("-", 1)[0]
        release = self._resolve_version(version, hardware, wanted)
        builds = release.builds_for(hardware.os.system, hardware.os.arch)
        if is_auto:
            candidates = [build for build, _ in rank_builds(builds, hardware)[0]]
        else:
            candidates = self._explicit_candidates(builds, backend, release.tag, hardware)
        if not candidates:
            raise RuntimeNotFoundError(
                f"{release.tag} has no build for {hardware.os.system}-{hardware.os.arch}"
            )

        def work(context: JobContext) -> dict[str, Any]:
            return self._install(context, release, candidates, not is_auto, activate, hardware)

        return self.jobs.submit("llama_runtime_install", candidates[0].id, work)

    @staticmethod
    def _explicit_candidates(
        builds: list[ParsedBuild], backend: str, tag: str, hardware: HardwareInfo
    ) -> list[ParsedBuild]:
        """A named backend means that one build and no fallback: `cuda-12.4`
        exactly, or plain `cuda` meaning the best of them for this machine."""
        matching = [
            build
            for build in builds
            if backend in (build.backend, f"{build.backend}-{build.backend_version}")
        ]
        if not matching:
            on_offer = sorted(
                {f"{build.backend}-{build.backend_version}" if build.backend_version else build.backend for build in builds}
            )
            raise RuntimeNotFoundError(
                f"{tag} has no '{backend}' build for {hardware.os.system}-{hardware.os.arch}. "
                f"Available: {', '.join(on_offer) or 'none'}"
            )
        ranked = [build for build, _ in rank_builds(matching, hardware)[0]]
        return ranked[:1] or sorted(matching, key=lambda build: _version_key(build.backend_version), reverse=True)[:1]

    def _install(
        self,
        context: JobContext,
        release: _ResolvedRelease,
        candidates: list[ParsedBuild],
        explicit: bool,
        activate: bool,
        hardware: HardwareInfo,
    ) -> dict[str, Any]:
        rejected: list[str] = []
        for build in candidates:
            context.check_cancelled()
            path = self.root / build.id
            already_installed = path.exists()
            if already_installed:
                manifest = self._existing_build(build, hardware)
            else:
                manifest = self._download_and_prepare(context, release, build, explicit, hardware)

            if manifest is None:  # tried and thrown away; the reason is in the notes
                rejected.append(build.id)
                continue
            if not manifest.get("verified"):
                reason = manifest.get("verification_error")
                if explicit:
                    context.note(f"{build.id} was installed but not activated: {reason}")
                    return self._install_result(build, manifest, activated=False, already_installed=already_installed)
                rejected.append(build.id)
                context.note(f"Skipped {build.id}: {reason}")
                continue

            activated = False
            if activate:
                self._set_active(build.id)
                activated = True
            return self._install_result(build, manifest, activated=activated, already_installed=already_installed)
        raise RuntimeInstallError(
            f"No build worked on this machine (tried {', '.join(rejected)}). See this job's notes for why."
        )

    @staticmethod
    def _install_result(
        build: ParsedBuild, manifest: dict[str, Any], activated: bool, already_installed: bool
    ) -> dict[str, Any]:
        return {
            "runtime_id": build.id,
            "tag": build.tag,
            "stable_version": manifest.get("stable_version"),
            "backend": build.backend,
            "verified": bool(manifest.get("verified")),
            "gpu_device": manifest.get("gpu_device"),
            "activated": activated,
            "already_installed": already_installed,
        }

    def _existing_build(self, build: ParsedBuild, hardware: HardwareInfo) -> dict[str, Any]:
        path = self.root / build.id
        if server_dir(path) is None:
            raise RuntimeInstallError(
                f"{path} already exists but has no {server_binary_name()}. Remove that folder and try again."
            )
        manifest = self._read_manifest(path)
        # A build that found no GPU earlier may do now (say, the SDK was
        # installed since), so a repeat request gets a fresh answer.
        if manifest is None or not manifest.get("verified"):
            manifest = self._verify_and_record(path, build, hardware.gpus, manifest)
        return manifest

    def _download_and_prepare(
        self,
        context: JobContext,
        release: _ResolvedRelease,
        build: ParsedBuild,
        explicit: bool,
        hardware: HardwareInfo,
    ) -> dict[str, Any] | None:
        """Downloads, extracts and tests a build away from the runtimes list, and
        only moves it into place if it's being kept. Returns None if it was
        tried and thrown away, with the reason in the job's notes."""
        asset = release.asset_for(build)
        companion = release.companion_for(build)
        if build.backend == "cuda" and companion is None:
            context.note(f"{release.tag} has no CUDA DLLs for {build.id}; it may not start")
        assets = [asset, companion] if companion else [asset]
        total = sum(item["size"] for item in assets)

        self.root.mkdir(parents=True, exist_ok=True)
        needed = total * ARCHIVE_SPACE_FACTOR
        free = shutil.disk_usage(self.root).free
        if free < needed:
            raise RuntimeInstallError(
                f"Not enough free space on the drive holding {self.root}: "
                f"{build.id} needs about {needed / 1e9:.1f} GB and {free / 1e9:.1f} GB is free"
            )

        downloads = self.root / DOWNLOADS_DIR
        archives: list[Path] = []
        context.update(stage="downloading")
        context.begin_transfer(total)
        transferred = 0
        for item in assets:
            digest = _digest_of(item)
            if digest is None:
                context.note(f"GitHub published no checksum for {item['name']}, so it wasn't verified")
            # Prefixed with the tag: the Windows CUDA DLLs have the same asset
            # name in every release, and a partial must not resume across them.
            archive = downloads / f"{release.tag}-{item['name']}.partial"
            download_file(
                self.client,
                item["browser_download_url"],
                archive,
                context,
                sha256=digest,
                base_bytes=transferred,
            )
            context.update(stage="downloading")
            transferred += item["size"]
            archives.append(archive)

        staging = self.root / f".tmp-{build.id}"
        if staging.exists():
            shutil.rmtree(staging)
        try:
            context.update(stage="extracting")
            content = staging / "build"
            safe_extract(archives[0], content, context.check_cancelled)
            content = _single_root(content)
            if not (content / server_binary_name()).is_file():
                raise RuntimeInstallError(f"{asset['name']} has no {server_binary_name()} in it")
            for extra in archives[1:]:
                extra_dir = staging / "extra"
                safe_extract(extra, extra_dir, context.check_cancelled)
                shutil.copytree(_single_root(extra_dir), content, dirs_exist_ok=True)
                shutil.rmtree(extra_dir)

            context.update(stage="testing")
            manifest = self._verify_and_record(content, build, hardware.gpus, None, release, _digest_of(asset))
            for archive in archives:
                archive.unlink(missing_ok=True)
            if not manifest.get("verified") and not explicit:
                context.note(f"{build.id} failed its test: {manifest.get('verification_error')}")
                return None

            context.update(stage="finalizing")
            os.replace(content, self.root / build.id)
            return manifest
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def _verify_and_record(
        self,
        path: Path,
        build: ParsedBuild,
        gpus: list[GpuInfo],
        existing: dict[str, Any] | None,
        release: _ResolvedRelease | None = None,
        digest: str | None = None,
    ) -> dict[str, Any]:
        verification = verify_runtime(server_dir(path) or path, build.backend, gpus)
        now = datetime.now(UTC).isoformat()
        manifest = dict(existing or {})
        manifest.update(
            id=build.id,
            tag=build.tag,
            verified=verification.verified,
            verification_error=verification.error,
            version=verification.version,
            commit=verification.commit,
            devices=[device.model_dump() for device in verification.devices],
            gpu_device=verification.gpu_device,
            verified_at=now,
        )
        if release is not None:
            manifest.update(stable_version=release.stable_version, digest=digest, installed_at=now)
        try:
            (path / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        except OSError as error:
            logger.warning("Could not save %s in %s: %s", MANIFEST_NAME, path, error)
        return manifest

    @staticmethod
    def _read_manifest(path: Path) -> dict[str, Any] | None:
        try:
            data = json.loads((path / MANIFEST_NAME).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    # What's installed

    def list_installed(self, refresh: bool = False) -> InstalledRuntimes:
        hardware = self.hardware.get_hardware()
        host = (hardware.os.system, hardware.os.arch)
        builds = self._installed_builds()
        active = self._read_active()

        manifests = {build.id: self._read_manifest(self.root / build.id) for build in builds}
        to_verify = [
            build
            for build in builds
            if (build.os, build.arch) == host
            and (refresh or manifests[build.id] is None)
            and server_dir(self.root / build.id) is not None
        ]
        if to_verify:
            with ThreadPoolExecutor(max_workers=VERIFY_WORKERS) as pool:
                results = pool.map(
                    lambda build: self._verify_and_record(
                        self.root / build.id, build, hardware.gpus, manifests[build.id]
                    ),
                    to_verify,
                )
                for build, manifest in zip(to_verify, results):
                    manifests[build.id] = manifest

        runtimes = [
            self._installed_runtime(build, manifests[build.id], host, build.id == active) for build in builds
        ]
        return InstalledRuntimes(
            runtimes_dir=str(self.root),
            active_id=active if any(runtime.id == active for runtime in runtimes) else None,
            runtimes=runtimes,
            update_available=self._update_status(next((build for build in builds if build.id == active), None), hardware),
        )

    def _installed_builds(self) -> list[ParsedBuild]:
        if not self.root.is_dir():
            return []
        builds = [parse_build_name(entry.name) for entry in self.root.iterdir() if entry.is_dir()]
        return sorted(
            (build for build in builds if build is not None),
            key=lambda build: (-build.build, build.id),
        )

    def _installed_runtime(
        self, build: ParsedBuild, manifest: dict[str, Any] | None, host: tuple[str, str], active: bool
    ) -> InstalledRuntime:
        manifest = manifest or {}
        path = self.root / build.id
        directory = server_dir(path)
        return InstalledRuntime(
            id=build.id,
            tag=build.tag,
            build=build.build,
            stable_version=manifest.get("stable_version"),
            os=build.os,
            arch=build.arch,
            backend=build.backend,
            backend_version=build.backend_version,
            path=str(path),
            server_path=str(directory / server_binary_name()) if directory else None,
            compatible=(build.os, build.arch) == host,
            active=active,
            verified=manifest.get("verified"),
            verification_error=manifest.get("verification_error"),
            version=manifest.get("version"),
            commit=manifest.get("commit"),
            devices=[RuntimeDevice.model_validate(device) for device in manifest.get("devices") or []],
            gpu_device=manifest.get("gpu_device"),
            installed_at=manifest.get("installed_at"),
            verified_at=manifest.get("verified_at"),
        )

    def _update_status(self, active: ParsedBuild | None, hardware: HardwareInfo) -> RuntimeUpdates | None:
        """Newer builds of the backend in use. `latest` skips one still uploading,
        so this never offers an update that can't be installed yet."""

        def is_newer(tag: str) -> bool | None:
            if active is None or not re.fullmatch(r"b\d+", tag):
                return None
            return int(tag[1:]) > active.build

        wanted = active.backend if active else backend_preferences(hardware)[0].backend
        try:
            stable = self._resolve_version("stable", hardware, wanted)
            latest = self._resolve_version("latest", hardware, wanted)
        except (GitHubRateLimitError, GitHubUnavailableError, RuntimeNotFoundError) as error:
            logger.info("Could not check for llama.cpp updates: %s", error)
            return None
        return RuntimeUpdates(
            stable_tag=stable.tag,
            stable_version=stable.stable_version,
            stable_is_newer=is_newer(stable.tag),
            latest_tag=latest.tag,
            latest_is_newer=is_newer(latest.tag),
        )

    def _installed_path(self, runtime_id: str) -> tuple[Path, ParsedBuild]:
        """Resolves an id from a request to a runtime folder. Checked strictly
        because deleting goes through here: it must only ever reach a real
        runtime folder directly inside the runtimes directory."""
        build = parse_build_name(runtime_id)
        if build is None or build.id != runtime_id or not _SAFE_RUNTIME_ID.match(runtime_id) or ".." in runtime_id:
            raise RuntimeNotFoundError(f"No installed runtime called {runtime_id}")
        path = self.root / runtime_id
        if path.is_symlink() or not path.is_dir() or path.resolve().parent != self.root.resolve():
            raise RuntimeNotFoundError(f"No installed runtime called {runtime_id}")
        return path, build

    def set_active(self, runtime_id: str) -> InstalledRuntimes:
        path, build = self._installed_path(runtime_id)
        hardware = self.hardware.get_hardware()
        if (build.os, build.arch) != (hardware.os.system, hardware.os.arch):
            raise RuntimeConflictError(
                f"{runtime_id} is for {build.os}-{build.arch}, and this machine is "
                f"{hardware.os.system}-{hardware.os.arch}"
            )
        try:
            manifest = self._existing_build(build, hardware)
        except RuntimeInstallError as error:
            raise RuntimeConflictError(str(error)) from error
        if not manifest.get("verified"):
            raise RuntimeConflictError(
                f"{runtime_id} didn't pass its test on this machine: {manifest.get('verification_error')}"
            )
        self._set_active(runtime_id)
        return self.list_installed()

    def delete(self, runtime_id: str) -> None:
        path, _ = self._installed_path(runtime_id)
        if self._read_active() == runtime_id:
            raise RuntimeConflictError(f"{runtime_id} is the active runtime. Switch to another one first.")
        if self.jobs.is_active(runtime_id):
            raise RuntimeConflictError(f"{runtime_id} is being installed right now")
        shutil.rmtree(path)

    def _read_active(self) -> str | None:
        try:
            data = json.loads((self.root / STATE_NAME).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        active = data.get("active") if isinstance(data, dict) else None
        return active if isinstance(active, str) else None

    def _set_active(self, runtime_id: str) -> None:
        with self._state_lock:
            self.root.mkdir(parents=True, exist_ok=True)
            temporary = self.root / f"{STATE_NAME}.tmp"
            temporary.write_text(json.dumps({"active": runtime_id}, indent=2), encoding="utf-8")
            os.replace(temporary, self.root / STATE_NAME)


llama_runtime_service = LlamaRuntimeService()

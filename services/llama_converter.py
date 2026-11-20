"""llama.cpp's converter from Hugging Face safetensors to GGUF, set up for each
build so a model is converted by the same llama.cpp that will run it.

- The converter is Python from llama.cpp's source, not its release builds:
  `convert_hf_to_gguf.py`, the `conversion` package and `gguf-py`. Only those
  parts of the build's source archive are kept, a few MB of its 40.
- It needs PyTorch and transformers at versions llama.cpp pins, which mustn't
  clash with the app's own. So it runs in a Python environment of its own. One
  comes to about a gigabyte, so builds whose converters ask for exactly the same
  packages share it.
- Once installed it's asked what it can convert. That's also the proof that
  its environment works.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import zlib
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import httpx

from models.jobs import JobInfo
from models.runtime import LlamaConverter, LlamaConverters
from services.jobs import ACTIVE_STATUSES, JobContext, JobStore, download_file
from services.llama_runtime import (
    LlamaRuntimeService,
    RuntimeConflictError,
    RuntimeInstallError,
    RuntimeNotFoundError,
    UnsafeArchiveError,
    llama_runtime_service,
    parse_build_name,
)
from services.processes import tie_to_app
from services.settings import llama_converter_dir

logger = logging.getLogger(__name__)

SOURCE_URL = "https://github.com/ggml-org/llama.cpp/archive/refs/tags/{tag}.tar.gz"
JOB_KIND = "llama_converter_install"
MANIFEST_NAME = "localai-converter.json"
ENVIRONMENT_MANIFEST_NAME = "localai-environment.json"
SOURCES_DIR = "sources"
ENVIRONMENTS_DIR = "environments"
DOWNLOADS_DIR = ".downloads"
SCRIPT_NAME = "convert_hf_to_gguf.py"
REQUIREMENTS_FILE = "requirements/requirements-convert_hf_to_gguf.txt"
# What the converter reads, out of the source's several thousand files. Some
# models' chat templates are in `models/templates`.
SOURCE_FOLDERS = ("conversion/", "gguf-py/", "requirements/", "models/templates/")
# PyTorch and the rest come to about a gigabyte installed, and pip needs room
# to unpack them first.
ENVIRONMENT_SPACE_BYTES = 3_000_000_000
SOURCE_SPACE_BYTES = 200_000_000
VENV_TIMEOUT_SECONDS = 5 * 60
# PyTorch alone is a couple of hundred MB to download.
PIP_TIMEOUT_SECONDS = 30 * 60
# Importing PyTorch the first time can take a while on a slow disk.
CHECK_TIMEOUT_SECONDS = 5 * 60
OUTPUT_TAIL_LINES = 20

_SAFE_TAG = re.compile(r"^b\d+$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
# `TEXT models:` or `ERROR:hf-to-gguf:MMPROJ models:`, then `  - LlamaForCausalLM`.
_MODELS_SECTION = re.compile(r"\b(?P<kind>[A-Z]+) models:\s*$")
_MODELS_ENTRY = re.compile(r"(?:^|:)\s+-\s+(?P<name>[A-Za-z0-9_.]+)\s*$")
_PIP_COLLECTING = re.compile(r"^\s*Collecting\s+(?P<requirement>\S+)")
_PIP_DOWNLOADING = re.compile(r"^\s*Downloading\s+(?P<file>\S+)")
_PIP_INSTALLING = re.compile(r"^\s*Installing collected packages:")
# The converter has no use for the app's secrets, and the app's own Python
# settings would point it at the app's packages instead of its own.
_WITHHELD_VARIABLES = {
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "GITHUB_TOKEN",
    "LOCALAI_API_KEY",
    "PYTHONPATH",
    "PYTHONHOME",
    "VIRTUAL_ENV",
}


class ConverterSpaceError(RuntimeInstallError):
    """The drive holding the converters is too full to set one up."""


def run_command(
    command: list[str],
    context: JobContext,
    *,
    timeout: float,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    on_line: Callable[[str], None] | None = None,
) -> tuple[int, list[str]]:
    """Runs a command to the end, giving each line it prints to `on_line`, and
    returns its exit code and last few lines. A cancel request ends it, and so
    does running past `timeout`, which raises."""
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    tie_to_app(process)
    lines: queue.Queue[str | None] = queue.Queue()

    def read() -> None:
        assert process.stdout is not None
        for raw in process.stdout:
            lines.put(raw.decode("utf-8", "replace").rstrip())
        lines.put(None)

    threading.Thread(target=read, name="converter-output", daemon=True).start()
    tail: deque[str] = deque(maxlen=OUTPUT_TAIL_LINES)
    deadline = time.monotonic() + timeout
    try:
        while True:
            context.check_cancelled()
            if time.monotonic() > deadline:
                raise RuntimeInstallError(f"{Path(command[0]).name} didn't finish within {timeout / 60:.0f} minutes")
            try:
                line = lines.get(timeout=0.2)
            except queue.Empty:
                continue
            if line is None:
                break
            tail.append(line)
            if on_line is not None:
                on_line(line)
        return process.wait(), list(tail)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def converter_environment() -> dict[str, str]:
    """What the converter and pip run with. Offline for Hugging Face, so a
    model's tokenizer can't send transformers off to fetch code to run."""
    env = {name: value for name, value in os.environ.items() if name.upper() not in _WITHHELD_VARIABLES}
    env.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return env


def environment_python(environment: Path) -> Path:
    return environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def extract_converter(archive: Path, destination: Path, check: Callable[[], None] | None = None) -> str | None:
    """Extracts the parts of llama.cpp's source the converter needs, from the
    archive's `llama.cpp-<tag>/` folder. Returns the commit GitHub made the
    archive from, which it records in the archive's header."""
    destination.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive, "r:gz") as bundle:
            for member in bundle:
                if check:
                    check()
                _, separator, inner = member.name.partition("/")
                if not separator or not (inner == SCRIPT_NAME or inner.startswith(SOURCE_FOLDERS)):
                    continue
                member.name = inner
                bundle.extract(member, destination, filter="data")
            commit = bundle.pax_headers.get("comment")
    except tarfile.FilterError as error:
        raise UnsafeArchiveError(f"Archive entry refused: {error}") from error
    except (tarfile.TarError, EOFError, zlib.error) as error:
        raise RuntimeInstallError(f"{archive.name} isn't a readable archive: {error}") from error
    for needed in (SCRIPT_NAME, REQUIREMENTS_FILE):
        if not (destination / needed).is_file():
            raise RuntimeInstallError(f"llama.cpp's source has no {needed} at this build")
    return commit if isinstance(commit, str) and _COMMIT.match(commit) else None


def requirement_files(source: Path) -> list[Path]:
    """The converter's requirements file and the ones it includes with `-r`."""
    root = source.resolve()
    found: list[Path] = []
    pending = [root / REQUIREMENTS_FILE]
    while pending:
        path = pending.pop()
        if path in found:
            continue
        found.append(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] in ("-r", "--requirement"):
                included = (path.parent / parts[1]).resolve()
                if not included.is_relative_to(root):
                    raise RuntimeInstallError(f"{path.name} includes a file outside llama.cpp's source: {parts[1]}")
                pending.append(included)
    return found


def environment_key(source: Path, python_version: str) -> str:
    """Names an environment by what's installed in it: the Python version and
    the requirements files. Builds with the same key can share one."""
    digest = hashlib.sha256(python_version.encode())
    for path in sorted(requirement_files(source), key=lambda path: path.name):
        digest.update(path.name.encode() + b"\0" + path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()[:16]


def parse_supported_models(output: str) -> dict[str, list[str]]:
    """`--print-supported-models` as {"TEXT": [...], "MMPROJ": [...]}."""
    sections: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in output.splitlines():
        if match := _MODELS_SECTION.search(line):
            current = sections.setdefault(match["kind"], [])
        elif current is not None and (match := _MODELS_ENTRY.search(line)):
            current.append(match["name"])
    return sections


def pip_current_file(line: str) -> tuple[bool, str | None]:
    """Whether a line of pip's output changes what it's working on, and to
    what: a requirement it's looking up, a package it's downloading, or nothing
    in particular once it's installing them all. Its metadata downloads are
    too small to show."""
    if match := _PIP_DOWNLOADING.match(line):
        name = unquote(match["file"].rsplit("/", 1)[-1])
        return (False, None) if name.endswith(".metadata") else (True, name)
    if match := _PIP_COLLECTING.match(line):
        return True, match["requirement"]
    if _PIP_INSTALLING.match(line):
        return True, None
    return False, None


def _python_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _last_lines(lines: list[str], count: int = 5) -> str:
    useful = [line.strip() for line in lines if line.strip()]
    return " / ".join(useful[-count:]) or "it printed nothing"


class LlamaConverterService:
    def __init__(
        self,
        runtimes: LlamaRuntimeService = llama_runtime_service,
        client: httpx.Client | None = None,
        root: Path | None = None,
        jobs: JobStore | None = None,
    ) -> None:
        self.runtimes = runtimes
        self.client = client or runtimes.client
        self._root = root
        # The runtimes' own, so a converter install is listed and followed with them.
        self.jobs = jobs or runtimes.jobs
        self._lock = threading.Lock()
        self._environment_locks: dict[str, threading.Lock] = {}

    @property
    def root(self) -> Path:
        return self._root or llama_converter_dir()

    def _source_dir(self, tag: str) -> Path:
        return self.root / SOURCES_DIR / tag

    def _environment_dir(self, key: str) -> Path:
        return self.root / ENVIRONMENTS_DIR / key

    def _environment_ready(self, key: object) -> bool:
        if not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{16}", key):
            return False
        environment = self._environment_dir(key)
        return (
            _read_json(environment / ENVIRONMENT_MANIFEST_NAME) is not None
            and environment_python(environment).is_file()
        )

    # What's installed

    def status(self) -> LlamaConverters:
        active = self.runtimes.active_id()
        build = parse_build_name(active) if active else None
        active_tag = build.tag if build else None
        converters = self._installed()
        return LlamaConverters(
            converter_dir=str(self.root),
            active_tag=active_tag,
            ready=any(converter.tag == active_tag and converter.ready for converter in converters),
            converters=converters,
        )

    def _installed(self) -> list[LlamaConverter]:
        sources = self.root / SOURCES_DIR
        if not sources.is_dir():
            return []
        converters = []
        for folder in sources.iterdir():
            manifest = _read_json(folder / MANIFEST_NAME) if _SAFE_TAG.match(folder.name) else None
            if manifest is None:
                continue
            converters.append(
                LlamaConverter(
                    tag=folder.name,
                    commit=manifest.get("commit"),
                    path=str(folder),
                    environment=str(manifest.get("environment")),
                    ready=self._environment_ready(manifest.get("environment")),
                    text_architectures=manifest.get("text_architectures") or [],
                    vision_architectures=manifest.get("vision_architectures") or [],
                    installed_at=manifest.get("installed_at"),
                )
            )
        return sorted(converters, key=lambda converter: -int(converter.tag[1:]))

    # Installing

    def start_install(self, runtime_id: str | None = None) -> JobInfo:
        """Looks the build up first, so a missing or unusable one is an
        immediate error rather than a failed job."""
        runtime = self.runtimes.launchable(runtime_id)
        build = parse_build_name(runtime.id)
        if build is None:
            raise RuntimeNotFoundError(f"No installed runtime called {runtime.id}")
        if not self._converter_ready(build.tag):
            has_environment = any(self._environment_ready(converter.environment) for converter in self._installed())
            self._check_space(SOURCE_SPACE_BYTES + (0 if has_environment else ENVIRONMENT_SPACE_BYTES))

        def work(context: JobContext) -> dict[str, Any]:
            return self._install(context, build.tag)

        details = {"tag": build.tag, "runtime_id": runtime.id}
        return self.jobs.submit(JOB_KIND, f"converter:{build.tag}", work, details=details)

    def _converter_ready(self, tag: str) -> bool:
        manifest = _read_json(self._source_dir(tag) / MANIFEST_NAME)
        return manifest is not None and self._environment_ready(manifest.get("environment"))

    def _check_space(self, needed: int) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(self.root).free
        if free < needed:
            raise ConverterSpaceError(
                f"Not enough free space on the drive holding {self.root}: the converter needs about "
                f"{needed / 1e9:.1f} GB and {free / 1e9:.1f} GB is free"
            )

    def _install(self, context: JobContext, tag: str) -> dict[str, Any]:
        source = self._source_dir(tag)
        manifest = _read_json(source / MANIFEST_NAME)
        if manifest is not None and self._environment_ready(manifest.get("environment")):
            return self._result(manifest, already_installed=True, new_environment=False)

        staging = self.root / SOURCES_DIR / f".tmp-{tag}"
        shutil.rmtree(staging, ignore_errors=True)
        try:
            commit = self._download_source(context, tag, staging)
            key = environment_key(staging, _python_version())
            with self._environment_lock(key):
                new_environment = not self._environment_ready(key)
                if new_environment:
                    self._create_environment(context, staging, key)
                else:
                    context.note(f"Using the Python environment {key}, which another build's converter set up.")
                try:
                    architectures = self._supported_models(context, staging, key)
                except Exception:
                    if new_environment:
                        shutil.rmtree(self._environment_dir(key), ignore_errors=True)
                    raise
            manifest = {
                "tag": tag,
                "commit": commit,
                "environment": key,
                "python": _python_version(),
                "text_architectures": architectures.get("TEXT", []),
                "vision_architectures": architectures.get("MMPROJ", []),
                "installed_at": datetime.now(UTC).isoformat(),
            }
            _write_json(staging / MANIFEST_NAME, manifest)
            context.update(stage="finalizing", current_file=None)
            with self._lock:
                # One without a manifest is what an interrupted install leaves.
                shutil.rmtree(source, ignore_errors=True)
                os.replace(staging, source)
            return self._result(manifest, already_installed=False, new_environment=new_environment)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def _environment_lock(self, key: str) -> threading.Lock:
        with self._lock:
            return self._environment_locks.setdefault(key, threading.Lock())

    @staticmethod
    def _result(manifest: dict[str, Any], already_installed: bool, new_environment: bool) -> dict[str, Any]:
        return {
            "tag": manifest.get("tag"),
            "commit": manifest.get("commit"),
            "environment": manifest.get("environment"),
            "new_environment": new_environment,
            "text_architectures": len(manifest.get("text_architectures") or []),
            "vision_architectures": len(manifest.get("vision_architectures") or []),
            "already_installed": already_installed,
        }

    def _download_source(self, context: JobContext, tag: str, destination: Path) -> str | None:
        downloads = self.root / DOWNLOADS_DIR
        downloads.mkdir(parents=True, exist_ok=True)
        archive = downloads / f"llama.cpp-{tag}.tar.gz"
        # GitHub makes these archives on the fly, so a partial one can't be
        # trusted to match what a resumed download would send.
        archive.unlink(missing_ok=True)
        context.update(stage="downloading", current_file=archive.name)
        context.begin_transfer(None)
        try:
            try:
                download_file(self.client, SOURCE_URL.format(tag=tag), archive, context)
            except httpx.HTTPStatusError as error:
                status = error.response.status_code
                if status == 404:
                    raise RuntimeNotFoundError(f"GitHub has no llama.cpp source for {tag}") from error
                raise RuntimeInstallError(f"GitHub answered {status} for llama.cpp's source at {tag}") from error
            except httpx.TransportError as error:
                raise RuntimeInstallError(f"Could not download llama.cpp's source for {tag}: {error}") from error
            context.update(stage="extracting", current_file=None)
            return extract_converter(archive, destination, context.check_cancelled)
        finally:
            archive.unlink(missing_ok=True)

    def _create_environment(self, context: JobContext, source: Path, key: str) -> None:
        environment = self._environment_dir(key)
        # One without a manifest is what an interrupted install leaves.
        shutil.rmtree(environment, ignore_errors=True)
        self._check_space(ENVIRONMENT_SPACE_BYTES)
        try:
            context.update(stage="creating_environment", current_file=None)
            code, output = run_command(
                [sys.executable, "-m", "venv", str(environment)],
                context,
                timeout=VENV_TIMEOUT_SECONDS,
                env=converter_environment(),
            )
            if code != 0 or not environment_python(environment).is_file():
                raise RuntimeInstallError(
                    f"Couldn't create a Python environment for the converter: {_last_lines(output)}"
                )

            context.update(stage="installing_packages")

            def show_package(line: str) -> None:
                changed, current = pip_current_file(line)
                if changed:
                    context.update(current_file=current)

            code, output = run_command(
                [
                    str(environment_python(environment)),
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "--no-input",
                    "--no-cache-dir",
                    "--progress-bar",
                    "off",
                    "-r",
                    str(source / REQUIREMENTS_FILE),
                ],
                context,
                timeout=PIP_TIMEOUT_SECONDS,
                cwd=(source / REQUIREMENTS_FILE).parent,
                env=converter_environment(),
                on_line=show_package,
            )
            if code != 0:
                raise RuntimeInstallError(f"Installing the converter's packages failed: {_last_lines(output)}")
            _write_json(
                environment / ENVIRONMENT_MANIFEST_NAME,
                {
                    "key": key,
                    "python": _python_version(),
                    "requirements": [path.name for path in requirement_files(source)],
                    "installed_at": datetime.now(UTC).isoformat(),
                },
            )
        except BaseException:
            shutil.rmtree(environment, ignore_errors=True)
            raise
        finally:
            context.update(current_file=None)

    def _supported_models(self, context: JobContext, source: Path, key: str) -> dict[str, list[str]]:
        context.update(stage="testing", current_file=None)
        lines: list[str] = []
        code, output = run_command(
            [
                str(environment_python(self._environment_dir(key))),
                str(source / SCRIPT_NAME),
                "--print-supported-models",
            ],
            context,
            timeout=CHECK_TIMEOUT_SECONDS,
            cwd=source,
            env=converter_environment(),
            on_line=lines.append,
        )
        supported = parse_supported_models("\n".join(lines))
        if code != 0 or not supported.get("TEXT"):
            raise RuntimeInstallError(f"The converter didn't run in its environment: {_last_lines(output)}")
        return supported

    # Removing

    def delete(self, tag: str) -> None:
        """Removes a build's converter, and its Python environment unless
        another build's converter uses it too."""
        source = self._source_dir(tag)
        if not _SAFE_TAG.match(tag) or source.is_symlink() or not source.is_dir():
            raise RuntimeNotFoundError(f"No converter is installed for {tag}")
        if self.jobs.is_active(f"converter:{tag}"):
            raise RuntimeConflictError(f"The converter for {tag} is being installed right now")
        shutil.rmtree(source)
        self._remove_unused_environments()

    def _remove_unused_environments(self) -> None:
        environments = self.root / ENVIRONMENTS_DIR
        if not environments.is_dir():
            return
        # An install in progress may be setting one up that nothing uses yet.
        if any(job.kind == JOB_KIND and job.status in ACTIVE_STATUSES for job in self.jobs.all_jobs()):
            return
        used = {converter.environment for converter in self._installed()}
        for folder in environments.iterdir():
            if folder.is_dir() and not folder.is_symlink() and folder.name not in used:
                shutil.rmtree(folder, ignore_errors=True)


llama_converter_service = LlamaConverterService()

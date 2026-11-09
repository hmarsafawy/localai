"""Runs downloaded models with llama-server, one process per model.

- llama.cpp sizes whatever isn't set (GPU layers, context) to the memory that's
  free when the model loads (`--fit`, on by default), so nothing is estimated here.
- A server listens on 127.0.0.1 only. The app itself may be reachable from the
  network, but llama-server has no authentication of its own.
- A server mustn't outlive the app, or it goes on holding GPU memory and its
  port. They're stopped when the app shuts down, and on Windows they're also
  tied to the app's process, so they go even if the app is killed.
"""

from __future__ import annotations

import logging
import re
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from models.servers import LoadedModel, ModelServer, ServerLogs, ServerSettings
from services.llama_runtime import (
    LaunchableRuntime,
    LlamaRuntimeService,
    llama_runtime_service,
    runtime_environment,
    server_binary_name,
)
from services.model_downloads import AUTO, LocalSelection, ModelDownloadService, model_download_service
from services.settings import llama_server_logs_dir

logger = logging.getLogger(__name__)

HOST = "127.0.0.1"
FIRST_PORT = 8080
PORT_COUNT = 100
HEALTH_INTERVAL_SECONDS = 0.5
# A big model on a slow disk takes minutes. One that takes longer than this has hung.
LOAD_TIMEOUT_SECONDS = 15 * 60
STOP_TIMEOUT_SECONDS = 10
MAX_FINISHED_SERVERS = 20
MAX_LOG_FILES = 30
LOG_TAIL_BYTES = 256 * 1024
# Where a crash's cause is: the end of the log, not some earlier request's error.
FAILURE_SEARCH_LINES = 60
ACTIVE_STATUSES = ("starting", "ready", "stopping")
FINISHED_STATUSES = ("stopped", "failed")

# llama-server takes settings from the environment too. The app passes everything
# it wants as arguments, and the server has no use for the app's tokens.
_ENVIRONMENT_SETTING_PREFIX = "LLAMA_ARG_"
_WITHHELD_VARIABLES = {"LLAMA_API_KEY", "MTMD_BACKEND_DEVICE", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "GITHUB_TOKEN"}
# `0.00.057.977 E gguf_init_from_file: failed to open GGUF file ...`
_ERROR_LINE = re.compile(r"^\S+ E (?P<message>.+)$")


class ServerNotFoundError(Exception):
    pass


class ServerConflictError(Exception):
    """The launch is valid but clashes with something running."""


class ServerStartError(Exception):
    """The process couldn't be started at all."""


def _now() -> datetime:
    return datetime.now(UTC)


def model_name_for(selection: LocalSelection) -> str:
    """What API requests send as `model`, in the `repo:quantization` form llama.cpp uses."""
    return f"{selection.repo_id}:{selection.model.id}"


def device_for(runtime: LaunchableRuntime, settings: ServerSettings) -> str | None:
    return "none" if settings.gpu_layers == 0 else runtime.gpu_device


def server_arguments(
    selection: LocalSelection, runtime: LaunchableRuntime, settings: ServerSettings, port: int
) -> list[str]:
    arguments = [
        "--model", selection.model.path,
        "--host", HOST,
        "--port", str(port),
        "--alias", model_name_for(selection),
        # Everything it needs is on disk, so it has no reason to go online.
        "--offline",
        "--log-colors", "off",
        # Its API has no key, so pages on other sites mustn't be able to call it.
        "--cors-origins", "localhost",
    ]
    if selection.vision is not None:
        arguments += ["--mmproj", selection.vision.path]
    device = device_for(runtime, settings)
    if device is not None:
        # Named, so a model that doesn't fit on the GPU spills into system RAM
        # rather than onto an integrated GPU, which is slower than the CPU.
        arguments += ["--device", device]
    if device not in (None, "none"):
        # Read the file instead of memory-mapping it. On Windows a mapping can't
        # be released piece by piece, so the parts copied to the GPU would stay
        # in RAM as well: 6.7 GB of RAM for a 6.5 GB model fully on the GPU,
        # against 1.1 GB read. A CPU-only model is used straight from the
        # mapping, so it keeps the default.
        arguments += ["--load-mode", "none"]
    if device != "none" and settings.gpu_layers != "auto":
        arguments += ["--gpu-layers", str(settings.gpu_layers)]
    if settings.context_size is not None:
        arguments += ["--ctx-size", str(settings.context_size)]
    return arguments


def server_environment(runtime: LaunchableRuntime) -> dict[str, str]:
    environment = runtime_environment(runtime.directory, runtime.backend)
    for name in list(environment):
        upper = name.upper()
        if upper.startswith(_ENVIRONMENT_SETTING_PREFIX) or upper in _WITHHELD_VARIABLES:
            del environment[name]
    return environment


def default_command(runtime: LaunchableRuntime) -> list[str]:
    return [str(runtime.directory / server_binary_name())]


def failure_reason(lines: list[str]) -> str | None:
    """The first error near the end of the log. It names the cause; the errors
    after it are the same failure reported further up. None if nothing was
    logged as an error, as when the process was killed."""
    recent = [line.strip() for line in lines[-FAILURE_SEARCH_LINES:] if line.strip()]
    for line in recent:
        match = _ERROR_LINE.match(line)
        if match:
            return _shorten(match["message"])
    # Older builds log in other formats.
    mentioned = next((line for line in recent if re.search(r"error|failed", line, re.IGNORECASE)), None)
    return _shorten(mentioned) if mentioned else None


def _shorten(text: str, limit: int = 400) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


def read_log_tail(path: Path, lines: int) -> list[str]:
    try:
        with open(path, "rb") as log:
            log.seek(0, 2)
            start = max(log.tell() - LOG_TAIL_BYTES, 0)
            # One byte early, to see whether the first line is whole.
            log.seek(max(start - 1, 0))
            data = log.read()
    except OSError:
        return []
    cut = False
    if start > 0:
        cut = not data.startswith(b"\n")
        data = data[1:]
    tail = data.decode("utf-8", errors="replace").splitlines()
    if cut:
        tail = tail[1:]
    return tail[-lines:] if lines > 0 else []


def port_is_free(port: int) -> bool:
    """Whether a server could listen on this port on 127.0.0.1. Binding alone
    doesn't show it: Windows lets a port that something is listening on for
    every address be bound again for one address, which on 127.0.0.1 would take
    it from the app itself."""
    try:
        with socket.create_connection((HOST, port), timeout=0.3):
            return False
    except OSError:
        pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((HOST, port))
        except OSError:
            return False
    return True


class _WindowsJob:
    """A Windows job object that ends every process in it once the app's handle
    to it closes, which happens however the app exits, even when it's killed."""

    JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        class IoCounters(ctypes.Structure):
            _fields_ = [
                (name, ctypes.c_ulonglong)
                for name in (
                    "ReadOperationCount",
                    "WriteOperationCount",
                    "OtherOperationCount",
                    "ReadTransferCount",
                    "WriteTransferCount",
                    "OtherTransferCount",
                )
            ]

        class BasicLimits(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimits),
                ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        self._ctypes = ctypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        self._kernel32 = kernel32

        # Not inheritable, so a server can't keep its own job alive.
        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = self.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            handle, self.JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        # Deliberately never closed: closing it is what ends the servers.
        self._handle = handle

    def add(self, process: subprocess.Popen[bytes]) -> None:
        handle = int(process._handle)  # type: ignore[attr-defined]
        if not self._kernel32.AssignProcessToJobObject(self._handle, handle):
            raise self._ctypes.WinError(self._ctypes.get_last_error())


_job: _WindowsJob | None = None
_job_lock = threading.Lock()


def tie_to_app(process: subprocess.Popen[bytes]) -> None:
    """Makes the process end with the app. Only needed on Windows: elsewhere a
    killed app is rare enough to leave to the shutdown stopping everything."""
    global _job
    if sys.platform != "win32":
        return
    try:
        with _job_lock:
            if _job is None:
                _job = _WindowsJob()
        _job.add(process)
    except OSError as error:
        logger.warning("llama-server (pid %s) could outlive the app if it crashes: %s", process.pid, error)


def _terminate(process: subprocess.Popen[bytes]) -> None:
    try:
        process.terminate()
    except OSError:
        pass  # Already gone.


def _reap(process: subprocess.Popen[bytes]) -> None:
    try:
        process.wait(STOP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


@dataclass
class _Server:
    info: ModelServer
    # The settings a repeated launch has to match to join this server.
    signature: tuple[Any, ...]
    process: subprocess.Popen[bytes]
    stop_requested: bool = False
    failure: str | None = None
    done: threading.Event = field(default_factory=threading.Event)


class LlamaServerService:
    def __init__(
        self,
        runtimes: LlamaRuntimeService = llama_runtime_service,
        library: ModelDownloadService = model_download_service,
        logs_dir: Path | None = None,
        command: Callable[[LaunchableRuntime], list[str]] = default_command,
        http: httpx.Client | None = None,
    ) -> None:
        self.runtimes = runtimes
        self.library = library
        self._logs_dir = logs_dir
        self._command = command
        # Local only, so a proxy configured for the internet mustn't be used.
        self._http = http or httpx.Client(trust_env=False, timeout=httpx.Timeout(2.0))
        self._servers: dict[str, _Server] = {}
        self._lock = threading.Lock()
        runtimes.add_busy_check(self._runtime_busy)
        library.add_busy_check(self._model_busy)

    @property
    def logs_dir(self) -> Path:
        return self._logs_dir or llama_server_logs_dir()

    # Starting

    def launch(
        self,
        repo_id: str,
        quantization: str = AUTO,
        vision_encoder: str = AUTO,
        context_size: int | None = None,
        gpu_layers: int | str = "auto",
        port: int | None = None,
        runtime_id: str | None = None,
    ) -> ModelServer:
        """Starts a server and returns while the model loads. Everything is
        checked first, so a bad request fails here rather than in the log."""
        selection = self.library.local_selection(repo_id, quantization, vision_encoder)
        runtime = self.runtimes.launchable(runtime_id)
        settings = ServerSettings(context_size=context_size, gpu_layers=gpu_layers)
        signature = (selection.vision.id if selection.vision else None, runtime.id, context_size, gpu_layers)
        # The library gives the repo's own spelling, so this matches however it was asked for.
        name = model_name_for(selection)

        with self._lock:
            running = next(
                (
                    server
                    for server in self._servers.values()
                    if server.info.status in ACTIVE_STATUSES and server.info.model_name == name
                ),
                None,
            )
            if running is not None:
                return self._join(running, signature, port)
            server = self._start(selection, runtime, settings, self._choose_port(port), signature)
            return server.info.model_copy(deep=True)

    @staticmethod
    def _join(server: _Server, signature: tuple[Any, ...], port: int | None) -> ModelServer:
        """The same launch again gets the server already running it. One model
        isn't run twice: a second copy would only take memory from the first."""
        info = server.info
        if info.status == "stopping":
            raise ServerConflictError(f"{info.model_name} is stopping. Try again once it has.")
        if server.signature != signature or port not in (None, info.port):
            raise ServerConflictError(
                f"{info.model_name} is already running with other settings, as server {info.id}. Stop it first."
            )
        return info.model_copy(deep=True)

    def _choose_port(self, wanted: int | None) -> int:
        taken = {server.info.port for server in self._servers.values() if server.info.status in ACTIVE_STATUSES}
        if wanted is not None:
            if wanted in taken or not port_is_free(wanted):
                raise ServerConflictError(f"Port {wanted} is in use.")
            return wanted
        for port in range(FIRST_PORT, FIRST_PORT + PORT_COUNT):
            if port not in taken and port_is_free(port):
                return port
        raise ServerConflictError(f"No port between {FIRST_PORT} and {FIRST_PORT + PORT_COUNT - 1} is free.")

    def _start(
        self,
        selection: LocalSelection,
        runtime: LaunchableRuntime,
        settings: ServerSettings,
        port: int,
        signature: tuple[Any, ...],
    ) -> _Server:
        server_id = uuid.uuid4().hex[:12]
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._prune_logs()
        log_path = self.logs_dir / f"{datetime.now():%Y%m%d-%H%M%S}-{server_id}.log"
        command = [*self._command(runtime), *server_arguments(selection, runtime, settings, port)]
        with open(log_path, "wb") as log:
            log.write(f"> {subprocess.list2cmdline(command)}\n\n".encode())
            log.flush()
            try:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=runtime.directory,
                    env=server_environment(runtime),
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
                )
            except OSError as error:
                raise ServerStartError(f"Couldn't start llama-server from {runtime.id}: {error}") from error
        tie_to_app(process)

        info = ModelServer(
            id=server_id,
            repo_id=selection.repo_id,
            quantization=selection.model.id,
            vision_encoder=selection.vision.id if selection.vision else None,
            model_path=selection.model.path,
            vision_encoder_path=selection.vision.path if selection.vision else None,
            runtime_id=runtime.id,
            backend=runtime.backend,
            device=device_for(runtime, settings),
            settings=settings,
            host=HOST,
            port=port,
            base_url=f"http://{HOST}:{port}",
            model_name=model_name_for(selection),
            pid=process.pid,
            log_path=str(log_path),
            created_at=_now(),
        )
        server = _Server(info=info, signature=signature, process=process)
        self._servers[server_id] = server
        self._prune_finished()
        threading.Thread(target=self._watch, args=(server,), name=f"llama-server-{server_id}", daemon=True).start()
        return server

    # Watching

    def _watch(self, server: _Server) -> None:
        """Waits for the model to load, then for the process to end, however it does."""
        process = server.process
        deadline = time.monotonic() + LOAD_TIMEOUT_SECONDS
        while self._status(server) == "starting":
            try:
                process.wait(HEALTH_INTERVAL_SECONDS)
                break  # It exited while loading.
            except subprocess.TimeoutExpired:
                pass
            if self._is_healthy(server.info.port):
                loaded = self._loaded_model(server.info.port)
                with self._lock:
                    if server.info.status == "starting":
                        server.info.status = "ready"
                        server.info.ready_at = _now()
                        server.info.loaded = loaded
                break
            if time.monotonic() > deadline:
                server.failure = f"The model didn't finish loading within {LOAD_TIMEOUT_SECONDS // 60} minutes."
                _terminate(process)
                _reap(process)
                break
        self._finish(server, process.wait())

    def _status(self, server: _Server) -> str:
        with self._lock:
            return server.info.status

    def _is_healthy(self, port: int) -> bool:
        try:
            return self._http.get(f"http://{HOST}:{port}/health").status_code == 200
        except httpx.HTTPError:
            return False

    def _loaded_model(self, port: int) -> LoadedModel | None:
        try:
            props = self._http.get(f"http://{HOST}:{port}/props").json()
            settings = props.get("default_generation_settings") or {}
            modalities = props.get("modalities") or {}
            return LoadedModel(
                context_size=settings.get("n_ctx"),
                slots=props.get("total_slots"),
                vision=bool(modalities.get("vision")),
                build=props.get("build_info"),
            )
        except (httpx.HTTPError, ValueError, AttributeError) as error:
            logger.info("Could not read what llama-server on port %s loaded: %s", port, error)
            return None

    def _finish(self, server: _Server, exit_code: int) -> None:
        with self._lock:
            was_ready = server.info.ready_at is not None
            stopped = server.stop_requested
        error = None
        if not stopped:
            reason = server.failure or failure_reason(read_log_tail(Path(server.info.log_path), FAILURE_SEARCH_LINES))
            when = "stopped unexpectedly" if was_ready else "exited while loading the model"
            cause = f": {reason}" if reason else ". Its log may say why"
            error = server.failure or f"llama-server {when} (exit code {exit_code}){cause}"
        with self._lock:
            server.info.status = "stopped" if stopped else "failed"
            server.info.exit_code = exit_code
            server.info.error = error
            server.info.stopped_at = _now()
        server.done.set()

    # Stopping

    def stop(self, server_id: str) -> ModelServer:
        """Stops a server and waits for it to go, freeing its memory."""
        with self._lock:
            server = self._get(server_id)
            if server.info.status not in FINISHED_STATUSES:
                server.stop_requested = True
                server.info.status = "stopping"
        if not server.done.is_set():
            _terminate(server.process)
            _reap(server.process)
            server.done.wait(STOP_TIMEOUT_SECONDS)
        return self.get(server_id)

    def stop_all(self) -> None:
        """For shutdown: asks every server to stop at once, then waits for them."""
        with self._lock:
            running = [server for server in self._servers.values() if server.info.status in ACTIVE_STATUSES]
            for server in running:
                server.stop_requested = True
                server.info.status = "stopping"
        for server in running:
            _terminate(server.process)
        for server in running:
            _reap(server.process)

    # Looking

    def get(self, server_id: str) -> ModelServer:
        with self._lock:
            return self._get(server_id).info.model_copy(deep=True)

    def all_servers(self) -> list[ModelServer]:
        """Newest first, including recently stopped ones."""
        with self._lock:
            return [server.info.model_copy(deep=True) for server in reversed(self._servers.values())]

    def logs(self, server_id: str, lines: int = 200) -> ServerLogs:
        info = self.get(server_id)
        return ServerLogs(id=info.id, path=info.log_path, lines=read_log_tail(Path(info.log_path), lines))

    def _get(self, server_id: str) -> _Server:
        server = self._servers.get(server_id)
        if server is None:
            raise ServerNotFoundError(f"No server {server_id}. Servers don't survive a restart of the app.")
        return server

    # What a running server needs left alone

    def _runtime_busy(self, runtime_id: str) -> str | None:
        return self._busy_reason(lambda info: info.runtime_id == runtime_id, runtime_id)

    def _model_busy(self, repo_id: str) -> str | None:
        return self._busy_reason(lambda info: info.repo_id.casefold() == repo_id.casefold(), repo_id)

    def _busy_reason(self, uses: Callable[[ModelServer], bool], what: str) -> str | None:
        with self._lock:
            active = (server for server in self._servers.values() if server.info.status in ACTIVE_STATUSES)
            server = next((server for server in active if uses(server.info)), None)
            if server is None:
                return None
            return f"{what} is in use by server {server.info.id} ({server.info.model_name}). Stop it first."

    # Housekeeping

    def _prune_finished(self) -> None:
        finished = [key for key, server in self._servers.items() if server.info.status in FINISHED_STATUSES]
        for key in finished[: max(len(finished) - MAX_FINISHED_SERVERS, 0)]:
            del self._servers[key]

    def _prune_logs(self) -> None:
        in_use = {
            Path(server.info.log_path).name
            for server in self._servers.values()
            if server.info.status in ACTIVE_STATUSES
        }
        logs = sorted(self.logs_dir.glob("*.log"), key=lambda path: path.name, reverse=True)
        for old in [path for path in logs if path.name not in in_use][MAX_LOG_FILES:]:
            try:
                old.unlink()
            except OSError:
                pass  # Open in an editor, say. It'll go next time.


llama_server_service = LlamaServerService()

"""Runs downloaded models with llama-server, one process per model.

- llama.cpp sizes whatever isn't set (GPU layers, context) to fit the GPU
  (`--fit`, on by default). It plans against the memory the driver says is free,
  which on Windows ignores what other programs are using, and it leaves out the
  vision encoder. So before each launch the app measures what's really free and
  tells it how much more to leave (`--fit-target`).
- A model that fits on the GPU whole is read in rather than memory-mapped: on
  Windows a mapping keeps the whole file in RAM too. One that doesn't fit keeps
  the default, because reading it in pins its CPU part in RAM for good.
- A server listens on 127.0.0.1 only. The app itself may be reachable from the
  network, but llama-server has no authentication of its own.
- A server mustn't outlive the app, or it goes on holding GPU memory and its
  port. They're stopped when the app shuts down, and on Windows they're also
  tied to the app's process, so they go even if the app is killed.
"""

from __future__ import annotations

import logging
import math
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

from models.hardware import GpuInfo
from models.servers import GpuMemory, GpuPlan, LoadedModel, ModelServer, ServerLogs, ServerSettings
from services.hardware import HardwareService, hardware_service
from services.llama_runtime import (
    LaunchableRuntime,
    LlamaRuntimeService,
    RuntimeConflictError,
    RuntimeNotFoundError,
    llama_runtime_service,
    match_gpu_device,
    runtime_environment,
    same_gpu_name,
    server_binary_name,
)
from services.model_downloads import (
    AUTO,
    LANGUAGE_MODEL,
    VISION_ENCODER,
    LocalModelNotFoundError,
    LocalSelection,
    ModelDownloadService,
    model_download_service,
)
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

# llama.cpp's own default for what `--fit` leaves free.
FIT_MARGIN_MIB = 1024
# A vision encoder's working buffers, on top of its file. Measured at 1.2 GB in
# all for a 0.9 GB encoder.
VISION_OVERHEAD_BYTES = 512 * 1024**2
# llama-fit-params takes a second or two, and longer when little fits.
FIT_TIMEOUT_SECONDS = 120
# Windows' GPU counters lag a little behind what was just allocated.
GPU_MEMORY_DELAY_SECONDS = 2.0
# More than this in system RAM means VRAM ran out. Buffers the GPU reads from
# RAM by design come to a few hundred MB.
SPILL_WARNING_BYTES = 1024**3

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


@dataclass(frozen=True)
class LaunchPlan:
    """How a launch uses the GPU, worked out just before it starts."""

    # What to pass to `--device`: an id like `ROCm1`, `none` for the CPU only, or
    # None to leave it to llama.cpp.
    device: str | None
    # That GPU as the hardware service names it, to read its memory use by.
    gpu_name: str | None = None
    gpu: GpuPlan | None = None


def model_name_for(selection: LocalSelection) -> str:
    """What API requests send as `model`, in the `repo:quantization` form llama.cpp uses."""
    return f"{selection.repo_id}:{selection.model.id}"


def fit_target_mib(reported_free: int | None, actual_free: int | None, vision_bytes: int = 0) -> int:
    """What `--fit` should leave free, in MiB: llama.cpp's own margin, plus any
    memory the driver counts as free that other programs are using, plus the
    vision encoder, which `--fit` doesn't count."""
    hidden = 0
    if reported_free is not None and actual_free is not None:
        hidden = max(reported_free - actual_free, 0)
    return FIT_MARGIN_MIB + math.ceil((hidden + vision_bytes) / 1024**2)


def fit_arguments(selection: LocalSelection, settings: ServerSettings, device: str, target_mib: int) -> list[str]:
    """For llama-fit-params: what llama-server will be given that affects the fit."""
    arguments = ["--model", selection.model.path, "--device", device, "--fit-target", str(target_mib)]
    if settings.context_size is not None:
        arguments += ["--ctx-size", str(settings.context_size)]
    if settings.gpu_layers == "all":
        arguments += ["--gpu-layers", "all"]
    return arguments


def whole_model_on_gpu(fitted: str) -> bool | None:
    """Reads llama-fit-params' answer, such as `-c 174080 -ngl -1`. A negative
    layer count means all of them. Anything else it adds moves tensors off the
    GPU, as it does with a mixture-of-experts model's experts. None if the
    answer can't be read."""
    tokens = fitted.split()
    if len(tokens) % 2:
        return None
    answer = dict(zip(tokens[::2], tokens[1::2], strict=True))
    try:
        layers = int(answer["-ngl"])
    except (KeyError, ValueError):
        return None
    if set(answer) - {"-c", "-ngl"}:
        return False
    return layers < 0


def run_fit_params(runtime: LaunchableRuntime, arguments: list[str]) -> str | None:
    """llama-fit-params' answer for these arguments: what llama-server would
    choose for whatever isn't set. It plans without loading the model. None if
    this build doesn't have it or it failed."""
    binary = runtime.directory / ("llama-fit-params.exe" if sys.platform == "win32" else "llama-fit-params")
    if not binary.is_file():
        return None
    try:
        completed = subprocess.run(
            [str(binary), *arguments],
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=FIT_TIMEOUT_SECONDS,
            cwd=runtime.directory,
            env=server_environment(runtime),
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
    except (OSError, subprocess.SubprocessError) as error:
        logger.info("llama-fit-params didn't run: %s", error)
        return None
    if completed.returncode != 0:
        logger.info("llama-fit-params failed (exit code %s)", completed.returncode)
        return None
    return completed.stdout.decode("utf-8", errors="replace").strip()


def server_arguments(selection: LocalSelection, settings: ServerSettings, port: int, plan: LaunchPlan) -> list[str]:
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
    if plan.device is not None:
        # Named, so a model that doesn't fit on the GPU spills into system RAM
        # rather than onto an integrated GPU, which is slower than the CPU.
        arguments += ["--device", plan.device]
    gpu = plan.gpu
    if gpu is not None and gpu.fit_target_mib is not None:
        arguments += ["--fit-target", str(gpu.fit_target_mib)]
    if gpu is not None and gpu.whole_model_on_gpu:
        # Read in rather than memory-mapped. On Windows a mapping can't be let go
        # of piece by piece, so the parts copied to the GPU would stay in RAM as
        # well: 6.7 GB of RAM for a 6.5 GB model, against 1.1 GB read in. And
        # `--no-host` keeps the little left on the CPU out of pinned memory.
        arguments += ["--load-mode", "none", "--no-host"]
    if plan.device != "none" and settings.gpu_layers != "auto":
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
    """Whether a server could listen on this port on 127.0.0.1 without taking
    it from anyone. Binding to 127.0.0.1 alone doesn't show it: Windows allows
    that while something listens on every address, as the app does on 0.0.0.0.
    So the catch-all addresses are tried too. Binding answers at once, where
    connecting to a free port makes Windows wait for a timeout."""
    for family, address in ((socket.AF_INET, HOST), (socket.AF_INET, "0.0.0.0"), (socket.AF_INET6, "::")):
        try:
            probe = socket.socket(family, socket.SOCK_STREAM)
        except OSError:
            continue  # No IPv6 on this machine.
        with probe:
            if family == socket.AF_INET6:
                # Both stacks, to see a listener on [::] that takes IPv4 as well.
                probe.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            try:
                probe.bind((address, port))
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
    gpu_name: str | None = None
    stop_requested: bool = False
    failure: str | None = None
    done: threading.Event = field(default_factory=threading.Event)


@dataclass(frozen=True)
class _Use:
    """What a server, or a launch on its way to being one, needs left alone."""

    model_name: str
    repo_id: str
    quantization: str
    vision_encoder: str | None
    runtime_id: str

    @classmethod
    def of(cls, info: ModelServer) -> _Use:
        return cls(info.model_name, info.repo_id, info.quantization, info.vision_encoder, info.runtime_id)


class LlamaServerService:
    def __init__(
        self,
        runtimes: LlamaRuntimeService = llama_runtime_service,
        library: ModelDownloadService = model_download_service,
        hardware: HardwareService = hardware_service,
        logs_dir: Path | None = None,
        command: Callable[[LaunchableRuntime], list[str]] = default_command,
        fit_params: Callable[[LaunchableRuntime, list[str]], str | None] = run_fit_params,
        http: httpx.Client | None = None,
    ) -> None:
        self.runtimes = runtimes
        self.library = library
        self._hardware = hardware
        self._logs_dir = logs_dir
        self._command = command
        self._fit_params = fit_params
        # Local only, so a proxy configured for the internet mustn't be used.
        self._http = http or httpx.Client(trust_env=False, timeout=httpx.Timeout(2.0))
        self._servers: dict[str, _Server] = {}
        # Launches between being checked and their server starting.
        self._launching: dict[str, _Use] = {}
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
            if any(use.model_name == name for use in self._launching.values()):
                raise ServerConflictError(f"{name} is being launched already. Try again once it has started.")
            if port is not None:
                self._choose_port(port)  # So a port that's taken fails now, not after the planning.
            token = uuid.uuid4().hex
            # From here its files and build can't be deleted, as for a running server.
            self._launching[token] = _Use(
                name,
                selection.repo_id,
                selection.model.id,
                selection.vision.id if selection.vision else None,
                runtime.id,
            )
        try:
            self._check_still_there(selection, runtime)
            plan = self._plan(selection, runtime, settings)
            with self._lock:
                server = self._start(selection, runtime, settings, self._choose_port(port), signature, plan)
                return server.info.model_copy(deep=True)
        finally:
            with self._lock:
                del self._launching[token]

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

    def _check_still_there(self, selection: LocalSelection, runtime: LaunchableRuntime) -> None:
        """The files were looked up before the launch was registered, so a delete
        could have come in between. One that comes after is refused."""
        for option in (selection.model, selection.vision):
            if option is not None and not Path(option.path).is_file():
                raise LocalModelNotFoundError(f"{selection.repo_id} {option.id} has just been deleted.")
        if not Path(self._command(runtime)[0]).is_file():
            raise RuntimeNotFoundError(f"{runtime.id} has just been deleted.")

    def _plan(self, selection: LocalSelection, runtime: LaunchableRuntime, settings: ServerSettings) -> LaunchPlan:
        """The device to run on, how much llama.cpp should leave free on it, and
        whether the model fits there whole. Takes a second or two: it asks the
        build what it sees, and llama-fit-params how it would fit the model."""
        if settings.gpu_layers == 0:
            return LaunchPlan(device="none")
        if runtime.gpu_device is None:
            return LaunchPlan(device=None)  # A CPU build, or one that found no GPU it could tell apart.
        gpus = self._gpus()
        device, device_name, reported_free = runtime.gpu_device, None, None
        listed = self.runtimes.current_devices(runtime)
        if listed is not None:
            devices, free = listed
            if not devices:
                raise RuntimeConflictError(
                    f"{runtime.id} can't see a GPU right now. Check the graphics driver, "
                    "then test the build again with GET /runtimes/llama.cpp?refresh=true."
                )
            # Found by name again, since numbering changes when a GPU is added or disabled.
            device = (match_gpu_device(devices, gpus) if gpus else None) or device
            current = next((listed_device for listed_device in devices if listed_device.id == device), None)
            if current is None:
                raise RuntimeConflictError(
                    f"{runtime.id} no longer has the device {device} it was tested with. "
                    "Test it again with GET /runtimes/llama.cpp?refresh=true."
                )
            device_name, reported_free = current.name, free.get(device)

        matches = [gpu for gpu in gpus if device_name is not None and same_gpu_name(device_name, gpu.name)]
        known_free = [gpu.vram_free_bytes for gpu in matches if gpu.vram_free_bytes is not None]
        # Two identical cards can't be told apart, so the fuller one counts.
        actual_free = min(known_free) if known_free else None
        vision_bytes = 0
        if selection.vision is not None:
            try:
                vision_bytes = Path(selection.vision.path).stat().st_size + VISION_OVERHEAD_BYTES
            except OSError:
                pass  # Gone since the check; the launch will say so.
        target = fit_target_mib(reported_free, actual_free, vision_bytes)
        whole = None
        if settings.gpu_layers in ("auto", "all"):
            fitted = self._fit_params(runtime, fit_arguments(selection, settings, device, target))
            whole = whole_model_on_gpu(fitted) if fitted else None
        return LaunchPlan(
            device=device,
            gpu_name=matches[0].name if matches else None,
            gpu=GpuPlan(vram_free_bytes=actual_free, fit_target_mib=target, whole_model_on_gpu=whole),
        )

    def _gpus(self) -> list[GpuInfo]:
        try:
            return list(self._hardware.get_hardware().gpus)
        except Exception:
            logger.exception("Could not read the GPUs to plan a launch")
            return []

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
        plan: LaunchPlan,
    ) -> _Server:
        server_id = uuid.uuid4().hex[:12]
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._prune_logs()
        log_path = self.logs_dir / f"{datetime.now():%Y%m%d-%H%M%S}-{server_id}.log"
        command = [*self._command(runtime), *server_arguments(selection, settings, port, plan)]
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
            device=plan.device,
            settings=settings,
            gpu_plan=plan.gpu,
            host=HOST,
            port=port,
            base_url=f"http://{HOST}:{port}",
            model_name=model_name_for(selection),
            pid=process.pid,
            log_path=str(log_path),
            created_at=_now(),
        )
        server = _Server(info=info, signature=signature, process=process, gpu_name=plan.gpu_name)
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
                self._record_gpu_memory(server)
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

    def _record_gpu_memory(self, server: _Server) -> None:
        """Where its GPU memory ended up. Windows quietly puts what doesn't fit
        in VRAM in system RAM, where the model runs several times slower once
        its context fills, so that's worth saying."""
        if server.gpu_name is None:
            return
        try:
            server.process.wait(GPU_MEMORY_DELAY_SECONDS)
            return  # It has already stopped.
        except subprocess.TimeoutExpired:
            pass
        reading = self._hardware.process_gpu_memory(server.process.pid, server.gpu_name)
        if reading is None:
            return
        vram, system_ram = reading
        with self._lock:
            if server.info.status != "ready":
                return
            loaded = server.info.loaded = server.info.loaded or LoadedModel()
            loaded.gpu_memory = GpuMemory(vram_bytes=vram, system_ram_bytes=system_ram)
            if system_ram > SPILL_WARNING_BYTES:
                server.info.warnings.append(
                    f"{system_ram / 1e9:.1f} GB of its GPU memory is in system RAM rather than VRAM, so it runs "
                    "slower, most of all as the context fills. Close other programs using the GPU or launch it "
                    "with a smaller context_size. If VRAM is free and this keeps happening, restart Windows."
                )

    def _finish(self, server: _Server, exit_code: int) -> None:
        with self._lock:
            was_ready = server.info.ready_at is not None
            stopped = server.stop_requested
        error = None
        if not stopped and server.failure:
            error = server.failure
        elif not stopped:
            reason = failure_reason(read_log_tail(Path(server.info.log_path), FAILURE_SEARCH_LINES))
            when = "stopped unexpectedly" if was_ready else "exited while loading the model"
            cause = f": {reason}" if reason else ". Its log may say why"
            error = f"llama-server {when} (exit code {exit_code}){cause}"
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
        return self._busy_reason(lambda use: use.runtime_id == runtime_id, runtime_id)

    def _model_busy(self, repo_id: str, options: set[tuple[str, str]] | None) -> str | None:
        """`options` are the (role, option id) pairs being deleted, or None for
        the whole repo. Deleting a quantization nothing is running is fine."""

        def uses(use: _Use) -> bool:
            if use.repo_id.casefold() != repo_id.casefold():
                return False
            if options is None:
                return True
            return bool({(LANGUAGE_MODEL, use.quantization), (VISION_ENCODER, use.vision_encoder)} & options)

        return self._busy_reason(uses, repo_id)

    def _busy_reason(self, uses: Callable[[_Use], bool], what: str) -> str | None:
        with self._lock:
            for server in self._servers.values():
                if server.info.status in ACTIVE_STATUSES and uses(_Use.of(server.info)):
                    return f"{what} is in use by server {server.info.id} ({server.info.model_name}). Stop it first."
            for use in self._launching.values():
                if uses(use):
                    return f"{what} is in use by {use.model_name}, which is starting. Stop it first, once it has."
        return None

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

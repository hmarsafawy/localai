"""`FAKE_SERVER` stands in for llama-server, behaving as the real one did when
tested: `/health` answers 503 while the model loads and 200 once it's ready, and
a load failure logs `E` lines and exits with code 1. What each fake does is read
from its "model" file, and it records the arguments and environment it got."""

from __future__ import annotations

import json
import math
import os
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import httpx
import psutil
import pytest
import uvicorn
from fastapi.testclient import TestClient

import main
import routes.inference_router as inference_router
import routes.server_router as server_router
import services.llama_servers as servers_module
import services.processes as processes_module
from main import app
from models.downloads import LocalModelOption
from models.hardware import GpuInfo
from models.runtime import RuntimeDevice
from models.servers import GpuMemory, GpuPlan, ModelServer, ServerSettings
from services.inference_proxy import InferenceProxy
from services.llama_runtime import LaunchableRuntime, RuntimeConflictError, RuntimeNotFoundError
from services.llama_servers import (
    LaunchPlan,
    LlamaServerService,
    ModelLoadingError,
    ModelNotRunningError,
    ServerConflictError,
    ServerNotFoundError,
    failure_reason,
    fit_arguments,
    fit_target_mib,
    port_is_free,
    read_log_tail,
    server_arguments,
    whole_model_on_gpu,
)
from services.model_downloads import (
    LocalModelBusyError,
    LocalModelChoiceError,
    LocalModelNotFoundError,
    LocalSelection,
)

# The venv's python.exe on Windows is a launcher that runs the real one as a
# child, so stopping it would leave the fake server running.
PYTHON = getattr(sys, "_base_executable", sys.executable)
REPO = "unsloth/Qwen3.8-27B-GGUF"
RUNTIME_ID = "llama-b11960-bin-win-rocm-10.0-x64"
MIB = 1024**2
GIB = 1024**3

# An RX 9070 XT with another 3 GB in use by other programs, which ROCm on
# Windows doesn't see: it says 16140 MiB is free.
RX_9070 = GpuInfo(
    name="AMD Radeon RX 9070 XT", vendor="amd", kind="discrete", vram_total_bytes=16 * GIB, vram_free_bytes=13 * GIB
)
IGPU = GpuInfo(name="AMD Radeon(TM) Graphics", vendor="amd", kind="integrated")
ROCM_DEVICES = (
    [
        RuntimeDevice(id="ROCm0", name="AMD Radeon(TM) Graphics"),
        RuntimeDevice(id="ROCm1", name="AMD Radeon RX 9070 XT"),
    ],
    {"ROCm0": 12193 * MIB, "ROCm1": 16140 * MIB},
)

FAKE_SERVER = r'''
import json, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

args = sys.argv[1:]
def value(flag):
    return args[args.index(flag) + 1] if flag in args else None

model = value("--model")
with open(model, encoding="utf-8") as file:
    behaviour = json.loads(file.read() or "{}")
with open(model + ".launch.json", "w", encoding="utf-8") as record:
    json.dump({"args": args, "env": dict(os.environ), "cwd": os.getcwd()}, record)

print("0.00.000.001 I srv  llama_server: initializing ...", flush=True)
if "fail" in behaviour:
    print("0.00.000.002 E llama_model_load: error loading model: " + behaviour["fail"], flush=True)
    print("0.00.000.003 E srv  llama_server: exiting due to model loading error", flush=True)
    sys.exit(1)
started = time.monotonic()

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *ignored):
        pass

    def send(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        loading = behaviour.get("hang") or time.monotonic() - started < behaviour.get("load_seconds", 0.1)
        if self.path == "/health":
            if loading:
                self.send(503, {"error": {"message": "Loading model", "code": 503}})
            else:
                self.send(200, {"status": "ok"})
        elif self.path == "/props":
            context = value("--ctx-size")
            self.send(200, {
                "default_generation_settings": {"n_ctx": int(context) if context else 262144},
                "total_slots": 4,
                "modalities": {"vision": "--mmproj" in args},
                "build_info": "b11960-fake",
            })
        else:
            self.send(404, {})

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        with open(model + ".requests.jsonl", "a", encoding="utf-8") as record:
            record.write(json.dumps({"path": self.path, "headers": dict(self.headers), "body": body.decode()}) + "\n")
        if self.path not in ("/v1/chat/completions", "/v1/completions"):
            return self.send(404, {"error": {"code": 404, "message": "File Not Found", "type": "not_found_error"}})
        if "reply_error" in behaviour:
            return self.send(400, {"error": {"code": 400, "message": behaviour["reply_error"], "type": "exceed_context_size_error"}})
        words = behaviour.get("reply", "Hello from the fake").split()
        if not json.loads(body).get("stream"):
            return self.send(200, {"object": "chat.completion", "choices": [{"message": {"content": " ".join(words)}}]})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            for word in words:
                chunk = {"object": "chat.completion.chunk", "choices": [{"delta": {"content": word + " "}}]}
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                self.wfile.flush()
                time.sleep(behaviour.get("word_seconds", 0))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except OSError:
            # The client went away, which is how llama-server knows to stop generating.
            open(model + ".cancelled", "w").close()

time.sleep(behaviour.get("listen_after", 0))
server = ThreadingHTTPServer((value("--host"), int(value("--port"))), Handler)
if behaviour.get("crashes"):
    def crash():
        # When the test says, by creating this file.
        while not os.path.exists(model + ".crash"):
            time.sleep(0.02)
        if behaviour.get("crash_message"):
            print("0.00.010.000 E " + behaviour["crash_message"], flush=True)
        os._exit(3)
    threading.Thread(target=crash, daemon=True).start()
print("0.00.000.004 I srv  server_on_li: listening on http://%s:%s" % (value("--host"), value("--port")), flush=True)
server.serve_forever()
'''


class FakeRuntimes:
    def __init__(self, runtime: LaunchableRuntime) -> None:
        self.runtime = runtime
        self.checks: list = []
        # What `--list-devices` says now. None: it couldn't be asked.
        self.devices: tuple[list[RuntimeDevice], dict[str, int]] | None = None

    def add_busy_check(self, check) -> None:
        self.checks.append(check)

    def launchable(self, runtime_id: str | None = None) -> LaunchableRuntime:
        if runtime_id not in (None, self.runtime.id):
            raise RuntimeNotFoundError(f"No installed runtime called {runtime_id}")
        return self.runtime

    def current_devices(self, runtime: LaunchableRuntime) -> tuple[list[RuntimeDevice], dict[str, int]] | None:
        return self.devices


class FakeHardware:
    def __init__(self) -> None:
        self.gpus: list[GpuInfo] = []
        self.memory: tuple[int, int] | None = None
        self.memory_asked: list[tuple[int, str]] = []

    def get_hardware(self) -> SimpleNamespace:
        return SimpleNamespace(gpus=self.gpus)

    def process_gpu_memory(self, pid: int, gpu_name: str) -> tuple[int, int] | None:
        self.memory_asked.append((pid, gpu_name))
        return self.memory


class FakeFitParams:
    """Stands in for llama-fit-params. `gate` holds a launch in its planning."""

    def __init__(self) -> None:
        self.answer: str | None = None
        self.calls: list[list[str]] = []
        self.called = threading.Event()
        self.gate: threading.Event | None = None

    def __call__(self, runtime: LaunchableRuntime, arguments: list[str]) -> str | None:
        self.calls.append(arguments)
        self.called.set()
        if self.gate is not None:
            self.gate.wait(10)
        return self.answer


class FakeLibrary:
    """Downloaded models as files whose content tells the fake server what to do."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.selections: dict[str, LocalSelection] = {}
        self.checks: list = []
        self.error: Exception | None = None

    def add_busy_check(self, check) -> None:
        self.checks.append(check)

    def add(self, repo_id: str = REPO, quantization: str = "Q4_K_M", vision: bool = False, **behaviour) -> Path:
        folder = self.folder / repo_id.replace("/", "--")
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"model-{quantization}.gguf"
        path.write_text(json.dumps(behaviour), encoding="utf-8")
        encoder = None
        if vision:
            (folder / "mmproj-F16.gguf").write_bytes(b"mmproj")
            encoder = LocalModelOption(id="F16", label="F16", status="complete", path=str(folder / "mmproj-F16.gguf"))
        selection = LocalSelection(
            repo_id=repo_id,
            model=LocalModelOption(id=quantization, label=quantization, status="complete", path=str(path)),
            vision=encoder,
        )
        # The last one added is what `auto` runs. The others are there by name.
        self.selections[repo_id] = self.selections[f"{repo_id}:{quantization}"] = selection
        return path

    def local_selection(self, repo_id: str, quantization: str = "auto", vision_encoder: str = "auto") -> LocalSelection:
        if self.error is not None:
            raise self.error
        selection = self.selections.get(f"{repo_id}:{quantization}") or self.selections.get(repo_id)
        if selection is None:
            raise LocalModelNotFoundError(f"{repo_id} isn't in the library")
        if vision_encoder == "none":
            return LocalSelection(repo_id=selection.repo_id, model=selection.model, vision=None)
        return selection


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def runtime(tmp_path: Path) -> LaunchableRuntime:
    directory = tmp_path / "runtimes" / RUNTIME_ID
    directory.mkdir(parents=True)
    return LaunchableRuntime(id=RUNTIME_ID, directory=directory, backend="rocm", gpu_device="ROCm1")


@pytest.fixture
def library(tmp_path: Path) -> FakeLibrary:
    return FakeLibrary(tmp_path / "model-files")


@pytest.fixture
def runtimes(runtime: LaunchableRuntime) -> FakeRuntimes:
    return FakeRuntimes(runtime)


@pytest.fixture
def hardware() -> FakeHardware:
    return FakeHardware()


@pytest.fixture
def fit_params() -> FakeFitParams:
    return FakeFitParams()


@pytest.fixture
def service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtimes: FakeRuntimes,
    library: FakeLibrary,
    hardware: FakeHardware,
    fit_params: FakeFitParams,
) -> Iterator[LlamaServerService]:
    monkeypatch.setattr(servers_module, "HEALTH_INTERVAL_SECONDS", 0.05)
    monkeypatch.setattr(servers_module, "GPU_MEMORY_DELAY_SECONDS", 0.05)
    # Well away from 8080, where real servers may be running.
    monkeypatch.setattr(servers_module, "FIRST_PORT", free_port())
    fake = tmp_path / "fake_llama_server.py"
    fake.write_text(FAKE_SERVER, encoding="utf-8")
    service = LlamaServerService(
        runtimes=runtimes,  # type: ignore[arg-type]
        library=library,  # type: ignore[arg-type]
        hardware=hardware,  # type: ignore[arg-type]
        logs_dir=tmp_path / "logs",
        command=lambda runtime: [PYTHON, str(fake)],
        fit_params=fit_params,
    )
    yield service
    service.stop_all()


def wait_until_settled(service: LlamaServerService, server: ModelServer, timeout: float = 20) -> ModelServer:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = service.get(server.id)
        if current.status not in ("starting", "stopping"):
            return current
        time.sleep(0.02)
    raise AssertionError(f"server didn't settle: {service.get(server.id)}")


def running(service: LlamaServerService, **launch) -> ModelServer:
    server = wait_until_settled(service, service.launch(launch.pop("repo_id", REPO), **launch))
    assert server.status == "ready", server.error
    return server


def launch_record(model_path: Path) -> dict:
    return json.loads(Path(f"{model_path}.launch.json").read_text(encoding="utf-8"))


def process_is_gone(pid: int, timeout: float = 5) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not psutil.pid_exists(pid):
            return True
        time.sleep(0.05)
    return False


def wait_while_ready(service: LlamaServerService, server: ModelServer, timeout: float = 10) -> ModelServer:
    deadline = time.monotonic() + timeout
    while service.get(server.id).status == "ready" and time.monotonic() < deadline:
        time.sleep(0.05)
    return service.get(server.id)


SELECTION = LocalSelection(
    repo_id=REPO,
    model=LocalModelOption(id="UD-Q4_K_M", label="UD-Q4_K_M", status="complete", path="C:/models/q4.gguf"),
    vision=None,
)
WHOLE = LaunchPlan(device="ROCm1", gpu=GpuPlan(fit_target_mib=4365, whole_model_on_gpu=True))


# The command


def test_a_model_that_fits_whole_is_read_in_and_fitted_to_what_is_really_free() -> None:
    arguments = server_arguments(SELECTION, ServerSettings(), 8080, WHOLE)
    assert arguments == [
        "--model", "C:/models/q4.gguf",
        "--host", "127.0.0.1",
        "--port", "8080",
        "--alias", f"{REPO}:UD-Q4_K_M",
        "--offline",
        "--log-colors", "off",
        "--cors-origins", "localhost",
        "--device", "ROCm1",
        "--fit-target", "4365",
        "--load-mode", "none", "--no-host",
    ]  # fmt: skip


@pytest.mark.parametrize("whole", [False, None])
def test_a_model_that_may_not_fit_whole_stays_memory_mapped(whole: bool | None) -> None:
    """Read in, the part left on the CPU would be pinned in RAM for good."""
    plan = LaunchPlan(device="ROCm1", gpu=GpuPlan(fit_target_mib=1024, whole_model_on_gpu=whole))
    arguments = server_arguments(SELECTION, ServerSettings(), 8080, plan)
    assert arguments[-4:] == ["--device", "ROCm1", "--fit-target", "1024"]


def test_settings_that_are_set_are_passed_on() -> None:
    plan = LaunchPlan(device="ROCm1", gpu=GpuPlan(fit_target_mib=1024))
    arguments = server_arguments(SELECTION, ServerSettings(context_size=8192, gpu_layers=20), 8080, plan)
    assert arguments[-8:] == [
        "--device", "ROCm1", "--fit-target", "1024", "--gpu-layers", "20", "--ctx-size", "8192"
    ]  # fmt: skip
    assert server_arguments(SELECTION, ServerSettings(gpu_layers="all"), 8080, WHOLE)[-2:] == ["--gpu-layers", "all"]


def test_no_gpu_layers_means_the_cpu_only_straight_from_the_mapping() -> None:
    arguments = server_arguments(SELECTION, ServerSettings(gpu_layers=0), 8080, LaunchPlan(device="none"))
    assert arguments[-2:] == ["--device", "none"]
    assert not {"--gpu-layers", "--fit-target", "--load-mode"} & set(arguments)


def test_a_cpu_build_is_given_no_device() -> None:
    arguments = server_arguments(SELECTION, ServerSettings(), 8080, LaunchPlan(device=None))
    assert not {"--device", "--fit-target", "--load-mode"} & set(arguments)


def test_a_vision_encoder_is_passed_with_the_model() -> None:
    encoder = LocalModelOption(id="F16", label="F16", status="complete", path="C:/models/mmproj-F16.gguf")
    selection = LocalSelection(repo_id=REPO, model=SELECTION.model, vision=encoder)
    arguments = server_arguments(selection, ServerSettings(), 8080, WHOLE)
    assert arguments[arguments.index("--mmproj") + 1] == "C:/models/mmproj-F16.gguf"


# Fitting it to the GPU


@pytest.mark.parametrize(
    ("reported", "actual", "vision", "expected"),
    [
        # ROCm on Windows: 2.8 GB used by other programs that it doesn't see.
        (16140 * MIB, 13312 * MIB, 0, 1024 + 2828),
        (16140 * MIB, 13312 * MIB, 1388 * MIB, 1024 + 2828 + 1388),
        # Drivers that count other programs already agree with the hardware.
        (8000 * MIB, 8000 * MIB, 0, 1024),
        # Never less than llama.cpp's own margin.
        (8000 * MIB, 9000 * MIB, 0, 1024),
        (None, 13312 * MIB, 0, 1024),
        (16140 * MIB, None, 0, 1024),
        (None, None, 1, 1025),  # Rounded up.
    ],
)
def test_the_fit_target_adds_what_llama_cpp_cant_see(
    reported: int | None, actual: int | None, vision: int, expected: int
) -> None:
    assert fit_target_mib(reported, actual, vision) == expected


@pytest.mark.parametrize(
    ("fitted", "whole"),
    [
        ("-c 174080 -ngl -1", True),
        ("-c 8192 -ngl -2", True),  # What `all` comes back as.
        ("-c 4096 -ngl 22", False),
        ("-c 4096 -ngl 0", False),
        # A mixture-of-experts model's experts moved to the CPU.
        ("-c 65536 -ngl -1 -ot blk\\.[0-9]+\\.ffn_.*_exps=CPU", False),
        ("-c 4096", None),
        ("-c 4096 -ngl", None),
        ("-c 4096 -ngl lots", None),
        ("", None),
    ],
)
def test_the_planners_answer_says_whether_the_model_fits_whole(fitted: str, whole: bool | None) -> None:
    assert whole_model_on_gpu(fitted) is whole


def test_the_planner_is_given_what_affects_the_fit() -> None:
    assert fit_arguments(SELECTION, ServerSettings(), "ROCm1", 2048) == [
        "--model", "C:/models/q4.gguf", "--device", "ROCm1", "--fit-target", "2048"
    ]  # fmt: skip
    settings = ServerSettings(context_size=65536, gpu_layers="all")
    assert fit_arguments(SELECTION, settings, "ROCm1", 2048)[-4:] == ["--ctx-size", "65536", "--gpu-layers", "all"]


# Reading its output


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        (
            [
                "0.00.000.001 I srv  load_model: loading model 'x.gguf'",
                "0.00.000.002 E gguf_init_from_file: failed to open GGUF file 'x.gguf' (No such file or directory)",
                "0.00.000.003 E llama_model_load_from_file_impl: failed to load model",
                "0.00.000.004 E srv  server_run_i: exiting due to model loading error",
            ],
            "gguf_init_from_file: failed to open GGUF file 'x.gguf' (No such file or directory)",
        ),
        (["I'm an older build", "error: unknown argument: --cors-origins"], "error: unknown argument: --cors-origins"),
        (["0.40.519.824 I slot      release: id  0 | task 0 | stop processing"], None),
        ([], None),
    ],
)
def test_the_reason_for_a_failure_is_the_first_error_logged(lines: list[str], expected: str | None) -> None:
    assert failure_reason(lines) == expected


def test_an_old_error_isnt_blamed_for_a_later_crash() -> None:
    lines = ["0.00.000.001 E srv  send_error: task 3: invalid request"] + ["0.00.000.002 I slot ..."] * 100
    assert failure_reason([*lines, "0.09.000.000 E ggml_abort: device lost"]) == "ggml_abort: device lost"


def test_the_log_tail_is_the_last_lines(tmp_path: Path) -> None:
    log = tmp_path / "server.log"
    log.write_bytes("".join(f"line {number}\n" for number in range(1000)).encode())
    assert read_log_tail(log, 3) == ["line 997", "line 998", "line 999"]
    assert read_log_tail(log, 0) == []
    assert read_log_tail(tmp_path / "missing.log", 10) == []


@pytest.mark.parametrize(
    ("tail_bytes", "first"), [(90, "line 990"), (91, "line 990"), (95, "line 990"), (99, "line 989")]
)
def test_a_line_cut_by_reading_only_the_end_is_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tail_bytes: int, first: str
) -> None:
    """Each line is 9 bytes, so 90 and 99 start on a line and 91 and 95 cut one."""
    log = tmp_path / "server.log"
    log.write_bytes("".join(f"line {number}\n" for number in range(100, 1000)).encode())
    monkeypatch.setattr(servers_module, "LOG_TAIL_BYTES", tail_bytes)
    tail = read_log_tail(log, 1000)
    assert tail[0] == first and tail[-1] == "line 999"


# Ports


def test_a_port_someone_listens_on_isnt_free() -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        assert not port_is_free(listener.getsockname()[1])


def test_a_port_listened_on_for_every_address_isnt_free_on_localhost() -> None:
    """Windows would let 127.0.0.1 bind to it: this is the app's own 0.0.0.0:8000."""
    with socket.socket() as listener:
        listener.bind(("0.0.0.0", 0))
        listener.listen()
        assert not port_is_free(listener.getsockname()[1])


def test_a_port_held_but_not_listened_on_isnt_free() -> None:
    with socket.socket() as holder:
        holder.bind(("127.0.0.1", 0))
        assert not port_is_free(holder.getsockname()[1])


def test_an_unused_port_is_free() -> None:
    assert port_is_free(free_port())


def test_a_port_listened_on_over_ipv6_and_ipv4_isnt_free() -> None:
    try:
        listener = socket.socket(socket.AF_INET6)
    except OSError:
        pytest.skip("No IPv6 here")
    with listener:
        listener.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        listener.bind(("::", 0))
        listener.listen()
        assert not port_is_free(listener.getsockname()[1])


def test_checking_a_port_is_instant() -> None:
    """It runs while launches wait on the service's lock. Connecting to a free
    port would make Windows wait out a timeout; binding doesn't."""
    port = free_port()
    started = time.perf_counter()
    assert port_is_free(port)
    assert time.perf_counter() - started < 0.1


# Running


def test_a_launch_loads_then_reports_what_it_loaded(service: LlamaServerService, library: FakeLibrary) -> None:
    model = library.add(load_seconds=0.3)
    started = service.launch(REPO)
    assert (started.status, started.loaded, started.ready_at) == ("starting", None, None)
    server = wait_until_settled(service, started)
    assert server.status == "ready"
    assert server.loaded is not None
    assert (server.loaded.context_size, server.loaded.slots, server.loaded.vision) == (262144, 4, False)
    assert server.model_name == f"{REPO}:Q4_K_M"
    assert server.base_url == f"http://127.0.0.1:{server.port}"
    assert (server.device, server.runtime_id, server.backend) == ("ROCm1", RUNTIME_ID, "rocm")
    assert server.model_path == str(model)
    assert server.ready_at is not None and server.error is None


def test_llama_server_runs_from_its_folder_without_the_apps_secrets(
    service: LlamaServerService, library: FakeLibrary, runtime: LaunchableRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in {
        "HF_TOKEN": "hf_secret",
        "GITHUB_TOKEN": "ghp_secret",
        "LLAMA_API_KEY": "key",
        "LLAMA_ARG_MODEL": "C:/somewhere/else.gguf",
        "LLAMA_ARG_HOST": "0.0.0.0",
    }.items():
        monkeypatch.setenv(name, value)
    model = library.add()
    running(service)
    record = launch_record(model)
    environment = {name.upper(): value for name, value in record["env"].items()}
    assert not {"HF_TOKEN", "GITHUB_TOKEN", "LLAMA_API_KEY"} & set(environment)
    assert not [name for name in environment if name.startswith("LLAMA_ARG_")]
    assert environment["PATH"].split(os.pathsep)[0] == str(runtime.directory)
    assert Path(record["cwd"]) == runtime.directory
    assert record["args"][record["args"].index("--host") + 1] == "127.0.0.1"


def test_the_log_starts_with_the_command(service: LlamaServerService, library: FakeLibrary) -> None:
    model = library.add()
    server = running(service)
    lines = service.logs(server.id).lines
    assert lines[0].startswith("> ") and str(model) in lines[0] and "--port" in lines[0]
    assert any("listening on" in line for line in lines)
    assert service.logs(server.id, lines=1).lines == lines[-1:]


def test_chosen_settings_are_used_and_reported(service: LlamaServerService, library: FakeLibrary) -> None:
    model = library.add(vision=True)
    server = running(service, context_size=8192, gpu_layers=20)
    assert server.loaded.context_size == 8192
    assert server.loaded.vision is True
    assert server.vision_encoder == "F16"
    assert server.settings == ServerSettings(context_size=8192, gpu_layers=20)
    arguments = launch_record(model)["args"]
    assert arguments[arguments.index("--gpu-layers") + 1] == "20"
    assert arguments[arguments.index("--mmproj") + 1] == server.vision_encoder_path


def test_launching_the_same_thing_again_returns_the_running_server(
    service: LlamaServerService, library: FakeLibrary
) -> None:
    library.add()
    server = running(service)
    assert service.launch(REPO).id == server.id
    assert service.launch(REPO, port=server.port).id == server.id
    assert len(service.all_servers()) == 1


@pytest.mark.parametrize(
    "change",
    [{"context_size": 4096}, {"gpu_layers": 0}, {"vision_encoder": "none"}, {"port": 1}],
)
def test_one_model_isnt_run_twice_with_different_settings(
    service: LlamaServerService, library: FakeLibrary, change: dict
) -> None:
    library.add(vision=True)
    server = running(service)
    with pytest.raises(ServerConflictError, match=f"already running with other settings, as server {server.id}"):
        service.launch(REPO, **change)


def test_different_models_run_side_by_side_on_their_own_ports(
    service: LlamaServerService, library: FakeLibrary
) -> None:
    library.add()
    library.add("ggml-org/tinygemma3-GGUF", "Q8_0")
    first = running(service)
    second = running(service, repo_id="ggml-org/tinygemma3-GGUF")
    assert first.port != second.port
    assert {server.status for server in service.all_servers()} == {"ready"}
    assert [server.id for server in service.all_servers()] == [second.id, first.id]


def test_a_server_still_starting_keeps_its_port(service: LlamaServerService, library: FakeLibrary) -> None:
    """It isn't listening yet, so only the app knows that port is taken."""
    library.add(listen_after=3)
    library.add("ggml-org/tinygemma3-GGUF", "Q8_0", listen_after=3)
    first = service.launch(REPO)
    second = service.launch("ggml-org/tinygemma3-GGUF")
    assert first.port != second.port


@pytest.mark.skipif(sys.platform != "win32", reason="Job objects are Windows only")
def test_a_server_is_tied_to_the_app(service: LlamaServerService, library: FakeLibrary) -> None:
    import ctypes
    from ctypes import wintypes

    library.add()
    server = running(service)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.IsProcessInJob.argtypes = (wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL))
    handle = kernel32.OpenProcess(0x1000, False, server.pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    try:
        in_job = wintypes.BOOL()
        assert kernel32.IsProcessInJob(handle, processes_module._job._handle, ctypes.byref(in_job))
        assert in_job.value
    finally:
        kernel32.CloseHandle(handle)


def test_a_port_in_use_is_refused_or_skipped(service: LlamaServerService, library: FakeLibrary) -> None:
    library.add()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", servers_module.FIRST_PORT))
        listener.listen()
        with pytest.raises(ServerConflictError, match=f"Port {servers_module.FIRST_PORT} is in use"):
            service.launch(REPO, port=servers_module.FIRST_PORT)
        assert running(service).port != servers_module.FIRST_PORT


def test_a_model_that_wont_load_says_why(service: LlamaServerService, library: FakeLibrary) -> None:
    library.add(fail="unknown model architecture: 'lfm3'")
    server = wait_until_settled(service, service.launch(REPO))
    assert server.status == "failed"
    assert server.exit_code == 1
    assert server.error == (
        "llama-server exited while loading the model (exit code 1): "
        "llama_model_load: error loading model: unknown model architecture: 'lfm3'"
    )


def crash(model: Path) -> None:
    Path(f"{model}.crash").touch()


def test_a_crash_after_loading_says_why(service: LlamaServerService, library: FakeLibrary) -> None:
    model = library.add(crashes=True, crash_message="ggml_abort: GPU device lost")
    server = running(service)
    crash(model)
    crashed = wait_while_ready(service, server)
    assert (crashed.status, crashed.exit_code) == ("failed", 3)
    assert crashed.error == "llama-server stopped unexpectedly (exit code 3): ggml_abort: GPU device lost"


def test_a_crash_with_nothing_logged_points_at_the_log(service: LlamaServerService, library: FakeLibrary) -> None:
    model = library.add(crashes=True)
    server = running(service)
    crash(model)
    crashed = wait_while_ready(service, server)
    assert crashed.error == "llama-server stopped unexpectedly (exit code 3). Its log may say why"


def test_a_crash_while_loading_is_told_apart(service: LlamaServerService, library: FakeLibrary) -> None:
    model = library.add(crashes=True, load_seconds=30, crash_message="ggml_cuda_init: failed")
    crash(model)
    crashed = wait_until_settled(service, service.launch(REPO))
    assert crashed.error == "llama-server exited while loading the model (exit code 3): ggml_cuda_init: failed"


def test_a_model_that_never_finishes_loading_is_given_up_on(
    service: LlamaServerService, library: FakeLibrary, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(servers_module, "LOAD_TIMEOUT_SECONDS", 1)
    library.add(hang=True)
    server = wait_until_settled(service, service.launch(REPO))
    assert server.status == "failed"
    assert "didn't finish loading within" in server.error
    assert process_is_gone(server.pid)


def test_stopping_ends_the_process(
    service: LlamaServerService, library: FakeLibrary, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Long, so a stop that only waits for the kill at the end of it is obvious.
    monkeypatch.setattr(servers_module, "STOP_TIMEOUT_SECONDS", 30)
    library.add()
    server = running(service)
    started = time.monotonic()
    stopped = service.stop(server.id)
    assert time.monotonic() - started < 10
    assert (stopped.status, stopped.error) == ("stopped", None)
    assert stopped.stopped_at is not None and stopped.exit_code is not None
    assert process_is_gone(server.pid)
    assert port_is_free(server.port)
    assert service.stop(server.id).status == "stopped"


def test_stopping_while_it_loads_isnt_a_failure(service: LlamaServerService, library: FakeLibrary) -> None:
    library.add(load_seconds=30)
    server = service.launch(REPO)
    time.sleep(0.3)
    stopped = service.stop(server.id)
    assert (stopped.status, stopped.error, stopped.ready_at) == ("stopped", None, None)


def test_a_stopped_model_can_be_run_again(service: LlamaServerService, library: FakeLibrary) -> None:
    library.add()
    first = running(service)
    service.stop(first.id)
    again = running(service, context_size=4096)
    assert again.id != first.id
    assert service.get(first.id).status == "stopped"


def test_shutting_down_stops_everything(
    service: LlamaServerService, library: FakeLibrary, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(servers_module, "STOP_TIMEOUT_SECONDS", 30)
    library.add()
    library.add("ggml-org/tinygemma3-GGUF", "Q8_0", load_seconds=30)
    ready = running(service)
    loading = service.launch("ggml-org/tinygemma3-GGUF")
    started = time.monotonic()
    service.stop_all()
    assert time.monotonic() - started < 10
    for server in (ready, loading):
        assert process_is_gone(server.pid)
        assert wait_until_settled(service, server).status == "stopped"


def test_a_running_model_keeps_its_files_and_build(
    service: LlamaServerService, library: FakeLibrary, runtimes: FakeRuntimes
) -> None:
    library.add(vision=True)
    (model_busy,) = library.checks
    (runtime_busy,) = runtimes.checks
    assert model_busy(REPO, None) is None and runtime_busy(RUNTIME_ID) is None
    server = running(service)
    busy = f"{REPO.upper()} is in use by server {server.id} ({REPO}:Q4_K_M). Stop it first."
    assert model_busy(REPO.upper(), None) == busy
    assert model_busy(REPO, {("language_model", "Q4_K_M")}) is not None
    assert model_busy(REPO, {("vision_encoder", "F16")}) is not None
    # Its encoder is F16, but a quantization called F16 is another file.
    assert model_busy(REPO, {("language_model", "F16")}) is None
    assert runtime_busy(RUNTIME_ID).startswith(f"{RUNTIME_ID} is in use by server {server.id}")
    assert model_busy("ggml-org/tinygemma3-GGUF", None) is None
    assert runtime_busy("llama-b1-bin-win-cpu-x64") is None
    service.stop(server.id)
    assert model_busy(REPO, None) is None and runtime_busy(RUNTIME_ID) is None


def test_files_a_running_model_doesnt_use_can_go(service: LlamaServerService, library: FakeLibrary) -> None:
    library.add(vision=True)
    (model_busy,) = library.checks
    running(service, vision_encoder="none")
    assert model_busy(REPO, {("language_model", "Q8_0")}) is None  # Another quantization.
    assert model_busy(REPO, {("vision_encoder", "F16")}) is None  # It runs text only.
    assert model_busy(REPO, {("language_model", "Q8_0"), ("language_model", "Q4_K_M")}) is not None


def test_old_logs_and_finished_servers_are_cleared_out(
    service: LlamaServerService, library: FakeLibrary, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(servers_module, "MAX_LOG_FILES", 2)
    monkeypatch.setattr(servers_module, "MAX_FINISHED_SERVERS", 1)
    service.logs_dir.mkdir(parents=True)
    for day in range(1, 6):
        (service.logs_dir / f"2026010{day}-120000-old{day}.log").write_text("old")
    library.add(fail="broken")
    first = wait_until_settled(service, service.launch(REPO))
    second = wait_until_settled(service, service.launch(REPO))
    third = wait_until_settled(service, service.launch(REPO))
    assert [server.id for server in service.all_servers()] == [third.id, second.id]
    with pytest.raises(ServerNotFoundError):
        service.get(first.id)
    names = sorted(path.name for path in service.logs_dir.glob("*.log"))
    assert [name for name in names if "old" in name] == []  # Newer logs of ours outnumbered them.
    assert Path(third.log_path).name in names


def test_errors_from_the_library_and_runtimes_come_through(
    service: LlamaServerService, library: FakeLibrary
) -> None:
    with pytest.raises(LocalModelNotFoundError):
        service.launch(REPO)
    library.add()
    with pytest.raises(RuntimeNotFoundError):
        service.launch(REPO, runtime_id="llama-b1-bin-win-cpu-x64")
    library.error = LocalModelBusyError("still downloading")
    with pytest.raises(LocalModelBusyError):
        service.launch(REPO)
    assert service.all_servers() == []


# Planning the launch


def on_a_9070(runtimes: FakeRuntimes, hardware: FakeHardware) -> None:
    runtimes.devices = ROCM_DEVICES
    hardware.gpus = [RX_9070, IGPU]


def test_the_plan_leaves_room_for_other_programs_and_the_vision_encoder(
    service: LlamaServerService,
    library: FakeLibrary,
    runtimes: FakeRuntimes,
    hardware: FakeHardware,
    fit_params: FakeFitParams,
) -> None:
    on_a_9070(runtimes, hardware)
    fit_params.answer = "-c 174080 -ngl -1"
    model = library.add(vision=True)
    server = running(service)
    encoder = len(b"mmproj") + servers_module.VISION_OVERHEAD_BYTES
    target = 1024 + math.ceil((16140 * MIB - 13 * GIB + encoder) / MIB)
    assert server.gpu_plan == GpuPlan(vram_free_bytes=13 * GIB, fit_target_mib=target, whole_model_on_gpu=True)
    assert fit_params.calls == [["--model", str(model), "--device", "ROCm1", "--fit-target", str(target)]]
    arguments = launch_record(model)["args"]
    assert arguments[arguments.index("--fit-target") + 1] == str(target)
    assert arguments[arguments.index("--load-mode") :][:3] == ["--load-mode", "none", "--no-host"]


def test_a_model_that_doesnt_fit_whole_stays_memory_mapped(
    service: LlamaServerService,
    library: FakeLibrary,
    runtimes: FakeRuntimes,
    hardware: FakeHardware,
    fit_params: FakeFitParams,
) -> None:
    on_a_9070(runtimes, hardware)
    fit_params.answer = "-c 65536 -ngl 14"
    model = library.add()
    server = running(service, context_size=65536)
    assert server.gpu_plan.whole_model_on_gpu is False
    assert fit_params.calls[0][-2:] == ["--ctx-size", "65536"]
    assert "--load-mode" not in launch_record(model)["args"]


def test_without_the_planner_it_stays_memory_mapped(
    service: LlamaServerService, library: FakeLibrary, runtimes: FakeRuntimes, hardware: FakeHardware
) -> None:
    on_a_9070(runtimes, hardware)
    model = library.add()  # The fake planner has no answer, as with a build that lacks it.
    assert running(service).gpu_plan.whole_model_on_gpu is None
    assert "--load-mode" not in launch_record(model)["args"]


def test_a_number_of_gpu_layers_isnt_planned_but_is_still_fitted(
    service: LlamaServerService,
    library: FakeLibrary,
    runtimes: FakeRuntimes,
    hardware: FakeHardware,
    fit_params: FakeFitParams,
) -> None:
    on_a_9070(runtimes, hardware)
    model = library.add()
    server = running(service, gpu_layers=20)
    assert fit_params.calls == []
    assert server.gpu_plan.whole_model_on_gpu is None
    assert "--fit-target" in launch_record(model)["args"]


def test_the_cpu_only_needs_no_plan(
    service: LlamaServerService,
    library: FakeLibrary,
    runtimes: FakeRuntimes,
    hardware: FakeHardware,
    fit_params: FakeFitParams,
) -> None:
    on_a_9070(runtimes, hardware)
    library.add()
    server = running(service, gpu_layers=0)
    assert (server.device, server.gpu_plan) == ("none", None)
    assert fit_params.calls == []


def test_the_gpu_is_found_by_name_when_its_number_changes(
    service: LlamaServerService, library: FakeLibrary, runtimes: FakeRuntimes, hardware: FakeHardware
) -> None:
    """Disabling the integrated GPU makes the 9070 XT ROCm0 instead of ROCm1."""
    runtimes.devices = ([RuntimeDevice(id="ROCm0", name="AMD Radeon RX 9070 XT")], {"ROCm0": 16140 * MIB})
    hardware.gpus = [RX_9070]
    model = library.add()
    assert running(service).device == "ROCm0"
    arguments = launch_record(model)["args"]
    assert arguments[arguments.index("--device") + 1] == "ROCm0"


def test_of_two_identical_cards_the_fuller_one_counts(
    service: LlamaServerService,
    library: FakeLibrary,
    runtimes: FakeRuntimes,
    hardware: FakeHardware,
) -> None:
    """Their names can't say which is which, so it plans for the worse case."""
    runtimes.devices = ROCM_DEVICES
    hardware.gpus = [RX_9070, RX_9070.model_copy(update={"vram_free_bytes": 6 * GIB}), IGPU]
    library.add()
    assert running(service).gpu_plan.vram_free_bytes == 6 * GIB


def test_a_gpu_the_build_cant_see_is_refused(
    service: LlamaServerService, library: FakeLibrary, runtimes: FakeRuntimes
) -> None:
    library.add()
    runtimes.devices = ([], {})
    with pytest.raises(RuntimeConflictError, match="can't see a GPU right now"):
        service.launch(REPO)
    # Something else in its place, and no way to tell which card is which.
    runtimes.devices = ([RuntimeDevice(id="ROCm0", name="Some Other Card")], {})
    with pytest.raises(RuntimeConflictError, match="no longer has the device ROCm1"):
        service.launch(REPO)
    (model_busy,) = library.checks
    assert service.all_servers() == [] and model_busy(REPO, None) is None


def wait_for_gpu_memory(service: LlamaServerService, server: ModelServer, timeout: float = 10) -> ModelServer:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = service.get(server.id)
        if current.loaded is not None and current.loaded.gpu_memory is not None:
            return current
        time.sleep(0.02)
    raise AssertionError("its GPU memory was never read")


def test_gpu_memory_that_ended_up_in_system_ram_is_reported(
    service: LlamaServerService, library: FakeLibrary, runtimes: FakeRuntimes, hardware: FakeHardware
) -> None:
    on_a_9070(runtimes, hardware)
    hardware.memory = (7_800_000_000, 2_400_000_000)
    library.add()
    server = wait_for_gpu_memory(service, running(service))
    assert server.loaded.gpu_memory == GpuMemory(vram_bytes=7_800_000_000, system_ram_bytes=2_400_000_000)
    assert len(server.warnings) == 1
    assert server.warnings[0].startswith("2.4 GB of its GPU memory is in system RAM rather than VRAM")
    assert hardware.memory_asked == [(server.pid, "AMD Radeon RX 9070 XT")]


def test_a_model_that_fits_in_vram_has_nothing_to_warn_about(
    service: LlamaServerService, library: FakeLibrary, runtimes: FakeRuntimes, hardware: FakeHardware
) -> None:
    on_a_9070(runtimes, hardware)
    hardware.memory = (9_800_000_000, 310_000_000)  # Buffers the GPU reads from RAM by design.
    library.add()
    server = wait_for_gpu_memory(service, running(service))
    assert server.loaded.gpu_memory.system_ram_bytes == 310_000_000
    assert server.warnings == []


def test_gpu_memory_isnt_read_without_knowing_which_gpu(
    service: LlamaServerService, library: FakeLibrary, hardware: FakeHardware
) -> None:
    hardware.memory = (1, 2)
    library.add()
    server = running(service)
    time.sleep(0.3)
    assert hardware.memory_asked == []
    assert service.get(server.id).loaded.gpu_memory is None


def test_a_launch_under_way_keeps_its_files_and_build(
    service: LlamaServerService,
    library: FakeLibrary,
    runtimes: FakeRuntimes,
    hardware: FakeHardware,
    fit_params: FakeFitParams,
) -> None:
    """Planning takes a second or two, and a delete then mustn't pull the
    files out from under it."""
    on_a_9070(runtimes, hardware)
    fit_params.gate = threading.Event()
    library.add()
    (model_busy,) = library.checks
    (runtime_busy,) = runtimes.checks
    launched: list[ModelServer] = []
    thread = threading.Thread(target=lambda: launched.append(service.launch(REPO)))
    thread.start()
    try:
        assert fit_params.called.wait(10)
        name = f"{REPO}:Q4_K_M"
        assert model_busy(REPO, None) == f"{REPO} is in use by {name}, which is starting. Stop it first, once it has."
        assert model_busy(REPO, {("language_model", "Q8_0")}) is None
        assert runtime_busy(RUNTIME_ID).startswith(f"{RUNTIME_ID} is in use by {name}")
        with pytest.raises(ServerConflictError, match="is being launched already"):
            service.launch(REPO)
    finally:
        fit_params.gate.set()
        thread.join(10)
    server = wait_until_settled(service, launched[0])
    assert server.status == "ready"
    assert model_busy(REPO, None).startswith(f"{REPO} is in use by server {server.id}")


def test_files_deleted_just_before_the_launch_are_noticed(service: LlamaServerService, library: FakeLibrary) -> None:
    library.add().unlink()
    with pytest.raises(LocalModelNotFoundError, match="Q4_K_M has just been deleted"):
        service.launch(REPO)
    (model_busy,) = library.checks
    assert model_busy(REPO, None) is None


def test_a_build_deleted_just_before_the_launch_is_noticed(
    service: LlamaServerService, library: FakeLibrary, tmp_path: Path
) -> None:
    library.add()
    service._command = lambda runtime: [str(tmp_path / "gone" / "llama-server.exe")]
    with pytest.raises(RuntimeNotFoundError, match="has just been deleted"):
        service.launch(REPO)


def test_a_port_in_use_is_refused_before_the_planning(
    service: LlamaServerService, library: FakeLibrary, fit_params: FakeFitParams
) -> None:
    library.add()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        with pytest.raises(ServerConflictError, match="is in use"):
            service.launch(REPO, port=listener.getsockname()[1])
    assert fit_params.calls == []


@pytest.mark.skipif(sys.platform != "win32", reason="Job objects are Windows only")
def test_the_windows_job_ends_its_processes_once_the_app_lets_go() -> None:
    import ctypes

    job = processes_module._WindowsJob()
    child = subprocess.Popen(
        [PYTHON, "-c", "import time; time.sleep(60)"], creationflags=subprocess.CREATE_NO_WINDOW
    )
    try:
        job.add(child)
        assert child.poll() is None
        ctypes.WinDLL("kernel32").CloseHandle(job._handle)  # What the app dying does.
        assert child.wait(5) is not None
    finally:
        if child.poll() is None:
            child.kill()


# The routes


@pytest.fixture
def client(service: LlamaServerService, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(server_router, "service", service)
    return TestClient(app)


def wait_for_route(client: TestClient, server_id: str) -> dict:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        body = client.get(f"/servers/{server_id}").json()
        if body["status"] != "starting":
            return body
        time.sleep(0.02)
    raise AssertionError("server didn't settle")


def test_the_routes_launch_watch_and_stop(client: TestClient, library: FakeLibrary) -> None:
    library.add()
    response = client.post("/servers", json={"repo_id": REPO, "context_size": 4096})
    assert response.status_code == 202
    assert response.json()["status"] == "starting"
    body = wait_for_route(client, response.json()["id"])
    assert body["status"] == "ready"
    assert body["loaded"]["context_size"] == 4096
    assert [server["id"] for server in client.get("/servers").json()] == [body["id"]]
    logs = client.get(f"/servers/{body['id']}/logs", params={"lines": 2}).json()
    assert len(logs["lines"]) == 2 and logs["path"] == body["log_path"]
    stopped = client.delete(f"/servers/{body['id']}")
    assert (stopped.status_code, stopped.json()["status"]) == (200, "stopped")


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (LocalModelNotFoundError("not in the library"), 404),
        (LocalModelChoiceError("pick one"), 422),
        (LocalModelBusyError("still downloading"), 409),
        (RuntimeNotFoundError("no runtime is active"), 404),
        (RuntimeConflictError("didn't pass its test"), 409),
    ],
)
def test_the_routes_map_errors(client: TestClient, library: FakeLibrary, error: Exception, status: int) -> None:
    library.error = error
    response = client.post("/servers", json={"repo_id": REPO})
    assert (response.status_code, response.json()["detail"]) == (status, str(error))


def test_a_port_clash_is_a_conflict(client: TestClient, library: FakeLibrary) -> None:
    library.add()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        response = client.post("/servers", json={"repo_id": REPO, "port": port})
    assert (response.status_code, response.json()["detail"]) == (409, f"Port {port} is in use.")


@pytest.mark.parametrize(
    "body",
    [
        {"repo_id": "../outside"},
        {"repo_id": REPO, "quantization": "Q4 K M"},
        {"repo_id": REPO, "gpu_layers": "most"},
        {"repo_id": REPO, "gpu_layers": -1},
        {"repo_id": REPO, "context_size": 100},
        {"repo_id": REPO, "port": 80},
        {"repo_id": REPO, "runtime_id": "../llama"},
    ],
)
def test_the_routes_refuse_bad_requests(client: TestClient, body: dict) -> None:
    assert client.post("/servers", json=body).status_code == 422


def test_the_routes_for_a_missing_server(client: TestClient) -> None:
    assert client.get("/servers/nope").status_code == 404
    assert client.get("/servers/nope/logs").status_code == 404
    assert client.delete("/servers/nope").status_code == 404


def test_the_app_stops_its_servers_when_it_shuts_down(
    service: LlamaServerService, library: FakeLibrary, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main, "llama_server_service", service)
    monkeypatch.setattr(main, "inference_proxy", local_proxy())
    monkeypatch.setattr(main.hardware_service, "preload", lambda: None)
    library.add()
    with TestClient(app):
        server = running(service)
    assert service.get(server.id).status == "stopped"
    assert process_is_gone(server.pid)


# Finding the server a request is for


def test_a_model_is_found_by_its_name_whatever_the_case(service: LlamaServerService, library: FakeLibrary) -> None:
    library.add()
    library.add("ggml-org/tinygemma3-GGUF", "Q8_0")
    first = running(service)
    running(service, repo_id="ggml-org/tinygemma3-GGUF")
    assert service.find_running(first.model_name.upper()).id == first.id


def test_a_repo_alone_finds_its_only_running_quantization(service: LlamaServerService, library: FakeLibrary) -> None:
    library.add()
    server = running(service)
    assert service.find_running(REPO.lower()).id == server.id
    library.add(quantization="Q8_0")
    other = running(service, quantization="Q8_0")
    with pytest.raises(ModelNotRunningError, match=f"Several running models match {REPO}") as error:
        service.find_running(REPO)
    assert server.model_name in str(error.value) and other.model_name in str(error.value)
    assert service.find_running(other.model_name).id == other.id


def test_without_a_model_the_only_one_running_is_used(service: LlamaServerService, library: FakeLibrary) -> None:
    library.add()
    library.add("ggml-org/tinygemma3-GGUF", "Q8_0")
    with pytest.raises(ModelNotRunningError, match="No model is running"):
        service.find_running(None)
    server = running(service)
    assert service.find_running(None).id == server.id
    running(service, repo_id="ggml-org/tinygemma3-GGUF")
    with pytest.raises(ModelNotRunningError, match="Several models are running"):
        service.find_running(None)


def test_a_model_not_running_names_those_that_are(service: LlamaServerService, library: FakeLibrary) -> None:
    library.add()
    server = running(service)
    with pytest.raises(ModelNotRunningError, match=f"other/model-GGUF isn't running. Running: {server.model_name}"):
        service.find_running("other/model-GGUF")
    service.stop(server.id)
    with pytest.raises(ModelNotRunningError, match="No model is running"):
        service.find_running(server.model_name)


def test_a_model_still_loading_is_found_but_not_ready(service: LlamaServerService, library: FakeLibrary) -> None:
    library.add(hang=True)
    server = service.launch(REPO)
    with pytest.raises(ModelLoadingError, match=f"{server.model_name} is still loading"):
        service.find_running(server.model_name)


# The OpenAI-compatible API


def local_proxy() -> InferenceProxy:
    """Without keep-alive: each test request runs in an event loop of its own,
    and a connection can't be used from another."""
    return InferenceProxy(httpx.AsyncClient(trust_env=False, limits=httpx.Limits(max_keepalive_connections=0)))


@pytest.fixture
def api(service: LlamaServerService, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(server_router, "service", service)
    monkeypatch.setattr(inference_router, "servers", service)
    monkeypatch.setattr(inference_router, "proxy", local_proxy())
    monkeypatch.delenv("LOCALAI_API_KEY", raising=False)
    return TestClient(app)


def chat(model: str | None = None, **extra) -> dict:
    body: dict = {"messages": [{"role": "user", "content": "Hello"}], **extra}
    if model is not None:
        body["model"] = model
    return body


def last_request(model_path: Path) -> dict:
    lines = Path(f"{model_path}.requests.jsonl").read_text(encoding="utf-8").splitlines()
    return json.loads(lines[-1])


def test_running_models_are_served_at_one_address(
    api: TestClient, service: LlamaServerService, library: FakeLibrary
) -> None:
    library.add(reply="Hi from Qwen")
    library.add("ggml-org/tinygemma3-GGUF", "Q8_0", reply="Hi from Gemma")
    first = running(service)
    second = running(service, repo_id="ggml-org/tinygemma3-GGUF")
    models = api.get("/v1/models").json()
    assert models["object"] == "list"
    assert [model["id"] for model in models["data"]] == [second.model_name, first.model_name]
    for server, reply in ((first, "Hi from Qwen"), (second, "Hi from Gemma")):
        response = api.post("/v1/chat/completions", json=chat(server.model_name))
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == reply


def test_only_ready_models_are_listed(api: TestClient, service: LlamaServerService, library: FakeLibrary) -> None:
    library.add(hang=True)
    service.launch(REPO)
    assert api.get("/v1/models").json() == {"object": "list", "data": []}


def test_a_streamed_reply_streams(api: TestClient, service: LlamaServerService, library: FakeLibrary) -> None:
    library.add(reply="one two three")
    server = running(service)
    with api.stream("POST", "/v1/chat/completions", json=chat(server.model_name, stream=True)) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["cache-control"] == "no-cache"
        events = [line for line in response.iter_lines() if line]
    assert events[-1] == "data: [DONE]"
    words = [json.loads(event.removeprefix("data: "))["choices"][0]["delta"]["content"] for event in events[:-1]]
    assert "".join(words) == "one two three "


def test_completions_are_served_too(api: TestClient, service: LlamaServerService, library: FakeLibrary) -> None:
    model = library.add()
    server = running(service)
    response = api.post("/v1/completions", json={"model": server.model_name, "prompt": "Once"})
    assert response.status_code == 200
    assert last_request(model)["path"] == "/v1/completions"


def test_the_request_goes_on_as_it_is_without_the_apps_key(
    api: TestClient, service: LlamaServerService, library: FakeLibrary, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCALAI_API_KEY", "app-secret")
    model = library.add()
    server = running(service)
    body = json.dumps(chat(server.model_name, temperature=0.25, something_new={"a": [1, 2]}), indent=1)
    response = api.post(
        "/v1/chat/completions",
        content=body,
        headers={"Authorization": "Bearer app-secret", "Content-Type": "application/json", "Cookie": "session=1"},
    )
    assert response.status_code == 200
    request = last_request(model)
    assert (request["path"], request["body"]) == ("/v1/chat/completions", body)
    headers = {name.lower(): value for name, value in request["headers"].items()}
    assert headers["content-type"] == "application/json"
    assert "authorization" not in headers and "cookie" not in headers


def test_llama_servers_own_errors_come_back_as_they_are(
    api: TestClient, service: LlamaServerService, library: FakeLibrary
) -> None:
    library.add(reply_error="the request exceeds the available context size")
    server = running(service)
    response = api.post("/v1/chat/completions", json=chat(server.model_name))
    assert response.status_code == 400
    assert response.json()["error"]["message"] == "the request exceeds the available context size"


def test_a_model_that_isnt_running_is_404_in_openais_format(
    api: TestClient, service: LlamaServerService, library: FakeLibrary
) -> None:
    library.add()
    server = running(service)
    response = api.post("/v1/chat/completions", json=chat("other/model-GGUF:Q4_K_M"))
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "model_not_found" and error["type"] == "invalid_request_error"
    assert server.model_name in error["message"]


def test_a_model_still_loading_is_503_with_when_to_retry(
    api: TestClient, service: LlamaServerService, library: FakeLibrary
) -> None:
    library.add(hang=True)
    server = service.launch(REPO)
    response = api.post("/v1/chat/completions", json=chat(server.model_name))
    assert response.status_code == 503
    assert response.headers["retry-after"] == "5"
    assert response.json()["error"]["code"] == "model_loading"


def test_a_server_that_has_just_died_is_502(
    api: TestClient, service: LlamaServerService, library: FakeLibrary, monkeypatch: pytest.MonkeyPatch
) -> None:
    library.add()
    server = running(service)
    # As if it went between being found and being asked.
    gone = server.model_copy(update={"base_url": f"http://127.0.0.1:{free_port()}"})
    monkeypatch.setattr(service, "find_running", lambda model: gone)
    response = api.post("/v1/chat/completions", json=chat(server.model_name))
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "model_unavailable"


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"not json", "The body isn't valid JSON"),
        (b"[1, 2]", "The body must be a JSON object."),
        (b'{"model": 3}', "`model` must be a string."),
    ],
)
def test_a_body_that_isnt_a_request_is_400(api: TestClient, content: bytes, message: str) -> None:
    response = api.post("/v1/chat/completions", content=content, headers={"Content-Type": "application/json"})
    assert response.status_code == 400
    assert response.json()["error"]["message"].startswith(message)


def test_the_api_key_guards_v1_only(api: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    assert api.get("/v1/models").status_code == 200
    monkeypatch.setenv("LOCALAI_API_KEY", "app-secret")
    for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": "Basic app-secret"}):
        response = api.get("/v1/models", headers=headers)
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "invalid_api_key"
    assert api.post("/v1/chat/completions", json=chat()).status_code == 401
    assert api.get("/v1/models", headers={"Authorization": "Bearer app-secret"}).status_code == 200
    assert api.get("/servers").status_code == 200


def test_a_client_that_leaves_stops_the_reply(
    service: LlamaServerService, library: FakeLibrary, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Over a real connection, since only a real server sees the client go.
    llama-server stops generating when the connection to it closes."""
    monkeypatch.setattr(inference_router, "servers", service)
    monkeypatch.setattr(inference_router, "proxy", local_proxy())
    monkeypatch.delenv("LOCALAI_API_KEY", raising=False)
    model = library.add(reply=" ".join(["word"] * 200), word_seconds=0.05)
    server = running(service)
    port = free_port()
    app_server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, lifespan="off", log_level="warning"))
    thread = threading.Thread(target=app_server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not app_server.started and time.monotonic() < deadline:
            time.sleep(0.02)
        url = f"http://127.0.0.1:{port}/v1/chat/completions"
        with httpx.Client(trust_env=False) as client:
            with client.stream("POST", url, json=chat(server.model_name, stream=True)) as response:
                assert next(response.iter_lines()).startswith("data: ")
        # The whole reply takes 10 seconds. It should end long before.
        cancelled = Path(f"{model}.cancelled")
        deadline = time.monotonic() + 5
        while not cancelled.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert cancelled.exists()
    finally:
        app_server.should_exit = True
        thread.join(10)

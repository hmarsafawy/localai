"""The source archive here copies the layout of llama.cpp's real one at b12359:
everything under `llama.cpp-<tag>/`, with the commit in the archive's pax
header. pip and the converter itself are faked, since the real ones would
download PyTorch."""

from __future__ import annotations

import io
import sys
import tarfile
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

import routes.runtime_router as runtime_router
import services.llama_converter as converter_module
from main import app
from models.jobs import JobInfo
from services.jobs import JobContext, JobStore
from services.llama_converter import (
    ENVIRONMENTS_DIR,
    REQUIREMENTS_FILE,
    SOURCES_DIR,
    ConverterSpaceError,
    LlamaConverterService,
    converter_environment,
    environment_key,
    environment_python,
    extract_converter,
    parse_supported_models,
    pip_current_file,
    run_command,
)
from services.llama_runtime import (
    LaunchableRuntime,
    RuntimeInstallError,
    RuntimeNotFoundError,
    UnsafeArchiveError,
)

COMMIT = "b1b770d90f00ba7e1c2d3e4f5a6b7c8d9e0f1a2b"
CONVERTER_REQUIREMENTS = b"-r ./requirements-convert_legacy_llama.txt\n--extra-index-url https://download.pytorch.org/whl/cpu\ntorch==2.11.0\n"
LEGACY_REQUIREMENTS = b"numpy~=2.2.6\nsentencepiece>=0.1.98,<0.3.0\ntransformers==4.57.6\ngguf>=0.1.0\n"
SUPPORTED_OUTPUT = [
    "ERROR:hf-to-gguf:TEXT models:",
    "ERROR:hf-to-gguf:  - LlamaForCausalLM",
    "ERROR:hf-to-gguf:  - Qwen3ForCausalLM",
    "ERROR:hf-to-gguf:MMPROJ models:",
    "ERROR:hf-to-gguf:  - Qwen2VLForConditionalGeneration",
]


def source_archive(tag: str, legacy: bytes = LEGACY_REQUIREMENTS, extra: dict[str, bytes] | None = None) -> bytes:
    files = {
        "convert_hf_to_gguf.py": b"import torch\n",
        "convert_hf_to_gguf_update.py": b"# not needed\n",
        "conversion/__init__.py": b"",
        "conversion/llama.py": b"",
        "gguf-py/gguf/__init__.py": b"",
        "requirements/requirements-convert_hf_to_gguf.txt": CONVERTER_REQUIREMENTS,
        "requirements/requirements-convert_legacy_llama.txt": legacy,
        "requirements/requirements-server-bench.txt": b"requests\n",
        "models/templates/Kimi-K3.jinja": b"{{ messages }}",
        "models/ggml-vocab-llama.gguf": b"vocab",
        "src/llama.cpp": b"int main() {}",
        "README.md": b"# llama.cpp",
        **(extra or {}),
    }
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz", format=tarfile.PAX_FORMAT, pax_headers={"comment": COMMIT}) as bundle:
        for name, content in files.items():
            info = tarfile.TarInfo(f"llama.cpp-{tag}/{name}")
            info.size = len(content)
            bundle.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


class FakeGitHub:
    def __init__(self) -> None:
        self.archives: dict[str, bytes] = {}
        self.requests: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(str(request.url))
        tag = request.url.path.rsplit("/", 1)[-1].removesuffix(".tar.gz")
        if tag not in self.archives:
            return httpx.Response(404)
        return httpx.Response(200, content=self.archives[tag])


class FakeRuntimes:
    def __init__(self, client: httpx.Client, installed: list[str]) -> None:
        self.client = client
        self.jobs = JobStore()
        self.installed = installed
        self.active: str | None = installed[0] if installed else None

    def active_id(self) -> str | None:
        return self.active

    def launchable(self, runtime_id: str | None = None) -> LaunchableRuntime:
        runtime_id = runtime_id or self.active
        if runtime_id is None or runtime_id not in self.installed:
            raise RuntimeNotFoundError(f"No installed runtime called {runtime_id}")
        return LaunchableRuntime(id=runtime_id, directory=Path("."), backend="vulkan", gpu_device=None)


class FakeCommands:
    """Stands in for `run_command`: makes the environment `venv` would, and
    prints what pip and the converter would."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.envs: list[dict[str, str] | None] = []
        self.results: dict[str, tuple[int, list[str]]] = {}
        self.hold: threading.Event | None = None

    def kind(self, command: list[str]) -> str:
        if command[1:3] == ["-m", "venv"]:
            return "venv"
        if command[1:4] == ["-m", "pip", "install"]:
            return "pip"
        if "--print-supported-models" in command:
            return "check"
        raise AssertionError(f"unexpected command {command}")

    def __call__(
        self,
        command: list[str],
        context: JobContext,
        *,
        timeout: float,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        on_line: Callable[[str], None] | None = None,
    ) -> tuple[int, list[str]]:
        kind = self.kind(command)
        self.calls.append(command)
        self.envs.append(env)
        if kind == "venv":
            python = environment_python(Path(command[3]))
            python.parent.mkdir(parents=True)
            python.write_bytes(b"")
        if kind == "pip" and self.hold is not None:
            context.wait(10)  # Until cancelled.
        code, lines = self.results.get(kind, (0, SUPPORTED_OUTPUT if kind == "check" else ["Collecting torch==2.11.0"]))
        for line in lines:
            if on_line:
                on_line(line)
        return code, lines[-20:]

    def kinds(self) -> list[str]:
        return [self.kind(command) for command in self.calls]


@pytest.fixture
def github() -> FakeGitHub:
    fake = FakeGitHub()
    fake.archives["b100"] = source_archive("b100")
    fake.archives["b101"] = source_archive("b101")
    return fake


@pytest.fixture
def commands(monkeypatch: pytest.MonkeyPatch) -> FakeCommands:
    fake = FakeCommands()
    monkeypatch.setattr(converter_module, "run_command", fake)
    return fake


@pytest.fixture
def runtimes(github: FakeGitHub) -> FakeRuntimes:
    client = httpx.Client(transport=httpx.MockTransport(github))
    return FakeRuntimes(client, ["llama-b100-bin-win-vulkan-x64", "llama-b101-bin-win-vulkan-x64"])


@pytest.fixture
def service(tmp_path: Path, runtimes: FakeRuntimes, commands: FakeCommands) -> Iterator[LlamaConverterService]:
    service = LlamaConverterService(runtimes=runtimes, root=tmp_path / "converter")  # type: ignore[arg-type]
    yield service
    # A job a test didn't wait for would run on into the next test, and use
    # that test's fakes: they replace the module's, which it looks up as it goes.
    for job in service.jobs.all_jobs():
        service.jobs.cancel(job.id)
        wait_for(service, job)


def wait_for(service: LlamaConverterService, job: JobInfo, timeout: float = 10) -> JobInfo:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = service.jobs.get(job.id)
        assert current is not None
        if current.status not in ("queued", "running"):
            return current
        time.sleep(0.01)
    raise AssertionError(f"job didn't finish: {service.jobs.get(job.id)}")


def install(service: LlamaConverterService, runtime_id: str | None = None) -> JobInfo:
    job = wait_for(service, service.start_install(runtime_id))
    assert job.status == "completed", job.error
    return job


# The source archive


def test_only_what_the_converter_reads_is_extracted(tmp_path: Path) -> None:
    archive = tmp_path / "source.tar.gz"
    archive.write_bytes(source_archive("b100"))
    destination = tmp_path / "out"
    assert extract_converter(archive, destination) == COMMIT
    kept = sorted(path.relative_to(destination).as_posix() for path in destination.rglob("*") if path.is_file())
    assert kept == [
        "conversion/__init__.py",
        "conversion/llama.py",
        "convert_hf_to_gguf.py",
        "gguf-py/gguf/__init__.py",
        "models/templates/Kimi-K3.jinja",
        "requirements/requirements-convert_hf_to_gguf.txt",
        "requirements/requirements-convert_legacy_llama.txt",
        "requirements/requirements-server-bench.txt",
    ]


def test_an_entry_that_escapes_is_refused(tmp_path: Path) -> None:
    archive = tmp_path / "source.tar.gz"
    archive.write_bytes(source_archive("b100", extra={"conversion/../../escaped.py": b"boom"}))
    with pytest.raises(UnsafeArchiveError):
        extract_converter(archive, tmp_path / "out")
    assert not (tmp_path / "escaped.py").exists()


def test_a_source_without_the_converter_is_refused(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        info = tarfile.TarInfo("llama.cpp-b1/README.md")
        bundle.addfile(info, io.BytesIO(b""))
    archive = tmp_path / "source.tar.gz"
    archive.write_bytes(buffer.getvalue())
    with pytest.raises(RuntimeInstallError, match="convert_hf_to_gguf.py"):
        extract_converter(archive, tmp_path / "out")


def test_a_corrupt_archive_fails_clearly(tmp_path: Path) -> None:
    archive = tmp_path / "source.tar.gz"
    archive.write_bytes(source_archive("b100")[:500])
    with pytest.raises(RuntimeInstallError, match="isn't a readable archive"):
        extract_converter(archive, tmp_path / "out")


def unpacked(tmp_path: Path, name: str, legacy: bytes = LEGACY_REQUIREMENTS) -> Path:
    archive = tmp_path / f"{name}.tar.gz"
    archive.write_bytes(source_archive(name, legacy=legacy))
    extract_converter(archive, tmp_path / name)
    return tmp_path / name


def test_environments_are_named_by_what_goes_in_them(tmp_path: Path) -> None:
    first, same = unpacked(tmp_path, "b1"), unpacked(tmp_path, "b2")
    newer = unpacked(tmp_path, "b3", legacy=LEGACY_REQUIREMENTS.replace(b"4.57.6", b"5.1.0"))
    assert environment_key(first, "3.12") == environment_key(same, "3.12")
    assert environment_key(first, "3.12") != environment_key(newer, "3.12")
    assert environment_key(first, "3.12") != environment_key(first, "3.13")


def test_requirements_cant_include_files_outside_the_source(tmp_path: Path) -> None:
    source = unpacked(tmp_path, "b1")
    (source / REQUIREMENTS_FILE).write_text("-r ../../outside.txt\n", encoding="utf-8")
    with pytest.raises(RuntimeInstallError, match="outside"):
        environment_key(source, "3.12")


def test_supported_models_are_read_with_or_without_the_log_prefix() -> None:
    plain = "TEXT models:\n  - LlamaForCausalLM\nMMPROJ models:\n  - LlavaForConditionalGeneration\n"
    assert parse_supported_models(plain) == {"TEXT": ["LlamaForCausalLM"], "MMPROJ": ["LlavaForConditionalGeneration"]}
    assert parse_supported_models("\n".join(SUPPORTED_OUTPUT)) == {
        "TEXT": ["LlamaForCausalLM", "Qwen3ForCausalLM"],
        "MMPROJ": ["Qwen2VLForConditionalGeneration"],
    }
    assert parse_supported_models("Traceback (most recent call last):\n  - not a model\n") == {}


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        # Lines as pip 25 printed them while installing the b12359 converter.
        ("Collecting torch==2.11.0 (from -r requirements-convert_hf_to_gguf.txt (line 5))", (True, "torch==2.11.0")),
        ("  Downloading numpy-2.2.6-cp312-cp312-win_amd64.whl.metadata (60 kB)", (False, None)),
        (
            "Downloading https://download-r2.pytorch.org/whl/cpu/torch-2.11.0%2Bcpu-cp312-cp312-win_amd64.whl (114.0 MB)",
            (True, "torch-2.11.0+cpu-cp312-cp312-win_amd64.whl"),
        ),
        ("Downloading transformers-4.57.6-py3-none-any.whl (12.0 MB)", (True, "transformers-4.57.6-py3-none-any.whl")),
        ("Installing collected packages: mpmath, urllib3, typing-extensions", (True, None)),
        ("Successfully installed torch-2.11.0+cpu transformers-4.57.6", (False, None)),
    ],
)
def test_pip_progress_shows_packages_not_metadata(line: str, expected: tuple[bool, str | None]) -> None:
    assert pip_current_file(line) == expected


def test_the_converter_runs_offline_without_the_apps_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    monkeypatch.setenv("GITHUB_TOKEN", "gh_secret")
    monkeypatch.setenv("PYTHONPATH", "C:/app")
    env = converter_environment()
    assert "HF_TOKEN" not in env and "GITHUB_TOKEN" not in env and "PYTHONPATH" not in env
    assert env["HF_HUB_OFFLINE"] == "1" and env["TRANSFORMERS_OFFLINE"] == "1"


# Installing


def test_installing_sets_up_the_converter_for_the_active_build(
    service: LlamaConverterService, commands: FakeCommands, github: FakeGitHub
) -> None:
    assert service.status().ready is False
    job = install(service)
    assert job.result is not None
    assert job.result["tag"] == "b100"
    assert job.result["commit"] == COMMIT
    assert job.result["new_environment"] is True
    assert job.result["text_architectures"] == 2
    assert job.result["already_installed"] is False
    assert commands.kinds() == ["venv", "pip", "check"]
    assert commands.calls[0][0] == sys.executable
    pip = commands.calls[1]
    assert pip[0] == str(environment_python(service.root / ENVIRONMENTS_DIR / job.result["environment"]))
    assert pip[-2:] == ["-r", str(service.root / SOURCES_DIR / ".tmp-b100" / REQUIREMENTS_FILE)]
    assert all(env is not None and env["HF_HUB_OFFLINE"] == "1" for env in commands.envs)
    assert github.requests == ["https://github.com/ggml-org/llama.cpp/archive/refs/tags/b100.tar.gz"]

    status = service.status()
    assert (status.active_tag, status.ready) == ("b100", True)
    [converter] = status.converters
    assert converter.tag == "b100" and converter.ready
    assert converter.text_architectures == ["LlamaForCausalLM", "Qwen3ForCausalLM"]
    assert converter.vision_architectures == ["Qwen2VLForConditionalGeneration"]
    # What downloads ask before converting a model, or its vision encoder.
    assert service.architectures("b100") == ["LlamaForCausalLM", "Qwen3ForCausalLM"]
    assert service.vision_architectures("b100") == ["Qwen2VLForConditionalGeneration"]
    assert service.vision_architectures("b99") is None
    assert (Path(converter.path) / "convert_hf_to_gguf.py").is_file()
    # Nothing is left behind but the converter and its environment.
    assert sorted(path.name for path in (service.root / SOURCES_DIR).iterdir()) == ["b100"]
    assert list((service.root / ".downloads").iterdir()) == []


def test_installing_again_changes_nothing(service: LlamaConverterService, commands: FakeCommands) -> None:
    install(service)
    again = install(service)
    assert again.result is not None and again.result["already_installed"] is True
    assert commands.kinds() == ["venv", "pip", "check"]


def test_builds_with_the_same_requirements_share_an_environment(
    service: LlamaConverterService, commands: FakeCommands
) -> None:
    first = install(service)
    second = install(service, "llama-b101-bin-win-vulkan-x64")
    assert first.result is not None and second.result is not None
    assert second.result["environment"] == first.result["environment"]
    assert second.result["new_environment"] is False
    assert any("another build's converter set up" in note for note in second.notes)
    assert commands.kinds() == ["venv", "pip", "check", "check"]
    assert len(list((service.root / ENVIRONMENTS_DIR).iterdir())) == 1


def test_new_requirements_get_a_new_environment(
    service: LlamaConverterService, commands: FakeCommands, github: FakeGitHub
) -> None:
    github.archives["b101"] = source_archive("b101", legacy=LEGACY_REQUIREMENTS.replace(b"4.57.6", b"5.1.0"))
    first = install(service)
    second = install(service, "llama-b101-bin-win-vulkan-x64")
    assert first.result is not None and second.result is not None
    assert second.result["environment"] != first.result["environment"]
    assert commands.kinds() == ["venv", "pip", "check", "venv", "pip", "check"]


def test_a_failed_pip_install_leaves_nothing_behind(service: LlamaConverterService, commands: FakeCommands) -> None:
    commands.results["pip"] = (1, ["Collecting torch==2.11.0", "ERROR: No matching distribution found for torch==2.11.0"])
    job = wait_for(service, service.start_install())
    assert job.status == "failed"
    assert job.error is not None and "No matching distribution found for torch" in job.error
    assert service.status().converters == []
    assert list((service.root / ENVIRONMENTS_DIR).iterdir()) == []
    assert list((service.root / SOURCES_DIR).iterdir()) == []


def test_an_environment_the_converter_cant_run_in_is_thrown_away(
    service: LlamaConverterService, commands: FakeCommands
) -> None:
    commands.results["check"] = (1, ["Traceback (most recent call last):", "ImportError: DLL load failed while importing torch"])
    job = wait_for(service, service.start_install())
    assert job.status == "failed"
    assert job.error is not None and "DLL load failed" in job.error
    assert list((service.root / ENVIRONMENTS_DIR).iterdir()) == []


def test_cancelling_during_pip_leaves_nothing_behind(service: LlamaConverterService, commands: FakeCommands) -> None:
    commands.hold = threading.Event()
    job = service.start_install()
    # Generous, for a busy machine: cancelling before pip would test something else.
    deadline = time.monotonic() + 30
    while "pip" not in commands.kinds() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert "pip" in commands.kinds(), f"the install never got to pip: {service.jobs.get(job.id)}"
    service.jobs.cancel(job.id)
    assert wait_for(service, job).status == "cancelled"
    assert list((service.root / ENVIRONMENTS_DIR).iterdir()) == []
    assert service.status().converters == []


def test_a_tag_github_doesnt_have_fails_the_job(service: LlamaConverterService, github: FakeGitHub) -> None:
    del github.archives["b100"]
    job = wait_for(service, service.start_install())
    assert job.status == "failed"
    assert job.error == "GitHub has no llama.cpp source for b100"


def test_there_has_to_be_a_build_to_install_for(service: LlamaConverterService, runtimes: FakeRuntimes) -> None:
    runtimes.active = None
    with pytest.raises(RuntimeNotFoundError):
        service.start_install()
    with pytest.raises(RuntimeNotFoundError):
        service.start_install("llama-b999-bin-win-vulkan-x64")
    assert service.status().active_tag is None


def test_not_enough_space_is_refused_up_front(
    service: LlamaConverterService, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(converter_module.shutil, "disk_usage", lambda path: type("U", (), {"free": 10**9})())
    with pytest.raises(ConverterSpaceError, match="about 3.2 GB and 1.0 GB is free"):
        service.start_install()


# Removing


def test_deleting_keeps_an_environment_another_build_uses(service: LlamaConverterService) -> None:
    install(service)
    install(service, "llama-b101-bin-win-vulkan-x64")
    service.delete("b100")
    assert [converter.tag for converter in service.status().converters] == ["b101"]
    assert len(list((service.root / ENVIRONMENTS_DIR).iterdir())) == 1
    service.delete("b101")
    assert service.status().converters == []
    assert list((service.root / ENVIRONMENTS_DIR).iterdir()) == []
    with pytest.raises(RuntimeNotFoundError):
        service.delete("b101")


@pytest.mark.parametrize("tag", ["..", "b1/../..", "sources", ".tmp-b100"])
def test_only_converters_can_be_deleted(service: LlamaConverterService, tag: str) -> None:
    install(service)
    with pytest.raises(RuntimeNotFoundError):
        service.delete(tag)
    assert service.status().converters


# Running commands


def _run_in_job(store: JobStore, work: Callable[[JobContext], dict]) -> JobInfo:
    job = store.submit("test", "key", work)
    deadline = time.monotonic() + 10
    while store.get(job.id).status in ("queued", "running"):
        assert time.monotonic() < deadline
        time.sleep(0.005)
    return store.get(job.id)


def test_a_command_s_output_comes_line_by_line() -> None:
    seen: list[str] = []

    def work(context: JobContext) -> dict:
        code, tail = run_command(
            [sys.executable, "-c", "import sys; print('one'); print('two'); sys.exit(3)"],
            context,
            timeout=30,
            on_line=seen.append,
        )
        return {"code": code, "tail": tail}

    job = _run_in_job(JobStore(), work)
    assert job.result == {"code": 3, "tail": ["one", "two"]}
    assert seen == ["one", "two"]


def test_cancelling_ends_a_command() -> None:
    store = JobStore()
    started = threading.Event()

    def work(context: JobContext) -> dict:
        started.set()
        run_command([sys.executable, "-c", "import time; time.sleep(60)"], context, timeout=120)
        return {}

    job = store.submit("test", "key", work)
    assert started.wait(5)
    time.sleep(0.3)
    began = time.monotonic()
    store.cancel(job.id)
    while store.get(job.id).status == "running":
        assert time.monotonic() - began < 5
        time.sleep(0.01)
    assert store.get(job.id).status == "cancelled"


def test_a_command_that_runs_too_long_is_ended() -> None:
    def work(context: JobContext) -> dict:
        run_command([sys.executable, "-c", "import time; time.sleep(60)"], context, timeout=0.5)
        return {}

    job = _run_in_job(JobStore(), work)
    assert job.status == "failed"
    assert job.error is not None and "didn't finish within" in job.error


# Routes


@pytest.fixture
def client(service: LlamaConverterService, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(runtime_router, "converter", service)
    monkeypatch.setattr(runtime_router, "service", service.runtimes)
    return TestClient(app)


def test_installing_and_removing_over_http(client: TestClient) -> None:
    assert client.get("/runtimes/llama.cpp/converter").json()["ready"] is False
    started = client.post("/runtimes/llama.cpp/converter/install")
    assert started.status_code == 202
    job_id = started.json()["id"]
    assert started.json()["details"]["tag"] == "b100"

    with client.stream("GET", f"/runtimes/llama.cpp/jobs/{job_id}/events") as response:
        lines = list(response.iter_lines())
    assert '"status":"completed"' in [line for line in lines if line.startswith("data: ")][-1]
    assert [job["id"] for job in client.get("/runtimes/llama.cpp/jobs").json()] == [job_id]

    body = client.get("/runtimes/llama.cpp/converter").json()
    assert (body["active_tag"], body["ready"]) == ("b100", True)
    assert body["converters"][0]["text_architectures"] == ["LlamaForCausalLM", "Qwen3ForCausalLM"]

    assert client.delete("/runtimes/llama.cpp/converter/b100").status_code == 204
    assert client.delete("/runtimes/llama.cpp/converter/b100").status_code == 404
    assert client.delete("/runtimes/llama.cpp/converter/latest").status_code == 422


def test_installing_for_a_named_build_over_http(client: TestClient) -> None:
    started = client.post("/runtimes/llama.cpp/converter/install", json={"runtime_id": "llama-b101-bin-win-vulkan-x64"})
    assert started.status_code == 202
    assert started.json()["details"]["tag"] == "b101"
    missing = client.post("/runtimes/llama.cpp/converter/install", json={"runtime_id": "llama-b999-bin-win-vulkan-x64"})
    assert missing.status_code == 404


def test_no_space_is_507_over_http(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(converter_module.shutil, "disk_usage", lambda path: type("U", (), {"free": 0})())
    response = client.post("/runtimes/llama.cpp/converter/install")
    assert response.status_code == 507
    assert "Not enough free space" in response.json()["detail"]

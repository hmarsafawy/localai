"""Downloads of repos with no GGUF, which are converted from their safetensors.
The converter and `llama-quantize` are stood in for: they're run for real by
hand (and their output below is theirs, from converting SmolLM2-135M)."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

import routes.library_router as library_router
import routes.model_download_router as model_download_router
import services.model_downloads as downloads_module
from main import app
from models.hf_model import HfModelRequirements, HfRequirementRole, HfRoleOption
from models.jobs import JobInfo
from services.jobs import JobContext, JobStore
from services.llama_converter import ConverterCommand, run_command
from services.llama_runtime import LaunchableRuntime, RuntimeNotFoundError
from services.model_conversion import (
    ConversionError,
    adapt_tokenizer_config,
    conversion_options,
    estimated_size_bytes,
    gguf_name,
    parse_quantized_tensor,
    parse_writing,
)
from services.model_downloads import (
    SOURCE_DIR,
    ModelDownloadRequestError,
    ModelDownloadService,
    NotEnoughSpaceError,
)
from services.model_options import MemoryBudget

REPO = "HuggingFaceTB/SmolLM2-135M-Instruct"
RUNTIME = "llama-b12359-bin-win-rocm-10.0-x64"
ROOMY = MemoryBudget(has_gpu=True, vram_free_bytes=16 * 10**9, ram_available_bytes=32 * 10**9)
FILES = {
    "config.json": b'{"architectures": ["LlamaForCausalLM"]}',
    "generation_config.json": b"{}",
    "model.safetensors": b"weights" * 1000,
    "tokenizer.json": b'{"model": {}}',
    "tokenizer_config.json": b"{}",
    "merges.txt": b"a b",
    "chat_template.jinja": b"{{ messages }}",
    # Never wanted: code, pickles, other formats and folders, documentation.
    "README.md": b"# SmolLM2",
    "modeling_smollm.py": b"import os",
    "training_args.bin": b"pickle",
    "onnx/model.onnx": b"onnx",
    "original/consolidated.safetensors": b"other copy",
}
WANTED = {name for name in FILES if "/" not in name and not name.endswith((".md", ".py", ".bin"))}
# config.json is read first as well, to tell what to convert the model as.
FETCHED = sorted([*WANTED, "config.json"])


class FakeHub:
    def __init__(self) -> None:
        self.files = dict(FILES)
        self.gets: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        file = request.url.path.split("/resolve/", 1)[1].split("/", 1)[1]
        if request.method == "HEAD":
            return httpx.Response(302)
        self.gets.append(file)
        if file not in self.files:
            return httpx.Response(404)
        return httpx.Response(200, content=self.files[file])


class FakeHf:
    token = None

    def __init__(self, hub: FakeHub) -> None:
        self.hub = hub
        self.api = self
        self.revision = "abc123"
        self._architectures: list[str] | None = ["LlamaForCausalLM"]
        self.parameters: int | None = 134_515_008
        self.vision = False
        # Whether the weights' tensor names are ones the requirements know.
        self.recognised = True

    @property
    def architectures(self) -> list[str] | None:
        return self._architectures

    @architectures.setter
    def architectures(self, value: list[str] | None) -> None:
        """The Hub's summary of config.json, and config.json with it."""
        self._architectures = value
        self.hub.files["config.json"] = json.dumps({"architectures": value} if value else {}).encode()

    def model_info(self, repo_id: str, files_metadata: bool = False) -> SimpleNamespace:
        return SimpleNamespace(
            id=REPO,
            sha=self.revision,
            config={"architectures": self._architectures} if self._architectures else {},
            safetensors=SimpleNamespace(total=self.parameters) if self.parameters else None,
            siblings=[
                SimpleNamespace(
                    rfilename=name,
                    size=len(body),
                    lfs=SimpleNamespace(sha256=hashlib.sha256(body).hexdigest()) if name.endswith("safetensors") else None,
                )
                for name, body in self.hub.files.items()
            ],
        )

    def get_model_requirements(self, repo_id: str) -> HfModelRequirements:
        if not self.recognised:
            unidentified = [name for name in self.hub.files if name.endswith(".safetensors")]
            return HfModelRequirements(model_id=REPO, roles=[], unidentified_files=unidentified)
        weights = [name for name in self.hub.files if name.endswith(".safetensors") and not name.startswith("original/")]
        options = [
            HfRoleOption(
                id="model",
                label="model",
                format="safetensors",
                files=weights,
                size_bytes=sum(len(self.hub.files[name]) for name in weights),
            )
        ] if weights else []
        role = HfRequirementRole(role="language_model", present=True, options=options, has_vision_weights=self.vision)
        return HfModelRequirements(model_id=REPO, roles=[role])


class FakeRuntimes:
    def __init__(self) -> None:
        self.active: str | None = RUNTIME

    def launchable(self, runtime_id: str | None = None) -> LaunchableRuntime:
        if self.active is None:
            raise RuntimeNotFoundError("No llama.cpp runtime is active.")
        return LaunchableRuntime(id=self.active, directory=Path("runtime"), backend="rocm", gpu_device="ROCm1")


class FakeConverter:
    def __init__(self) -> None:
        self.runtimes = FakeRuntimes()
        self.supported: list[str] | None = ["LlamaForCausalLM", "Qwen3ForCausalLM"]
        self.after_install: list[str] = ["LlamaForCausalLM"]
        self.installs: list[str] = []

    def architectures(self, tag: str) -> list[str] | None:
        return self.supported

    def vision_architectures(self, tag: str) -> list[str] | None:
        return None if self.supported is None else []

    def ensure_installed(self, context: JobContext, tag: str) -> dict:
        self.installs.append(tag)
        self.supported = self.after_install
        return {}

    def command(self, tag: str) -> ConverterCommand:
        return ConverterCommand(python=Path("python"), script=Path("convert_hf_to_gguf.py"))


class Tools:
    """Stands in for the converter and llama-quantize: writes their output."""

    def __init__(self) -> None:
        self.converted: list[tuple[Path, Path, str]] = []
        self.quantized: list[tuple[Path, Path, str]] = []
        self.quantize_error: Exception | None = None

    def convert(self, context: JobContext, converter: ConverterCommand, source: Path, outfile: Path, quantization: str) -> None:
        assert (source / "model.safetensors").is_file() and (source / "config.json").is_file()
        self.converted.append((source, outfile, quantization))
        context.begin_transfer(None)
        outfile.write_bytes(f"GGUF {quantization}".encode() * 100)

    def quantize(self, context: JobContext, runtime: LaunchableRuntime, source: Path, outfile: Path, quantization: str) -> None:
        assert source.is_file()
        self.quantized.append((source, outfile, quantization))
        if self.quantize_error is not None:
            raise self.quantize_error
        outfile.write_bytes(f"GGUF {quantization}".encode() * 10)


@pytest.fixture
def hub() -> FakeHub:
    return FakeHub()


@pytest.fixture
def hf(hub: FakeHub) -> FakeHf:
    return FakeHf(hub)


@pytest.fixture
def converter() -> FakeConverter:
    return FakeConverter()


@pytest.fixture
def tools(monkeypatch: pytest.MonkeyPatch) -> Tools:
    fake = Tools()
    monkeypatch.setattr(downloads_module, "convert", fake.convert)
    monkeypatch.setattr(downloads_module, "quantize", fake.quantize)
    return fake


@pytest.fixture
def space(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Free space as the service sees it, so a test doesn't depend on this disk."""
    free = {"bytes": 10**13}
    monkeypatch.setattr(downloads_module, "_free_bytes", lambda path: free["bytes"])
    return free


@pytest.fixture
def service(tmp_path: Path, hub: FakeHub, hf: FakeHf, converter: FakeConverter, tools: Tools, space: dict) -> ModelDownloadService:
    return ModelDownloadService(
        hf=hf,  # type: ignore[arg-type]
        budget=lambda: ROOMY,
        client=httpx.Client(transport=httpx.MockTransport(hub)),
        jobs=JobStore(max_concurrent=1),
        root=tmp_path / "model-files" / "text-generation",
        converter=converter,  # type: ignore[arg-type]
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


def folder_of(service: ModelDownloadService) -> Path:
    return service.root / "HuggingFaceTB--SmolLM2-135M-Instruct"


# Converting


def test_a_safetensors_repo_is_downloaded_converted_and_quantized(
    service: ModelDownloadService, hub: FakeHub, tools: Tools
) -> None:
    job = service.start(REPO, quantization="Q4_K_M")
    assert job.kind == "model_conversion"
    assert job.details["quantization"] == "Q4_K_M"
    assert job.details["conversion"]["architecture"] == "LlamaForCausalLM"
    assert job.details["conversion"]["converter"] == "b12359"
    assert job.details["conversion"]["parameters"] == 134_515_008
    assert job.details["size_bytes"] == sum(len(FILES[name]) for name in WANTED)

    done = wait_for(service, job)
    assert done.status == "completed", done.error
    folder = folder_of(service)
    final = folder / "SmolLM2-135M-Instruct-Q4_K_M.gguf"
    assert done.result is not None
    assert done.result["model_path"] == str(final)
    assert done.result["converted_with"] == "b12359"

    # Only the weights and what the converter reads alongside them.
    assert sorted(hub.gets) == FETCHED
    [(source, outfile, outtype)] = tools.converted
    assert source == folder / SOURCE_DIR
    assert outfile == folder / SOURCE_DIR / "SmolLM2-135M-Instruct-BF16.gguf.partial"
    assert outtype == "BF16"
    [(quantized_from, quantized_to, quantization)] = tools.quantized
    assert quantized_from == folder / SOURCE_DIR / "SmolLM2-135M-Instruct-BF16.gguf"
    assert quantized_to == folder / "SmolLM2-135M-Instruct-Q4_K_M.gguf.partial"
    assert quantization == "Q4_K_M"

    # Only the GGUF is kept, and it's in the library like any download.
    assert sorted(path.name for path in folder.iterdir()) == ["SmolLM2-135M-Instruct-Q4_K_M.gguf", "localai-model.json"]
    [model] = service.library().models
    [option] = model.quantizations
    assert (option.id, option.status, option.converted) == ("Q4_K_M", "complete", True)
    assert option.size_bytes == final.stat().st_size == model.size_on_disk_bytes
    assert service.local_selection(REPO).model.path == str(final)


def test_auto_picks_the_biggest_quantization_that_fits(
    service: ModelDownloadService, hf: FakeHf
) -> None:
    hf.parameters = 8_030_000_000
    service._budget = lambda: MemoryBudget(has_gpu=True, vram_free_bytes=6 * 10**9, ram_available_bytes=32 * 10**9)
    job = service.start(REPO)
    # Q4_K_M is about 4.9 GB, 5.9 GB with llama.cpp's buffers; Q5_K_S doesn't fit.
    assert job.details["quantization"] == "Q4_K_M"
    wait_for(service, job)


def test_what_the_converter_writes_itself_isnt_quantized(service: ModelDownloadService, tools: Tools) -> None:
    done = wait_for(service, service.start(REPO, quantization="Q8_0"))
    assert done.status == "completed", done.error
    [(_, outfile, outtype)] = tools.converted
    assert outfile == folder_of(service) / "SmolLM2-135M-Instruct-Q8_0.gguf.partial"
    assert outtype == "Q8_0"
    assert tools.quantized == []


def test_converting_again_is_already_done(service: ModelDownloadService, hub: FakeHub, tools: Tools) -> None:
    wait_for(service, service.start(REPO, quantization="Q4_K_M"))
    hub.gets.clear()
    again = wait_for(service, service.start(REPO, quantization="Q4_K_M"))
    assert again.result is not None and again.result["already_downloaded"] is True
    assert hub.gets == ["config.json"] and len(tools.converted) == 1


def test_a_failed_quantize_keeps_the_bf16_and_trying_again_starts_from_it(
    service: ModelDownloadService, hub: FakeHub, tools: Tools
) -> None:
    tools.quantize_error = ConversionError("llama-quantize failed (exit code 1): out of memory")
    failed = wait_for(service, service.start(REPO, quantization="Q4_K_M"))
    assert failed.status == "failed"
    assert failed.error == "llama-quantize failed (exit code 1): out of memory"
    source = folder_of(service) / SOURCE_DIR
    assert sorted(path.name for path in source.iterdir()) == ["SmolLM2-135M-Instruct-BF16.gguf", "localai-source.json"]
    [option] = service.library().models[0].quantizations
    assert (option.status, option.converted) == ("incomplete", True)

    tools.quantize_error = None
    hub.gets.clear()
    done = wait_for(service, service.start(REPO, quantization="Q4_K_M"))
    assert done.status == "completed", done.error
    assert hub.gets == ["config.json"] and len(tools.converted) == 1 and len(tools.quantized) == 2
    assert not source.exists()


def test_a_bf16_of_an_older_version_isnt_reused(service: ModelDownloadService, hf: FakeHf, tools: Tools) -> None:
    tools.quantize_error = ConversionError("llama-quantize failed")
    wait_for(service, service.start(REPO, quantization="Q4_K_M"))
    tools.quantize_error = None
    hf.revision = "def456"
    done = wait_for(service, service.start(REPO, quantization="Q4_K_M"))
    assert done.status == "completed", done.error
    assert len(tools.converted) == 2


def test_deleting_an_unfinished_conversion_removes_its_download(service: ModelDownloadService, tools: Tools) -> None:
    tools.quantize_error = ConversionError("llama-quantize failed")
    wait_for(service, service.start(REPO, quantization="Q4_K_M"))
    assert service.library().models[0].size_on_disk_bytes > 0
    service.delete(REPO, quantization="Q4_K_M")
    assert not folder_of(service).exists()


def test_the_converter_is_set_up_first_when_it_isnt_yet(service: ModelDownloadService, converter: FakeConverter) -> None:
    converter.supported = None
    job = service.start(REPO, quantization="Q4_K_M")
    done = wait_for(service, job)
    assert done.status == "completed", done.error
    assert converter.installs == ["b12359"]
    assert any("isn't set up yet, so it's set up first" in note for note in done.notes)


def test_a_model_the_new_converter_cant_convert_fails_before_downloading(
    service: ModelDownloadService, converter: FakeConverter, hub: FakeHub
) -> None:
    converter.supported = None
    converter.after_install = ["Qwen3ForCausalLM"]
    done = wait_for(service, service.start(REPO, quantization="Q4_K_M"))
    assert done.status == "failed"
    assert done.error is not None and "can't convert LlamaForCausalLM models" in done.error
    assert hub.gets == ["config.json"]


# Refused up front


def test_a_model_the_converter_cant_convert_is_refused(service: ModelDownloadService, converter: FakeConverter) -> None:
    converter.supported = ["Qwen3ForCausalLM"]
    with pytest.raises(ModelDownloadRequestError, match="llama.cpp b12359 can't convert LlamaForCausalLM models"):
        service.start(REPO)
    assert not service.jobs.all_jobs()


def test_weights_with_unrecognised_tensor_names_are_converted_by_what_config_json_says(
    service: ModelDownloadService, hub: FakeHub, hf: FakeHf, tools: Tools
) -> None:
    # A new or custom architecture: the converter reads the top folder's model*.safetensors.
    hf.recognised = False
    done = wait_for(service, service.start(REPO, quantization="Q4_K_M"))
    assert done.status == "completed", done.error
    assert sorted(hub.gets) == FETCHED


def test_unrecognised_weights_of_a_model_the_converter_doesnt_know_are_refused_for_that(
    service: ModelDownloadService, hub: FakeHub, hf: FakeHf
) -> None:
    # AuroraAI-Research/Aurora2-400M-A20M was refused as having no safetensors at all.
    hf.recognised = False
    hf.architectures = ["Aurora2ForCausalLM"]
    with pytest.raises(ModelDownloadRequestError, match="b12359 can't convert Aurora2ForCausalLM models"):
        service.start(REPO)
    del hub.files["model.safetensors"]
    with pytest.raises(ModelDownloadRequestError, match="neither GGUF files to run nor safetensors weights"):
        service.start(REPO)
    assert not service.jobs.all_jobs()


def test_a_model_that_doesnt_say_what_it_is_is_refused(service: ModelDownloadService, hf: FakeHf) -> None:
    hf.architectures = None
    with pytest.raises(ModelDownloadRequestError, match="doesn't say what model it is"):
        service.start(REPO)


def test_converting_needs_a_runtime(service: ModelDownloadService, converter: FakeConverter) -> None:
    converter.runtimes.active = None
    with pytest.raises(ModelDownloadRequestError, match="needs a llama.cpp runtime: No llama.cpp runtime is active"):
        service.start(REPO)


def test_only_known_quantizations_can_be_chosen(service: ModelDownloadService) -> None:
    with pytest.raises(ModelDownloadRequestError, match="'IQ1_S' isn't available. Available: BF16, F16, Q8_0"):
        service.start(REPO, quantization="IQ1_S")


def test_not_enough_space_for_the_download_and_its_bf16_is_refused(
    service: ModelDownloadService, hf: FakeHf, space: dict
) -> None:
    hf.parameters = 8_030_000_000
    space["bytes"] = 20 * 10**9
    with pytest.raises(NotEnoughSpaceError, match="needs about 21.0 GB at once, and 20.0 GB is free"):
        service.start(REPO, quantization="Q4_K_M")
    assert not service.jobs.all_jobs()


def test_weights_in_a_subfolder_are_refused(service: ModelDownloadService, hub: FakeHub) -> None:
    hub.files = {name: body for name, body in FILES.items() if name != "model.safetensors"}
    hub.files["weights/model.safetensors"] = b"x"
    with pytest.raises(ModelDownloadRequestError, match="keeps its weights in a subfolder"):
        service.start(REPO)


# Estimates and the tools' output


def test_sizes_are_estimated_from_llama_quantizes_own_table() -> None:
    # `llama-quantize --help`: Q4_K_M is 4.58G and Q8_0 7.96G for Llama-3-8B.
    assert estimated_size_bytes(8_030_000_000, "Q4_K_M") == pytest.approx(4.58 * 1024**3, rel=0.01)
    assert estimated_size_bytes(8_030_000_000, "Q8_0") == pytest.approx(7.96 * 1024**3, rel=0.01)
    assert gguf_name(REPO, "Q6_K") == "SmolLM2-135M-Instruct-Q6_K.gguf"
    options = conversion_options(REPO, 1_000_000_000)
    assert [option.id for option in options][:3] == ["BF16", "F16", "Q8_0"]
    assert all(option.format == "gguf" and option.files == [gguf_name(REPO, option.id)] for option in options)


def test_the_converters_progress_bar_is_read() -> None:
    assert parse_writing("Writing:  21%|██        | 56.6M/269M [00:00<00:00, 238Mbyte/s]") == (56_600_000, 269_000_000)
    assert parse_writing("Writing: 100%|██████████| 16.4G/16.4G [01:15<00:00, 217Mbyte/s]") == (16_400_000_000, 16_400_000_000)
    assert parse_writing("INFO:hf-to-gguf:Set model tokenizer") is None


def test_llama_quantizes_tensor_lines_are_read() -> None:
    line = (
        "[ 263/ 272]                 blk.28.ffn_up.weight - [   576,   1536,      1,      1], "
        "type =   bf16, converting to q5_0 .. size =     1.69 MiB ->     0.58 MiB"
    )
    assert parse_quantized_tensor(line) == (263, 272, int(1.69 * 1024**2))
    kept = "[   1/ 272]                   output_norm.weight - [   576,      1,      1,      1], type =    f32, size =    0.002 MiB"
    assert parse_quantized_tensor(kept) == (1, 272, int(0.002 * 1024**2))
    assert parse_quantized_tensor("llama_model_quantize_impl: model size  =   256.63 MiB (16.00 BPW)") is None


def test_a_tokenizer_config_from_transformers_5_is_rewritten_for_4(tmp_path: Path) -> None:
    # LiquidAI/LFM3-1.7B's, cut down. The converter's transformers 4.57 refused it.
    config = {
        "backend": "tokenizers",
        "bos_token": "<|startoftext|>",
        "extra_special_tokens": ["<|image|>", {"content": "<|audio|>", "special": True}],
        "additional_special_tokens": ["<|tool|>", "<|image|>"],
        "image_token": "<|image|>",
        "tokenizer_class": "TokenizersBackend",
    }
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(config), encoding="utf-8")

    assert adapt_tokenizer_config(tmp_path)
    adapted = json.loads((tmp_path / "tokenizer_config.json").read_text(encoding="utf-8"))
    assert adapted["tokenizer_class"] == "PreTrainedTokenizerFast"
    assert "extra_special_tokens" not in adapted
    assert adapted["additional_special_tokens"] == ["<|tool|>", "<|image|>", {"content": "<|audio|>", "special": True}]
    assert adapted["bos_token"] == "<|startoftext|>" and adapted["image_token"] == "<|image|>"
    assert not adapt_tokenizer_config(tmp_path)


def test_a_tokenizer_config_transformers_4_reads_is_left_alone(tmp_path: Path) -> None:
    text = '{"tokenizer_class": "GPT2Tokenizer", "extra_special_tokens": {"image_token": "<image>"}}'
    (tmp_path / "tokenizer_config.json").write_text(text, encoding="utf-8")
    assert not adapt_tokenizer_config(tmp_path)
    assert (tmp_path / "tokenizer_config.json").read_text(encoding="utf-8") == text
    (tmp_path / "tokenizer_config.json").write_text("not json", encoding="utf-8")
    assert not adapt_tokenizer_config(tmp_path)
    assert not adapt_tokenizer_config(tmp_path / "missing")


def test_a_progress_bars_carriage_returns_split_lines() -> None:
    store = JobStore()
    seen: list[str] = []

    def work(context: JobContext) -> dict:
        script = "import sys; sys.stdout.write('one\\rtwo\\r\\nthree\\n'); sys.stdout.write('four')"
        run_command([sys.executable, "-c", script], context, timeout=30, on_line=seen.append)
        return {}

    job = store.submit("test", "key", work)
    deadline = time.monotonic() + 10
    while store.get(job.id).status in ("queued", "running"):
        assert time.monotonic() < deadline
        time.sleep(0.005)
    assert store.get(job.id).status == "completed"
    assert seen == ["one", "two", "three", "four"]


# Over HTTP


@pytest.fixture
def client(service: ModelDownloadService, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(model_download_router, "service", service)
    monkeypatch.setattr(library_router, "service", service)
    return TestClient(app)


def test_converting_over_http(client: TestClient) -> None:
    started = client.post("/downloads", json={"repo_id": REPO, "quantization": "Q4_K_M"})
    assert started.status_code == 202
    body = started.json()
    assert body["kind"] == "model_conversion"
    assert body["details"]["conversion"]["disk_needed_bytes"] > 0

    with client.stream("GET", f"/downloads/{body['id']}/events") as response:
        lines = [line for line in response.iter_lines() if line.startswith("data: ")]
    assert '"status":"completed"' in lines[-1]

    [model] = client.get("/library").json()["models"]
    assert model["quantizations"][0]["converted"] is True
    assert client.post("/downloads", json={"repo_id": REPO, "quantization": "IQ1_S"}).status_code == 422

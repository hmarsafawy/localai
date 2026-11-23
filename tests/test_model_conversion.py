"""Downloads of repos with no GGUF, which are converted from their safetensors,
along with a vision model's vision encoder. The converter and `llama-quantize`
are stood in for: they're run for real by hand (and their output below is
theirs, from converting SmolLM2-135M)."""

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
    text_architectures,
    vision_architecture,
    vision_conversion_options,
    vision_gguf_name,
)
from services.model_downloads import (
    SOURCE_DIR,
    LocalModelChoiceError,
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
# SmolVLM-256M-Instruct's, cut down: the converter takes its text model by the
# text config's name and its vision encoder by the model's.
VISION_CONFIG = {
    "architectures": ["Idefics3ForConditionalGeneration"],
    "text_config": {"architectures": ["VLlama3ForCausalLM"], "model_type": "llama"},
    "vision_config": {"model_type": "idefics3"},
}
VISION_PARAMETERS = 93_000_000
MMPROJ = "mmproj-SmolLM2-135M-Instruct-F16.gguf"


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
        self.vision_parameters = 0
        self.headers_error: Exception | None = None
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

    def make_vision(self) -> None:
        """Makes the model a vision model, with its vision encoder in its weights."""
        self._architectures = VISION_CONFIG["architectures"]
        self.hub.files["config.json"] = json.dumps(VISION_CONFIG).encode()
        self.vision = True
        self.vision_parameters = VISION_PARAMETERS

    def model_info(self, repo_id: str, files_metadata: bool = False) -> SimpleNamespace:
        return SimpleNamespace(
            id=REPO,
            sha=self.revision,
            # The Hub's summary leaves out a vision model's text and vision configs.
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

    def get_safetensors_metadata(self, repo_id: str, revision: str | None = None) -> SimpleNamespace:
        if self.headers_error is not None:
            raise self.headers_error
        tensors = {"model.text_model.layers.0.mlp.weight": SimpleNamespace(parameter_count=self.parameters or 0)}
        if self.vision_parameters:
            tensors["model.vision_model.encoder.layers.0.mlp.weight"] = SimpleNamespace(
                parameter_count=self.vision_parameters
            )
        return SimpleNamespace(files_metadata={"model.safetensors": SimpleNamespace(tensors=tensors)})


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
        self.supported: list[str] | None = ["LlamaForCausalLM", "Qwen3ForCausalLM", "VLlama3ForCausalLM"]
        self.after_install: list[str] = ["LlamaForCausalLM", "VLlama3ForCausalLM"]
        self.vision_supported: list[str] | None = ["Idefics3ForConditionalGeneration"]
        self.vision_after_install: list[str] = ["Idefics3ForConditionalGeneration"]
        self.installs: list[str] = []

    def architectures(self, tag: str) -> list[str] | None:
        return self.supported

    def vision_architectures(self, tag: str) -> list[str] | None:
        return None if self.supported is None else self.vision_supported

    def ensure_installed(self, context: JobContext, tag: str) -> dict:
        self.installs.append(tag)
        self.supported = self.after_install
        self.vision_supported = self.vision_after_install
        return {}

    def command(self, tag: str) -> ConverterCommand:
        return ConverterCommand(python=Path("python"), script=Path("convert_hf_to_gguf.py"))


class Tools:
    """Stands in for the converter and llama-quantize: writes their output."""

    def __init__(self) -> None:
        self.converted: list[tuple[Path, Path, str]] = []
        self.converted_vision: list[tuple[Path, Path, str]] = []
        self.quantized: list[tuple[Path, Path, str]] = []
        self.quantize_error: Exception | None = None
        self.vision_error: Exception | None = None

    def convert(
        self,
        context: JobContext,
        converter: ConverterCommand,
        source: Path,
        outfile: Path,
        quantization: str,
        vision: bool = False,
    ) -> None:
        assert (source / "model.safetensors").is_file() and (source / "config.json").is_file()
        context.begin_transfer(None)
        if vision:
            self.converted_vision.append((source, outfile, quantization))
            outfile.write_bytes(b"half written")
            if self.vision_error is not None:
                raise self.vision_error
            outfile.write_bytes(f"MMPROJ {quantization}".encode() * 30)
            return
        self.converted.append((source, outfile, quantization))
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


# Vision encoders


def test_a_vision_models_encoder_is_converted_from_the_same_download(
    service: ModelDownloadService, hub: FakeHub, hf: FakeHf, tools: Tools
) -> None:
    hf.make_vision()
    job = service.start(REPO, quantization="Q4_K_M")
    assert job.details["vision_encoder"] == "F16"
    conversion = job.details["conversion"]
    # The text model goes by its text config, and leaves the vision encoder out.
    assert conversion["architecture"] == "VLlama3ForCausalLM"
    assert conversion["parameters"] == 134_515_008 - VISION_PARAMETERS
    assert conversion["vision_encoder"] == {
        "architecture": "Idefics3ForConditionalGeneration",
        "parameters": VISION_PARAMETERS,
        "estimated_size_bytes": VISION_PARAMETERS * 2 + 1_000_000,
    }

    done = wait_for(service, job)
    assert done.status == "completed", done.error
    folder = folder_of(service)
    source = folder / SOURCE_DIR
    assert sorted(hub.gets) == FETCHED  # Downloaded once for both.
    [(converted_from, outfile, outtype)] = tools.converted_vision
    assert (converted_from, outfile, outtype) == (source, source / f"{MMPROJ}.partial", "F16")
    assert done.result is not None
    assert done.result["vision_encoder"] == "F16"
    assert done.result["vision_encoder_path"] == str(folder / MMPROJ)
    assert sorted(path.name for path in folder.iterdir()) == [
        "SmolLM2-135M-Instruct-Q4_K_M.gguf",
        "localai-model.json",
        MMPROJ,
    ]

    [model] = service.library().models
    [vision] = model.vision_encoders
    assert (vision.id, vision.status, vision.converted) == ("F16", "complete", True)
    assert vision.size_bytes == (folder / MMPROJ).stat().st_size
    assert model.size_on_disk_bytes == sum(path.stat().st_size for path in folder.glob("*.gguf"))
    # And it's run with the model, as a downloaded one is.
    selection = service.local_selection(REPO)
    assert selection.vision is not None and selection.vision.path == str(folder / MMPROJ)


def source_bytes(hub: FakeHub) -> int:
    return sum(len(body) for name, body in hub.files.items() if name in WANTED)


def test_space_for_the_vision_encoder_is_counted(
    service: ModelDownloadService, hub: FakeHub, hf: FakeHf, space: dict
) -> None:
    hf.make_vision()
    hf.parameters = 8_030_000_000
    hf.vision_parameters = 600_000_000
    text = source_bytes(hub) + estimated_size_bytes(8_030_000_000 - 600_000_000, "BF16")
    vision = 600_000_000 * 2 + 1_000_000
    space["bytes"] = text + vision + 10**9 - 1
    with pytest.raises(NotEnoughSpaceError, match="converting HuggingFaceTB/SmolLM2-135M-Instruct to BF16"):
        service.start(REPO, quantization="BF16")
    space["bytes"] += 1
    job = service.start(REPO, quantization="BF16")
    assert job.details["conversion"]["disk_needed_bytes"] == text + vision
    wait_for(service, job)


def test_a_vision_encoder_can_be_left_out_and_added_later(
    service: ModelDownloadService, hub: FakeHub, hf: FakeHf, tools: Tools
) -> None:
    hf.make_vision()
    done = wait_for(service, service.start(REPO, quantization="Q4_K_M", vision_encoder="none"))
    assert done.status == "completed", done.error
    assert done.details["vision_encoder"] is None and done.details["conversion"]["vision_encoder"] is None
    assert tools.converted_vision == [] and done.notes == []
    assert service.library().models[0].vision_encoders == []

    # Asked for later, it's converted on its own from a new download.
    hub.gets.clear()
    job = service.start(REPO, quantization="Q4_K_M", vision_encoder="Q8_0")
    # Only the download and the vision encoder.
    [q8_0] = [option for option in vision_conversion_options(REPO, VISION_PARAMETERS) if option.id == "Q8_0"]
    assert job.details["conversion"]["disk_needed_bytes"] == source_bytes(hub) + q8_0.size_bytes
    done = wait_for(service, job)
    assert done.status == "completed", done.error
    assert sorted(hub.gets) == FETCHED
    assert len(tools.converted) == 1 and len(tools.quantized) == 1
    assert [outtype for _, _, outtype in tools.converted_vision] == ["Q8_0"]
    assert not (folder_of(service) / SOURCE_DIR).exists()
    [vision] = service.library().models[0].vision_encoders
    assert vision.id == "Q8_0"

    # And then it's all there already.
    again = wait_for(service, service.start(REPO, quantization="Q4_K_M", vision_encoder="Q8_0"))
    assert again.result is not None and again.result["already_downloaded"] is True


def test_a_vision_encoder_is_chosen_from_what_the_converter_writes(service: ModelDownloadService, hf: FakeHf) -> None:
    hf.make_vision()
    with pytest.raises(ModelDownloadRequestError, match="Vision encoder 'Q4_K_M' isn't available. Available: F16, BF16, Q8_0"):
        service.start(REPO, vision_encoder="Q4_K_M")
    job = service.start(REPO, quantization="Q8_0", vision_encoder="bf16")
    assert job.details["vision_encoder"] == "BF16"
    wait_for(service, job)


def test_asking_for_a_vision_encoder_a_model_lacks_is_an_error(service: ModelDownloadService, hf: FakeHf) -> None:
    with pytest.raises(ModelDownloadRequestError, match="has no vision encoder to convert"):
        service.start(REPO, vision_encoder="F16")
    # One its config declares, but whose weights aren't there, isn't one either.
    hf.make_vision()
    hf.vision = False
    with pytest.raises(ModelDownloadRequestError, match="has no vision encoder to convert"):
        service.start(REPO, vision_encoder="F16")
    job = service.start(REPO, quantization="Q4_K_M")
    assert job.details["vision_encoder"] is None
    wait_for(service, job)


def test_weights_whose_headers_have_no_vision_encoder_dont_get_one(
    service: ModelDownloadService, hf: FakeHf, tools: Tools
) -> None:
    # Unrecognised tensor names: whether there's a vision encoder is up to the headers.
    hf.make_vision()
    hf.recognised = False
    hf.vision_parameters = 0
    done = wait_for(service, service.start(REPO, quantization="Q4_K_M"))
    assert done.status == "completed", done.error
    assert done.details["vision_encoder"] is None and tools.converted_vision == []


def test_unreadable_headers_still_get_the_vision_encoder_converted(
    service: ModelDownloadService, hf: FakeHf, tools: Tools
) -> None:
    hf.make_vision()
    hf.headers_error = OSError("connection reset")
    job = service.start(REPO, quantization="Q4_K_M")
    # Without them, there's no telling the vision encoder's size.
    assert job.details["conversion"]["parameters"] == 134_515_008
    assert job.details["conversion"]["vision_encoder"]["estimated_size_bytes"] is None
    done = wait_for(service, job)
    assert done.status == "completed", done.error
    assert len(tools.converted_vision) == 1


def test_a_vision_encoder_the_converter_cant_convert_is_left_out_unless_asked_for(
    service: ModelDownloadService, hf: FakeHf, converter: FakeConverter, tools: Tools
) -> None:
    hf.make_vision()
    converter.vision_supported = ["Qwen2VLForConditionalGeneration"]
    reason = "llama.cpp b12359 can't convert Idefics3ForConditionalGeneration's vision encoder."
    with pytest.raises(ModelDownloadRequestError, match=f"{reason} Choose vision_encoder none"):
        service.start(REPO, vision_encoder="F16")
    done = wait_for(service, service.start(REPO, quantization="Q4_K_M"))
    assert done.status == "completed", done.error
    assert done.notes == [f"Converted as a text model: {reason}"]
    assert done.details["vision_encoder"] is None and tools.converted_vision == []


def test_a_new_converters_vision_encoders_are_checked_once_its_set_up(
    service: ModelDownloadService, hf: FakeHf, converter: FakeConverter, hub: FakeHub, tools: Tools
) -> None:
    hf.make_vision()
    converter.supported = None
    converter.vision_after_install = []
    job = service.start(REPO, quantization="Q4_K_M")
    assert job.details["vision_encoder"] == "F16"
    done = wait_for(service, job)
    assert done.status == "completed", done.error
    assert any("can't convert Idefics3ForConditionalGeneration's vision encoder" in note for note in done.notes)
    assert done.details["vision_encoder"] is None and done.details["conversion"]["vision_encoder"] is None
    assert tools.converted_vision == []

    # Asked for by name, it fails before downloading anything.
    service.delete(REPO)
    converter.supported = None
    hub.gets.clear()
    failed = wait_for(service, service.start(REPO, quantization="Q4_K_M", vision_encoder="F16"))
    assert failed.status == "failed"
    assert failed.error == "llama.cpp b12359 can't convert Idefics3ForConditionalGeneration's vision encoder."
    assert hub.gets == ["config.json"]


def test_a_vision_encoder_left_to_auto_that_fails_leaves_the_text_model(
    service: ModelDownloadService, hf: FakeHf, tools: Tools
) -> None:
    hf.make_vision()
    tools.vision_error = ConversionError("llama.cpp's converter failed (exit code 1): KeyError: 'image_mean'")
    done = wait_for(service, service.start(REPO, quantization="Q4_K_M"))
    assert done.status == "completed", done.error
    assert done.notes == [
        (
            "Converted as a text model: its vision encoder couldn't be converted. "
            "llama.cpp's converter failed (exit code 1): KeyError: 'image_mean'"
        )
    ]
    assert done.result is not None and done.result["vision_encoder"] is None
    assert done.details["vision_encoder"] is None
    folder = folder_of(service)
    assert sorted(path.name for path in folder.iterdir()) == ["SmolLM2-135M-Instruct-Q4_K_M.gguf", "localai-model.json"]
    assert service.library().models[0].vision_encoders == []


def test_a_vision_encoder_asked_for_that_fails_fails_the_job_and_trying_again_resumes(
    service: ModelDownloadService, hub: FakeHub, hf: FakeHf, tools: Tools
) -> None:
    hf.make_vision()
    tools.vision_error = ConversionError("llama.cpp's converter failed (exit code 1): out of memory")
    failed = wait_for(service, service.start(REPO, quantization="Q4_K_M", vision_encoder="F16"))
    assert failed.status == "failed"
    assert failed.error == "llama.cpp's converter failed (exit code 1): out of memory"
    # The download and the BF16 are kept, and nothing half written.
    source = folder_of(service) / SOURCE_DIR
    assert f"{MMPROJ}.partial" not in {path.name for path in source.iterdir()}
    assert {"model.safetensors", "SmolLM2-135M-Instruct-BF16.gguf"} <= {path.name for path in source.iterdir()}

    tools.vision_error = None
    hub.gets.clear()
    done = wait_for(service, service.start(REPO, quantization="Q4_K_M", vision_encoder="F16"))
    assert done.status == "completed", done.error
    assert hub.gets == ["config.json"] and len(tools.converted) == 1 and len(tools.converted_vision) == 2
    assert (folder_of(service) / MMPROJ).is_file() and not source.exists()


def test_the_text_model_of_a_vision_model_is_checked_by_the_name_the_converter_goes_by(
    service: ModelDownloadService, hf: FakeHf, converter: FakeConverter
) -> None:
    hf.make_vision()
    converter.supported = ["LlamaForCausalLM"]
    with pytest.raises(ModelDownloadRequestError, match="can't convert Idefics3ForConditionalGeneration models"):
        service.start(REPO)
    # A few the converter takes by the model's own name.
    converter.supported = ["Idefics3ForConditionalGeneration"]
    job = service.start(REPO, quantization="Q4_K_M")
    assert job.details["conversion"]["architecture"] == "Idefics3ForConditionalGeneration"
    wait_for(service, job)


def test_deleting_a_converted_vision_encoder_leaves_the_model(service: ModelDownloadService, hf: FakeHf) -> None:
    hf.make_vision()
    wait_for(service, service.start(REPO, quantization="Q4_K_M"))
    service.delete(REPO, vision_encoder="F16")
    [model] = service.library().models
    assert model.vision_encoders == [] and model.quantizations[0].status == "complete"
    assert not (folder_of(service) / MMPROJ).exists()
    with pytest.raises(LocalModelChoiceError, match="has no vision encoder downloaded"):
        service.local_selection(REPO, vision_encoder="F16")


def test_deleting_a_vision_encoder_keeps_an_unfinished_models_download(
    service: ModelDownloadService, hf: FakeHf, tools: Tools
) -> None:
    hf.make_vision()
    wait_for(service, service.start(REPO, quantization="Q8_0"))
    tools.quantize_error = ConversionError("llama-quantize failed")
    wait_for(service, service.start(REPO, quantization="Q4_K_M"))
    service.delete(REPO, vision_encoder="F16")
    source = folder_of(service) / SOURCE_DIR
    assert (source / "SmolLM2-135M-Instruct-BF16.gguf").is_file()
    service.delete(REPO, quantization="Q4_K_M")
    assert not source.exists()


def test_what_the_converter_goes_by_comes_from_config_json() -> None:
    assert text_architectures({"architectures": ["LlamaForCausalLM"]}) == ["LlamaForCausalLM"]
    assert text_architectures(VISION_CONFIG) == ["VLlama3ForCausalLM", "Idefics3ForConditionalGeneration"]
    # llava_onevision2 only names its text model.
    assert text_architectures({"text_config": {"architectures": ["Qwen3ForCausalLM"]}}) == ["Qwen3ForCausalLM"]
    assert text_architectures({}) == []
    assert vision_architecture(VISION_CONFIG) == "Idefics3ForConditionalGeneration"
    named = {**VISION_CONFIG, "vision_config": {"architectures": ["RADIOModel"]}}
    assert vision_architecture(named) == "RADIOModel"
    assert vision_architecture({"architectures": ["LlamaForCausalLM"]}) is None
    # An audio model has an encoder too, but not a vision one.
    assert vision_architecture({"architectures": ["Lfm2AudioForConditionalGeneration"], "audio_config": {}}) is None


def test_vision_encoder_options_are_what_the_converter_writes() -> None:
    assert vision_gguf_name(REPO, "F16") == MMPROJ
    options = vision_conversion_options(REPO, 400_000_000)
    assert [(option.id, option.size_bytes) for option in options] == [
        ("F16", 801_000_000),
        ("BF16", 801_000_000),
        ("Q8_0", 426_500_000),
    ]
    assert all(option.size_bytes is None for option in vision_conversion_options(REPO, None))


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
    assert client.post("/downloads", json={"repo_id": REPO, "vision_encoder": "F16"}).status_code == 422


def test_converting_a_vision_model_over_http(client: TestClient, hf: FakeHf) -> None:
    hf.make_vision()
    started = client.post("/downloads", json={"repo_id": REPO, "quantization": "Q8_0"})
    assert started.status_code == 202
    assert started.json()["details"]["vision_encoder"] == "F16"
    with client.stream("GET", f"/downloads/{started.json()['id']}/events") as response:
        lines = [line for line in response.iter_lines() if line.startswith("data: ")]
    assert '"status":"completed"' in lines[-1]
    [model] = client.get("/library").json()["models"]
    assert model["vision_encoders"][0]["converted"] is True

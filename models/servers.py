from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from models.hf_model import HfMemoryEstimate


class ServerSettings(BaseModel):
    """What was asked for. Anything left unset is chosen by llama.cpp, which
    sizes it to the VRAM that's free when the model loads."""

    # Null: the model's full context, cut down to what fits.
    context_size: int | None = None
    # "auto", "all", or a number of layers. 0 runs on the CPU only.
    gpu_layers: int | str = "auto"


class GpuPlan(BaseModel):
    """How the model was fitted to the GPU before it started."""

    # VRAM free when it was launched, counting what other programs use.
    vram_free_bytes: int | None = None
    # What llama.cpp was told to leave free (`--fit-target`). It's more than its
    # own margin when the driver overstates what's free, and when there's a
    # vision encoder, which llama.cpp leaves out of its sums.
    fit_target_mib: int | None = None
    # Whether llama.cpp planned to put the whole model on the GPU. Null when
    # that couldn't be asked, or when `gpu_layers` was set to a number.
    whole_model_on_gpu: bool | None = None
    # The longest context the memory estimate says keeps the whole model on the
    # GPU, with what was free. Null if even 4,096 tokens don't, or if the
    # model's files couldn't be read.
    max_gpu_context: int | None = None


class GpuMemory(BaseModel):
    """Where the server's GPU memory ended up once it had loaded."""

    vram_bytes: int
    # What Windows placed in system RAM instead ("shared GPU memory"). A few
    # hundred MB is normal; more means VRAM ran out and it runs slower.
    system_ram_bytes: int


class LoadedModel(BaseModel):
    """What llama-server reports once the model is loaded."""

    # The context, shared by its slots: one request can use all of it.
    context_size: int | None = None
    # How many requests it serves at once.
    slots: int | None = None
    vision: bool = False
    build: str | None = None
    # What the memory estimate says the whole model takes on the GPU at the
    # context it loaded, worked out from its files. Set when it runs on a GPU.
    # Next to `gpu_memory`, it shows how close the estimate came.
    memory: HfMemoryEstimate | None = None
    # Read from Windows a moment after it's ready. Null elsewhere.
    gpu_memory: GpuMemory | None = None


class ModelServer(BaseModel):
    """A llama-server process running one downloaded model."""

    id: str
    status: str = "starting"  # "starting" | "ready" | "stopping" | "stopped" | "failed"
    repo_id: str
    quantization: str
    vision_encoder: str | None = None
    model_path: str
    vision_encoder_path: str | None = None
    runtime_id: str
    backend: str
    # The device it was given (`ROCm1`), `none` for the CPU only, or null to let
    # llama.cpp choose.
    device: str | None = None
    settings: ServerSettings = Field(default_factory=ServerSettings)
    gpu_plan: GpuPlan | None = None
    host: str
    port: int
    # llama.cpp's own chat page, on this machine only. The app serves the model
    # at its own `/v1`, which other machines can reach too.
    base_url: str
    # What to send as `model` to the app's `/v1` API.
    model_name: str
    loaded: LoadedModel | None = None
    # Things that work but not as well as they should, such as memory that
    # didn't fit in VRAM.
    warnings: list[str] = Field(default_factory=list)
    pid: int | None = None
    exit_code: int | None = None
    error: str | None = None
    log_path: str
    created_at: datetime
    ready_at: datetime | None = None
    stopped_at: datetime | None = None


class ServerLogs(BaseModel):
    id: str
    path: str
    lines: list[str] = Field(default_factory=list)

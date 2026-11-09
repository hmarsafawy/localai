from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ServerSettings(BaseModel):
    """What was asked for. Anything left unset is chosen by llama.cpp, which
    sizes it to the memory that's free when the model loads."""

    # Null: the model's full context, reduced if it wouldn't fit.
    context_size: int | None = None
    # "auto", "all", or a number of layers. 0 runs on the CPU only.
    gpu_layers: int | str = "auto"


class LoadedModel(BaseModel):
    """What llama-server reports once the model is loaded."""

    # The context each request can use.
    context_size: int | None = None
    # How many requests it serves at once.
    slots: int | None = None
    vision: bool = False
    build: str | None = None


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
    # The device it was given (`ROCm1`), or null to let llama.cpp choose.
    device: str | None = None
    settings: ServerSettings = Field(default_factory=ServerSettings)
    host: str
    port: int
    # llama.cpp's own chat page. Its OpenAI-compatible API is under `/v1`.
    base_url: str
    # What to send as `model` in API requests.
    model_name: str
    loaded: LoadedModel | None = None
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

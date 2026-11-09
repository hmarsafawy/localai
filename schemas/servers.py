from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from models.servers import ModelServer, ServerLogs
from schemas.downloads import OPTION_PATTERN, REPO_ID_PATTERN

RUNTIME_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"


class ServerLaunchRequest(BaseModel):
    """Only the repo is needed, and it has to be downloaded. `quantization` and
    `vision_encoder` take an id from `GET /library` or a quantization name. By
    default it runs the only quantization downloaded, with a vision encoder if
    one is; `vision_encoder` can also be `none`.

    Anything left unset, llama.cpp sizes to the memory free when the model loads:
    the model's full context, cut down if it wouldn't fit, and as many layers on
    the GPU as fit there."""

    # Without this, the docs page fills in each number's minimum, so trying it
    # out as given would load a model with a 512-token context. The optional
    # numbers are left out because FastAPI drops nulls from the docs anyway.
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "repo_id": "unsloth/Qwen3.8-27B-GGUF",
                    "quantization": "auto",
                    "vision_encoder": "auto",
                    "gpu_layers": "auto",
                }
            ]
        }
    )

    repo_id: str = Field(pattern=REPO_ID_PATTERN, max_length=200)
    quantization: str = Field(default="auto", pattern=OPTION_PATTERN, max_length=200)
    vision_encoder: str = Field(default="auto", pattern=OPTION_PATTERN, max_length=200)
    context_size: int | None = Field(default=None, ge=512, le=4_194_304)
    # 0 runs on the CPU only.
    gpu_layers: Annotated[int, Field(ge=0, le=10_000)] | Literal["auto", "all"] = "auto"
    # By default, the first free port from 8080.
    port: int | None = Field(default=None, ge=1024, le=65535)
    # By default, the active runtime.
    runtime_id: str | None = Field(default=None, pattern=RUNTIME_ID_PATTERN, max_length=200)


class ModelServerResponse(ModelServer):
    """Public response shape for a running (or recently stopped) model server."""

    model_config = ConfigDict(from_attributes=True)


class ServerLogsResponse(ServerLogs):
    model_config = ConfigDict(from_attributes=True)

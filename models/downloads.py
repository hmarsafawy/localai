from __future__ import annotations

from pydantic import BaseModel, Field


class LocalModelOption(BaseModel):
    """One downloaded (or partly downloaded) quantization or vision encoder."""

    id: str
    label: str
    quantization: str | None = None
    variant: str | None = None
    status: str = "incomplete"  # "complete" | "incomplete"
    # What llama-server is given. For a split model it's the first file, and it
    # finds the others next to it.
    path: str
    # Repo paths, which are also their paths inside the model's directory.
    files: list[str] = Field(default_factory=list)
    shards: int = 1
    size_bytes: int = 0
    bytes_on_disk: int = 0


class LocalModel(BaseModel):
    repo_id: str
    task: str
    directory: str
    size_on_disk_bytes: int = 0
    quantizations: list[LocalModelOption] = Field(default_factory=list)
    vision_encoders: list[LocalModelOption] = Field(default_factory=list)


class ModelLibrary(BaseModel):
    """Every model on this machine that the app downloaded."""

    models_dir: str
    free_bytes: int | None = None
    size_on_disk_bytes: int = 0
    models: list[LocalModel] = Field(default_factory=list)

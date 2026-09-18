from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HfModelArtifact(BaseModel):
    model_config = ConfigDict(extra="allow")

    filename: str
    size_bytes: int | None = Field(default=None, alias="size")
    lfs: dict[str, Any] | None = None


class HfModel(BaseModel):
    """Normalized Hugging Face model metadata used by the application."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    model_id: str
    author: str | None = None
    sha: str | None = None
    downloads: int = 0
    likes: int = 0
    trending_score: float | None = None
    private: bool = False
    gated: bool | str = False
    disabled: bool = False
    pipeline_tag: str | None = None
    library_name: str | None = None
    tags: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    last_modified: datetime | None = None
    created_at: datetime | None = None
    config: dict[str, Any] | None = None
    card_data: dict[str, Any] | None = None
    siblings: list[HfModelArtifact] = Field(default_factory=list)

    @field_validator("languages", "datasets", mode="before")
    @classmethod
    def normalize_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return list(value)


class HfPipelineFile(BaseModel):
    """A single file found inside a pipeline component's subfolder."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    filename: str
    size_bytes: int | None = Field(default=None, alias="size")


class HfPipelineComponent(BaseModel):
    """One entry of a diffusers `model_index.json` (e.g. `unet`, `vae`, `text_encoder`)."""

    model_config = ConfigDict(extra="allow")

    name: str
    library: str | None = None
    class_name: str | None = None
    required: bool = True
    subfolder: str
    files: list[HfPipelineFile] = Field(default_factory=list)
    formats: list[str] = Field(default_factory=list)
    config: dict[str, Any] | None = None


class HfPipelineInfo(BaseModel):
    """Normalized breakdown of what a diffusers pipeline repo needs to run."""

    model_config = ConfigDict(extra="allow")

    model_id: str
    class_name: str | None = None
    diffusers_version: str | None = None
    components: list[HfPipelineComponent] = Field(default_factory=list)
    missing_components: list[str] = Field(default_factory=list)
    extra_config: dict[str, Any] = Field(default_factory=dict)


class HfSuggestedSource(BaseModel):
    """A repo (and, when known, specific files) where a missing role can be sourced from."""

    model_config = ConfigDict(extra="allow")

    repo_id: str
    files: list[str] = Field(default_factory=list)
    note: str | None = None


class HfRequirementRole(BaseModel):
    """One thing a pipeline needs (a text encoder, a VAE, ...), regardless of
    whether the repo itself has a `model_index.json`."""

    model_config = ConfigDict(extra="allow")

    role: str
    architecture: str | None = None
    required: bool = True
    present: bool = False
    source: str = "repo"  # "repo" | "base_model" | "fingerprint" | "registry"
    confidence: str = "high"  # "high" | "low"
    subfolder: str | None = None
    files: list[str] = Field(default_factory=list)
    sources: list[HfSuggestedSource] = Field(default_factory=list)


class HfModelRequirements(BaseModel):
    """What a model needs to run, resolved without requiring a `model_index.json`:
    falls back through the repo's `base_model` ancestry, weight tensor-name
    fingerprinting, and a small curated table of known pipeline families."""

    model_config = ConfigDict(extra="allow")

    model_id: str
    family: str | None = None
    pipeline_class: str | None = None
    resolved_from: str | None = None
    roles: list[HfRequirementRole] = Field(default_factory=list)
    unidentified_files: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

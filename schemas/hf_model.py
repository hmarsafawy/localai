from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from models.hf_model import HfModel, HfPipelineInfo


class HfModelSearchRequest(BaseModel):
    search: str | None = Field(default=None, min_length=1, max_length=200)
    author: str | None = Field(default=None, min_length=1, max_length=100)
    task: str | None = Field(default=None, min_length=1, max_length=100)
    library: str | None = Field(default=None, min_length=1, max_length=100)
    tags: list[str] = Field(default_factory=list, max_length=20)
    languages: list[str] = Field(default_factory=list, max_length=20)
    gated: bool | None = None
    sort: str = Field(
        default="downloads",
        pattern="^(downloads|likes|trendingScore|lastModified|createdAt)$",
    )
    direction: int = Field(default=-1, ge=-1, le=1)
    limit: int = Field(default=20, ge=1, le=100)


class HfModelResponse(HfModel):
    """Public response shape for normalized Hugging Face model data."""

    model_config = ConfigDict(from_attributes=True)


class HfModelSearchResponse(BaseModel):
    items: list[HfModelResponse]
    limit: int
    has_more: bool


class HfPipelineResponse(HfPipelineInfo):
    """Public response shape for a parsed diffusers pipeline breakdown."""

    model_config = ConfigDict(from_attributes=True)

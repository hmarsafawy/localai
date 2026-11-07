from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from models.downloads import ModelLibrary

_REPO_NAME = r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9_-])?"
# `owner/name`, or a bare `name` for the Hub's oldest repos. These become folder
# names, so anything that could step out of a folder or get renamed by Windows
# (a trailing dot) is refused here.
REPO_ID_PATTERN = rf"^{_REPO_NAME}(?:/{_REPO_NAME})?$"
# An option's id (`noMTP-Q4_K_M`), a quantization (`Q4_K_M`), `auto` or `none`.
OPTION_PATTERN = r"^[A-Za-z0-9._-]+$"


class ModelDownloadRequest(BaseModel):
    """Only the repo is needed. `quantization` and `vision_encoder` take an option
    `id` from `GET /models/{repo}/requirements`, or just a quantization name. By
    default each is the recommended one; `vision_encoder` can also be `none`."""

    repo_id: str = Field(pattern=REPO_ID_PATTERN, max_length=200)
    quantization: str = Field(default="auto", pattern=OPTION_PATTERN, max_length=200)
    vision_encoder: str = Field(default="auto", pattern=OPTION_PATTERN, max_length=200)


class ModelLibraryResponse(ModelLibrary):
    """Public response shape for the models on this machine."""

    model_config = ConfigDict(from_attributes=True)

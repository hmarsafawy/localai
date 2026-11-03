from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from models.jobs import JobInfo
from models.runtime import AvailableBuilds, InstalledRuntime, InstalledRuntimes

VERSION_PATTERN = r"^(stable|latest|v\d+(\.\d+){1,3}|b\d+)$"
# "auto", a backend ("vulkan"), or a backend at a version ("cuda-12.4").
BACKEND_PATTERN = r"^[a-z0-9][a-z0-9.\-]*$"


class RuntimeInstallRequest(BaseModel):
    """All optional, so an empty request is the one-click install: the stable
    release, in the best build for this machine, switched to when it's ready."""

    version: str = Field(default="stable", pattern=VERSION_PATTERN)
    backend: str = Field(default="auto", pattern=BACKEND_PATTERN, max_length=60)
    activate: bool = True


class RuntimeActivateRequest(BaseModel):
    id: str = Field(min_length=1, max_length=200)


class InstalledRuntimeResponse(InstalledRuntime):
    model_config = ConfigDict(from_attributes=True)


class InstalledRuntimesResponse(InstalledRuntimes):
    """Public response shape for the runtimes on this machine."""

    model_config = ConfigDict(from_attributes=True)


class AvailableBuildsResponse(AvailableBuilds):
    """Public response shape for the builds of a release that can run here."""

    model_config = ConfigDict(from_attributes=True)


class JobResponse(JobInfo):
    """Public response shape for a background job's progress."""

    model_config = ConfigDict(from_attributes=True)

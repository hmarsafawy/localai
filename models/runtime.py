from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class RuntimeDevice(BaseModel):
    """A device as a runtime reports it from `--list-devices`."""

    # The name to pass to `--device`. Numbering differs per backend: the same
    # card can be `Vulkan0` in one build and `ROCm1` in another.
    id: str
    name: str
    memory_total_bytes: int | None = None


class InstalledRuntime(BaseModel):
    # The release asset's name, which is also the folder name: a tag alone isn't
    # unique, because every tag ships builds for many backends.
    id: str
    tag: str
    build: int
    # The stable release this build is, when known (b11960 is v0.8.0).
    stable_version: str | None = None
    os: str
    arch: str
    backend: str
    backend_version: str | None = None
    path: str
    # The executable, once found. Not always directly in `path`: a Linux tarball
    # extracted by hand keeps its `llama-bNNNN/` wrapper folder.
    server_path: str | None = None
    # Whether it can run on this OS and CPU at all.
    compatible: bool
    active: bool = False
    # Whether it ran here and, for a GPU build, found a GPU. Null if not checked,
    # which is always the case for a build for another OS.
    verified: bool | None = None
    verification_error: str | None = None
    version: str | None = None
    commit: str | None = None
    devices: list[RuntimeDevice] = Field(default_factory=list)
    # Which of `devices` to run on: the discrete GPU if there is one.
    gpu_device: str | None = None
    installed_at: datetime | None = None
    verified_at: datetime | None = None


class RuntimeUpdates(BaseModel):
    stable_tag: str | None = None
    stable_version: str | None = None
    stable_is_newer: bool | None = None
    latest_tag: str | None = None
    latest_is_newer: bool | None = None


class InstalledRuntimes(BaseModel):
    runtimes_dir: str
    active_id: str | None = None
    runtimes: list[InstalledRuntime] = Field(default_factory=list)
    # Null if GitHub couldn't be reached; the installed list doesn't depend on it.
    update_available: RuntimeUpdates | None = None


class AvailableBuild(BaseModel):
    id: str
    tag: str
    backend: str
    backend_version: str | None = None
    arch: str
    size_bytes: int
    # An extra asset that's installed with this build, such as CUDA's DLLs.
    companion_asset: str | None = None
    # Position in the automatic order, or null if `auto` would never choose it.
    rank: int | None = None
    recommended: bool = False
    reason: str
    installed: bool = False


class AvailableBuilds(BaseModel):
    requested: str
    tag: str
    stable_version: str | None = None
    published_at: datetime | None = None
    os: str
    arch: str
    builds: list[AvailableBuild] = Field(default_factory=list)

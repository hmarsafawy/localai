from __future__ import annotations

from pydantic import BaseModel, Field


class OsInfo(BaseModel):
    system: str  # "windows" | "linux" | "macos"
    release: str | None = None
    version: str | None = None
    arch: str  # "x64" | "arm64" | the raw machine name for anything else
    is_wsl: bool = False


class CpuFeatures(BaseModel):
    avx: bool | None = None
    avx2: bool | None = None
    avx512: bool | None = None
    neon: bool | None = None


class CpuInfo(BaseModel):
    name: str | None = None
    vendor: str | None = None
    physical_cores: int | None = None
    logical_cores: int | None = None
    max_frequency_mhz: int | None = None
    features: CpuFeatures = Field(default_factory=CpuFeatures)


class MemoryInfo(BaseModel):
    total_bytes: int
    available_bytes: int
    used_bytes: int
    percent_used: float
    swap_total_bytes: int = 0
    swap_used_bytes: int = 0
    module_count: int | None = None
    speed_mhz: int | None = None
    type: str | None = None  # "DDR5", "DDR4", ...


class GpuApis(BaseModel):
    """What this GPU can actually be driven through on this machine."""

    # Highest CUDA version the installed driver supports, from nvidia-smi.
    cuda: str | None = None
    rocm: bool = False
    vulkan: bool = False
    metal: bool = False


class GpuInfo(BaseModel):
    name: str
    vendor: str  # "nvidia" | "amd" | "intel" | "apple" | "unknown"
    # "discrete" | "integrated" | "unified" (Apple Silicon: VRAM is system RAM).
    # Integrated GPUs never count as VRAM: backends report them as owning most
    # of the system's memory, which is shared rather than dedicated.
    kind: str
    vendor_id: str | None = None
    device_id: str | None = None
    vram_total_bytes: int | None = None
    vram_used_bytes: int | None = None
    vram_free_bytes: int | None = None
    driver_version: str | None = None
    apis: GpuApis = Field(default_factory=GpuApis)


class StorageVolume(BaseModel):
    mount: str
    label: str | None = None
    filesystem: str | None = None
    total_bytes: int
    used_bytes: int
    free_bytes: int
    media: str = "unknown"  # "nvme" | "ssd" | "hdd" | "unknown"
    bus: str | None = None
    disk_name: str | None = None
    # The volume holding the app's data folder (runtimes, models).
    is_data_drive: bool = False


class HardwareInfo(BaseModel):
    os: OsInfo
    cpu: CpuInfo
    memory: MemoryInfo
    gpus: list[GpuInfo] = Field(default_factory=list)
    storage: list[StorageVolume] = Field(default_factory=list)
    data_dir: str
    # A probe that failed: its fields are null and the reason is listed here.
    warnings: list[str] = Field(default_factory=list)

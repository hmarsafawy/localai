"""What this machine has to run models with.

Facts that don't change while the app runs (CPU, GPU list, disk types) take
several seconds to gather on Windows, so they're cached and loaded in the
background at startup. Anything that moves (free RAM, VRAM in use, free disk)
is read on every call.

Each probe fails on its own: a missing value is null plus a `warnings` entry, so
one broken tool never takes down the whole endpoint.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
from ctypes.util import find_library
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil

from models.hardware import (
    CpuFeatures,
    CpuInfo,
    GpuApis,
    GpuInfo,
    HardwareInfo,
    MemoryInfo,
    OsInfo,
    StorageVolume,
)
from services.model_options import MemoryBudget
from services.settings import data_dir

logger = logging.getLogger(__name__)

VENDORS_BY_PCI_ID = {
    0x10DE: "nvidia",
    0x1002: "amd",
    0x1022: "amd",
    0x8086: "intel",
    0x106B: "apple",
    0x5143: "qualcomm",
}
# Microsoft's PCI id is the Basic Render Driver and other software adapters.
MICROSOFT_PCI_ID = 0x1414

# Below this a GPU's "dedicated" memory is a BIOS carve-out from system RAM.
MIN_DISCRETE_VRAM_BYTES = 2 * 1024**3

DXGI_ADAPTER_FLAG_SOFTWARE = 2
PF_AVX_INSTRUCTIONS_AVAILABLE = 39
PF_AVX2_INSTRUCTIONS_AVAILABLE = 40
PF_AVX512F_INSTRUCTIONS_AVAILABLE = 41

# SMBIOS memory type codes.
MEMORY_TYPES = {24: "DDR3", 26: "DDR4", 27: "LPDDR", 28: "LPDDR2", 29: "LPDDR3", 30: "LPDDR4", 34: "DDR5", 35: "LPDDR5"}
# `Get-PhysicalDisk` reports names, but the underlying CIM class reports codes.
DISK_MEDIA_CODES = {3: "HDD", 4: "SSD", 5: "SCM"}
DISK_BUS_CODES = {7: "USB", 11: "SATA", 17: "NVMe", 8: "RAID"}

LINUX_IGNORED_FILESYSTEMS = {"squashfs", "tmpfs", "devtmpfs", "overlay", "iso9660"}
# How WSL mounts the Windows drives; psutil doesn't count them as physical disks.
WSL_WINDOWS_DRIVE = re.compile(r"^/mnt/[a-z]$")

_WINDOWS_STATIC_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$out = @{ errors = @() }
function Add-Section($name, [scriptblock]$probe) {
    try { $out[$name] = @(& $probe) } catch { $out.errors += ($name + ': ' + $_.Exception.Message) }
}
Add-Section 'cpu' { Get-CimInstance Win32_Processor | Select-Object Name,Manufacturer,MaxClockSpeed }
Add-Section 'memory' { Get-CimInstance Win32_PhysicalMemory | Select-Object Capacity,Speed,ConfiguredClockSpeed,SMBIOSMemoryType }
Add-Section 'video' { Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,PNPDeviceID }
Add-Section 'gpu_memory' { Get-CimInstance Win32_PerfRawData_GPUPerformanceCounters_GPUAdapterMemory | Select-Object Name,DedicatedUsage }
Add-Section 'partitions' { Get-Partition | Where-Object DriveLetter | Select-Object DriveLetter,DiskNumber }
Add-Section 'disks' { Get-PhysicalDisk | Select-Object DeviceId,FriendlyName,MediaType,BusType }
Add-Section 'volumes' { Get-Volume | Where-Object DriveLetter | Select-Object DriveLetter,FileSystemLabel }
$out | ConvertTo-Json -Depth 4 -Compress
"""

_WINDOWS_GPU_MEMORY_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
@(Get-CimInstance Win32_PerfRawData_GPUPerformanceCounters_GPUAdapterMemory | Select-Object Name,DedicatedUsage) | ConvertTo-Json -Compress
"""


@dataclass
class _GpuFacts:
    """A GPU's fixed facts plus what's needed to look up its live usage."""

    name: str
    vendor: str
    kind: str
    vendor_id: str | None = None
    device_id: str | None = None
    vram_total_bytes: int | None = None
    driver_version: str | None = None
    apis: GpuApis = field(default_factory=GpuApis)
    luid: str | None = None  # Windows: key into the GPU performance counters.
    sysfs_device: Path | None = None  # Linux AMD: where mem_info_vram_used lives.
    nvidia_smi_index: int | None = None


@dataclass
class _VolumeFacts:
    label: str | None = None
    media: str = "unknown"
    bus: str | None = None
    disk_name: str | None = None


@dataclass
class _StaticFacts:
    os: OsInfo
    cpu: CpuInfo
    gpus: list[_GpuFacts]
    volumes: dict[str, _VolumeFacts]  # by normalized mount point
    module_count: int | None
    memory_speed_mhz: int | None
    memory_type: str | None
    warnings: list[str]


def _run(args: list[str], timeout: float = 15) -> str | None:
    """A tool's stdout, or None if it's missing, failed, or timed out."""
    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            timeout=timeout,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.decode("utf-8", errors="replace")


def _powershell_json(script: str, timeout: float = 60) -> Any:
    # Encoded rather than passed as text so quoting can't mangle the script.
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    output = _run(["powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded], timeout)
    if not output:
        return None
    try:
        # Switching the console to UTF-8 can make PowerShell emit a BOM first.
        return json.loads(output.lstrip("﻿"))
    except ValueError:
        return None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def normalize_arch(machine: str) -> str:
    machine = machine.lower()
    if machine in ("amd64", "x86_64"):
        return "x64"
    if machine in ("arm64", "aarch64"):
        return "arm64"
    return machine


def classify_gpu(vendor: str, name: str, dedicated_bytes: int | None) -> str:
    """Whether a GPU has its own memory. By name first, because an APU's BIOS
    can reserve many GB of system RAM and report it as dedicated."""
    if vendor == "apple":
        return "unified"
    if vendor == "nvidia":
        return "discrete"
    if vendor == "amd":
        if re.search(r"\bRX\b|\bPro\b|\bWX\b|FirePro|\bR[579]\b|\bHD \d{4}\b", name):
            return "discrete"
        if re.search(r"Radeon(?:\(TM\)|™)?\s*(?:Graphics|Vega\b|\d{3,4}[MS]\b)", name, re.IGNORECASE):
            return "integrated"
    if vendor == "intel":
        return "discrete" if re.search(r"Arc(?:\(TM\)|™)?\s+[AB]\d{3}", name) else "integrated"
    if dedicated_bytes is None:
        return "integrated"
    return "discrete" if dedicated_bytes >= MIN_DISCRETE_VRAM_BYTES else "integrated"


def parse_cpuinfo(text: str) -> dict[str, Any]:
    """Name, vendor and instruction-set flags from `/proc/cpuinfo`."""
    fields: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            fields.setdefault(key.strip().lower(), value.strip())
    flags = set((fields.get("flags") or fields.get("features") or "").split())
    return {
        "name": fields.get("model name"),
        "vendor": fields.get("vendor_id"),
        "avx": "avx" in flags,
        "avx2": "avx2" in flags,
        "avx512": "avx512f" in flags,
        "neon": "neon" in flags or "asimd" in flags,
    }


def normalize_cpu_vendor(vendor: str | None) -> str | None:
    if not vendor:
        return None
    return {"authenticamd": "AMD", "genuineintel": "Intel"}.get(vendor.strip().lower(), vendor.strip())


def parse_nvidia_smi_csv(text: str) -> list[dict[str, Any]]:
    """`--query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader,nounits`"""
    gpus: list[dict[str, Any]] = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        gpus.append(
            {
                "name": parts[0],
                "total_bytes": _mib_to_bytes(parts[1]),
                "used_bytes": _mib_to_bytes(parts[2]),
                "driver_version": parts[3],
            }
        )
    return gpus


def _mib_to_bytes(value: str) -> int | None:
    try:
        return int(float(value) * 1024**2)
    except ValueError:
        return None


def parse_cuda_version(nvidia_smi_output: str) -> str | None:
    match = re.search(r"CUDA Version:\s*([\d.]+)", nvidia_smi_output)
    return match.group(1) if match else None


def rocm_sdk_directories() -> list[str]:
    """ROCm SDK folders in this Python environment or named by `ROCM_PATH`.
    llama.cpp's Windows ROCm builds find no GPU unless these are on `PATH`."""
    roots: list[Path] = []
    if os.environ.get("VIRTUAL_ENV"):
        roots.append(Path(os.environ["VIRTUAL_ENV"]))
    roots.append(Path(sys.prefix))
    directories: list[str] = []
    for root in roots:
        site_packages = root / "Lib" / "site-packages"
        for directory in (site_packages / "_rocm_sdk_core" / "bin", site_packages / "_rocm_sdk_libraries" / "bin"):
            if directory.is_dir() and str(directory) not in directories:
                directories.append(str(directory))
    rocm_path = os.environ.get("ROCM_PATH")
    if rocm_path and (Path(rocm_path) / "bin").is_dir():
        directories.append(str(Path(rocm_path) / "bin"))
    return directories


def rocm_runtime_available() -> bool:
    """Whether a ROCm build could see a GPU here. This only rules out the sure
    failures; installing a build still tests it with `--list-devices`."""
    if sys.platform == "win32":
        directories = rocm_sdk_directories()
        # The SDK's core package alone isn't enough: it was the libraries
        # package that made devices appear when tested on a real card.
        return bool(os.environ.get("ROCM_PATH")) or any("_rocm_sdk_libraries" in entry for entry in directories)
    if sys.platform == "linux":
        return os.path.exists("/dev/kfd")
    return False


def _vulkan_loader_available() -> bool:
    if sys.platform == "win32":
        return (Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "vulkan-1.dll").is_file()
    if sys.platform == "linux":
        return find_library("vulkan") is not None
    return False


def _dxgi_adapters() -> list[dict[str, Any]]:
    """Every graphics adapter Windows knows about. Unlike WMI's `AdapterRAM`,
    which is 32-bit and tops out at 4 GB, this reports real memory sizes."""
    import ctypes
    from ctypes import POINTER, byref, c_void_p, wintypes

    class Guid(ctypes.Structure):
        _fields_ = [
            ("data1", ctypes.c_uint32),
            ("data2", ctypes.c_uint16),
            ("data3", ctypes.c_uint16),
            ("data4", ctypes.c_ubyte * 8),
        ]

    class Luid(ctypes.Structure):
        _fields_ = [("low", wintypes.DWORD), ("high", wintypes.LONG)]

    class AdapterDesc1(ctypes.Structure):
        _fields_ = [
            ("description", ctypes.c_wchar * 128),
            ("vendor_id", wintypes.UINT),
            ("device_id", wintypes.UINT),
            ("subsys_id", wintypes.UINT),
            ("revision", wintypes.UINT),
            ("dedicated_video_memory", ctypes.c_size_t),
            ("dedicated_system_memory", ctypes.c_size_t),
            ("shared_system_memory", ctypes.c_size_t),
            ("luid", Luid),
            ("flags", wintypes.UINT),
        ]

    def method(instance: c_void_p, index: int, *argtypes: Any) -> Any:
        table = ctypes.cast(ctypes.cast(instance, POINTER(c_void_p))[0], POINTER(c_void_p))
        return ctypes.WINFUNCTYPE(ctypes.HRESULT, c_void_p, *argtypes)(table[index])

    def release(instance: c_void_p) -> None:
        table = ctypes.cast(ctypes.cast(instance, POINTER(c_void_p))[0], POINTER(c_void_p))
        ctypes.WINFUNCTYPE(ctypes.c_ulong, c_void_p)(table[2])(instance)

    # IDXGIFactory1
    iid = Guid(0x770AAE78, 0xF26F, 0x4DBA, (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87))
    factory = c_void_p()
    result = ctypes.windll.dxgi.CreateDXGIFactory1(byref(iid), byref(factory))
    if result != 0:
        raise OSError(f"CreateDXGIFactory1 failed ({result:#x})")

    adapters: list[dict[str, Any]] = []
    try:
        index = 0
        while True:
            adapter = c_void_p()
            try:
                method(factory, 12, wintypes.UINT, POINTER(c_void_p))(factory, index, byref(adapter))  # EnumAdapters1
            except OSError:  # DXGI_ERROR_NOT_FOUND: past the last adapter
                break
            try:
                desc = AdapterDesc1()
                method(adapter, 10, POINTER(AdapterDesc1))(adapter, byref(desc))  # GetDesc1
                adapters.append(
                    {
                        "name": desc.description,
                        "vendor_id": desc.vendor_id,
                        "device_id": desc.device_id,
                        "subsys_id": desc.subsys_id,
                        "dedicated_bytes": desc.dedicated_video_memory,
                        "luid": f"luid_0x{desc.luid.high & 0xFFFFFFFF:08x}_0x{desc.luid.low:08x}",
                        "software": bool(desc.flags & DXGI_ADAPTER_FLAG_SOFTWARE),
                    }
                )
            finally:
                release(adapter)
            index += 1
    finally:
        release(factory)
    return adapters


def parse_gpu_memory_counters(rows: Any) -> dict[str, int]:
    """LUID -> dedicated memory in use, from the GPU performance counters. An
    adapter can have several `phys_N` instances, so they're added up."""
    usage: dict[str, int] = {}
    for row in _as_list(rows):
        match = re.match(r"(luid_0x[0-9a-f]+_0x[0-9a-f]+)", str(row.get("Name", "")), re.IGNORECASE)
        if match:
            luid = match.group(1).lower()
            usage[luid] = usage.get(luid, 0) + int(row.get("DedicatedUsage") or 0)
    return usage


def select_real_adapters(adapters: list[dict[str, Any]], live_luids: set[str]) -> list[dict[str, Any]]:
    """Drops software adapters and duplicates. Windows can list one card twice
    under different LUIDs (seen alongside a virtual display driver); only the
    real one has performance counters. Two identical physical cards both have
    counters, so both are kept."""
    real = [
        adapter
        for adapter in adapters
        if not adapter["software"] and adapter["vendor_id"] != MICROSOFT_PCI_ID
    ]
    groups: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for adapter in real:
        groups.setdefault((adapter["vendor_id"], adapter["device_id"], adapter["subsys_id"]), []).append(adapter)
    selected: list[dict[str, Any]] = []
    for group in groups.values():
        live = [adapter for adapter in group if adapter["luid"] in live_luids]
        selected.extend(live or group[:1])
    return selected


def _driver_versions_by_pci_id(video_rows: Any) -> dict[tuple[int, int], str]:
    versions: dict[tuple[int, int], str] = {}
    for row in _as_list(video_rows):
        match = re.search(r"VEN_([0-9A-F]{4})&DEV_([0-9A-F]{4})", str(row.get("PNPDeviceID", "")), re.IGNORECASE)
        if match and row.get("DriverVersion"):
            versions[(int(match.group(1), 16), int(match.group(2), 16))] = row["DriverVersion"]
    return versions


def _disk_media(media_type: Any, bus_type: Any) -> tuple[str, str | None]:
    media = DISK_MEDIA_CODES.get(media_type, media_type) if isinstance(media_type, int) else media_type
    bus = DISK_BUS_CODES.get(bus_type, str(bus_type)) if isinstance(bus_type, int) else bus_type
    if isinstance(bus, str) and bus.lower() == "nvme":
        return "nvme", bus
    if isinstance(media, str) and media.lower() in ("ssd", "hdd"):
        return media.lower(), bus
    return "unknown", bus


def windows_volumes(probe: dict[str, Any]) -> dict[str, _VolumeFacts]:
    disks = {str(row.get("DeviceId")): row for row in _as_list(probe.get("disks"))}
    labels = {str(row.get("DriveLetter")): row.get("FileSystemLabel") for row in _as_list(probe.get("volumes"))}
    volumes: dict[str, _VolumeFacts] = {}
    for partition in _as_list(probe.get("partitions")):
        letter = str(partition.get("DriveLetter"))
        disk = disks.get(str(partition.get("DiskNumber")), {})
        media, bus = _disk_media(disk.get("MediaType"), disk.get("BusType"))
        volumes[_mount_key(f"{letter}:\\")] = _VolumeFacts(
            label=labels.get(letter) or None, media=media, bus=bus, disk_name=disk.get("FriendlyName")
        )
    return volumes


def _mount_key(mount: str) -> str:
    return os.path.normcase(os.path.normpath(mount))


def _linux_volume_facts(device: str) -> _VolumeFacts:
    """Media type of a partition's parent disk, from sysfs."""
    name = Path(device).name
    block = Path("/sys/class/block") / name
    try:
        resolved = block.resolve()
        parent = resolved.parent if (resolved / "partition").exists() else resolved
        if parent.name.startswith("nvme"):
            return _VolumeFacts(media="nvme", bus="NVMe", disk_name=parent.name)
        rotational = (parent / "queue" / "rotational").read_text().strip()
    except OSError:
        return _VolumeFacts()
    return _VolumeFacts(media="hdd" if rotational == "1" else "ssd", disk_name=parent.name)


def linux_sysfs_gpus(drm_root: Path = Path("/sys/class/drm")) -> list[_GpuFacts]:
    """GPUs from sysfs. NVIDIA cards are left to nvidia-smi, which knows more."""
    gpus: list[_GpuFacts] = []
    for card in sorted(drm_root.glob("card[0-9]*")):
        if "-" in card.name:  # `card0-HDMI-A-1`: a connector, not a card
            continue
        device = card / "device"
        try:
            vendor_id = int((device / "vendor").read_text().strip(), 16)
            device_id = int((device / "device").read_text().strip(), 16)
        except (OSError, ValueError):
            continue
        vendor = VENDORS_BY_PCI_ID.get(vendor_id, "unknown")
        total = _read_int(device / "mem_info_vram_total")
        name = _linux_gpu_name(device) or f"{vendor.upper()} GPU {device_id:#06x}"
        gpus.append(
            _GpuFacts(
                name=name,
                vendor=vendor,
                kind=classify_gpu(vendor, name, total),
                vendor_id=f"{vendor_id:#06x}",
                device_id=f"{device_id:#06x}",
                vram_total_bytes=total,
                sysfs_device=device,
            )
        )
    return gpus


def usable_partitions(partitions: list[Any]) -> list[Any]:
    """Volumes something could be stored on, once each. Linux can mount a device
    in several places (WSL binds its system disk under `/usr` and `/mnt/wslg`), so
    only the shortest mount of each counts, and read-only ones can't hold a model."""
    by_device: dict[str, Any] = {}
    for partition in partitions:
        options = partition.opts.split(",")
        if "cdrom" in options or "ro" in options or not partition.fstype:
            continue
        if partition.fstype.lower() in LINUX_IGNORED_FILESYSTEMS or "/loop" in partition.device:
            continue
        current = by_device.get(partition.device)
        if current is None or len(partition.mountpoint) < len(current.mountpoint):
            by_device[partition.device] = partition
    return list(by_device.values())


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _linux_gpu_name(device: Path) -> str | None:
    try:
        slot = re.search(r"PCI_SLOT_NAME=(\S+)", (device / "uevent").read_text())
    except OSError:
        return None
    if slot is None:
        return None
    output = _run(["lspci", "-s", slot.group(1)], timeout=5)
    if not output or ": " not in output:
        return None
    return output.splitlines()[0].split(": ", 1)[1].strip()


class HardwareService:
    def __init__(self) -> None:
        self._static: _StaticFacts | None = None
        self._lock = threading.Lock()

    def preload(self) -> None:
        """Gathers the slow facts ahead of the first request."""
        threading.Thread(target=self._static_facts, name="hardware-preload", daemon=True).start()

    def get_hardware(self, refresh: bool = False) -> HardwareInfo:
        static = self._static_facts(refresh)
        warnings = list(static.warnings)
        memory = self._read_memory(static)
        gpus = self._read_gpus(static, warnings)
        return HardwareInfo(
            os=static.os,
            cpu=static.cpu,
            memory=memory,
            gpus=gpus,
            storage=self._read_storage(static, warnings),
            data_dir=str(data_dir()),
            warnings=warnings,
        )

    def memory_budget(self) -> MemoryBudget | None:
        """What a model can use right now, or None if that can't be read. A
        failure here must not fail whoever is only after a hint."""
        try:
            return budget_from_hardware(self.get_hardware())
        except Exception:
            logger.exception("Could not read the hardware for a memory estimate")
            return None

    def _static_facts(self, refresh: bool = False) -> _StaticFacts:
        # Held while probing so a request arriving during the startup preload
        # waits for it instead of starting a second slow probe.
        with self._lock:
            if self._static is None or refresh:
                self._static = self._gather_static()
            return self._static

    def _gather_static(self) -> _StaticFacts:
        warnings: list[str] = []
        os_info = OsInfo(
            system={"win32": "windows", "darwin": "macos"}.get(sys.platform, sys.platform),
            release=platform.release() or None,
            version=platform.version() or None,
            arch=normalize_arch(platform.machine()),
            is_wsl="microsoft" in platform.release().lower(),
        )
        if sys.platform == "win32":
            return self._gather_windows(os_info, warnings)
        return self._gather_posix(os_info, warnings)

    def _gather_windows(self, os_info: OsInfo, warnings: list[str]) -> _StaticFacts:
        probe = _powershell_json(_WINDOWS_STATIC_SCRIPT)
        if not isinstance(probe, dict):
            warnings.append("PowerShell hardware probe failed: CPU, RAM module and disk type details are missing")
            probe = {}
        warnings.extend(str(error) for error in _as_list(probe.get("errors")))

        cpu_row = next(iter(_as_list(probe.get("cpu"))), {})
        cpu = CpuInfo(
            name=(cpu_row.get("Name") or "").strip() or platform.processor() or None,
            vendor=normalize_cpu_vendor(cpu_row.get("Manufacturer")),
            physical_cores=psutil.cpu_count(logical=False),
            logical_cores=psutil.cpu_count(logical=True),
            max_frequency_mhz=cpu_row.get("MaxClockSpeed"),
            features=self._windows_cpu_features(warnings),
        )

        modules = _as_list(probe.get("memory"))
        first_module = modules[0] if modules else {}
        return _StaticFacts(
            os=os_info,
            cpu=cpu,
            gpus=self._windows_gpus(probe, warnings),
            volumes=windows_volumes(probe),
            module_count=len(modules) or None,
            memory_speed_mhz=first_module.get("ConfiguredClockSpeed") or first_module.get("Speed") or None,
            memory_type=MEMORY_TYPES.get(first_module.get("SMBIOSMemoryType")),
            warnings=warnings,
        )

    @staticmethod
    def _windows_cpu_features(warnings: list[str]) -> CpuFeatures:
        import ctypes

        try:
            present = ctypes.windll.kernel32.IsProcessorFeaturePresent
            return CpuFeatures(
                avx=bool(present(PF_AVX_INSTRUCTIONS_AVAILABLE)),
                avx2=bool(present(PF_AVX2_INSTRUCTIONS_AVAILABLE)),
                avx512=bool(present(PF_AVX512F_INSTRUCTIONS_AVAILABLE)),
            )
        except OSError as error:
            warnings.append(f"Could not read CPU features: {error}")
            return CpuFeatures()

    def _windows_gpus(self, probe: dict[str, Any], warnings: list[str]) -> list[_GpuFacts]:
        try:
            adapters = _dxgi_adapters()
        except OSError as error:
            warnings.append(f"Could not list GPUs through DXGI: {error}")
            return []
        live_luids = set(parse_gpu_memory_counters(probe.get("gpu_memory")))
        drivers = _driver_versions_by_pci_id(probe.get("video"))
        cuda_version = self._nvidia_cuda_version()
        rocm = rocm_runtime_available()
        vulkan = _vulkan_loader_available()

        gpus: list[_GpuFacts] = []
        for adapter in select_real_adapters(adapters, live_luids):
            vendor = VENDORS_BY_PCI_ID.get(adapter["vendor_id"], "unknown")
            gpus.append(
                _GpuFacts(
                    name=adapter["name"],
                    vendor=vendor,
                    kind=classify_gpu(vendor, adapter["name"], adapter["dedicated_bytes"]),
                    vendor_id=f"{adapter['vendor_id']:#06x}",
                    device_id=f"{adapter['device_id']:#06x}",
                    vram_total_bytes=adapter["dedicated_bytes"],
                    driver_version=drivers.get((adapter["vendor_id"], adapter["device_id"])),
                    apis=GpuApis(
                        cuda=cuda_version if vendor == "nvidia" else None,
                        rocm=rocm and vendor == "amd",
                        vulkan=vulkan,
                    ),
                    luid=adapter["luid"],
                )
            )
        return self._discrete_first(gpus)

    @staticmethod
    def _discrete_first(gpus: list[_GpuFacts]) -> list[_GpuFacts]:
        return sorted(gpus, key=lambda gpu: (gpu.kind != "discrete", -(gpu.vram_total_bytes or 0)))

    @staticmethod
    def _nvidia_cuda_version() -> str | None:
        if shutil.which("nvidia-smi") is None:
            return None
        output = _run(["nvidia-smi"])
        return parse_cuda_version(output) if output else None

    def _gather_posix(self, os_info: OsInfo, warnings: list[str]) -> _StaticFacts:
        gpus: list[_GpuFacts] = []
        if sys.platform == "darwin":
            cpu = self._mac_cpu()
            if os_info.arch == "arm64":
                gpus.append(
                    _GpuFacts(
                        name=cpu.name or "Apple GPU",
                        vendor="apple",
                        kind="unified",
                        vram_total_bytes=psutil.virtual_memory().total,
                        apis=GpuApis(metal=True),
                    )
                )
        else:
            cpu = self._linux_cpu(warnings)
            gpus = self._linux_gpus(warnings)
        return _StaticFacts(
            os=os_info,
            cpu=cpu,
            gpus=self._discrete_first(gpus),
            volumes={},
            module_count=None,
            memory_speed_mhz=None,
            memory_type=None,
            warnings=warnings,
        )

    @staticmethod
    def _linux_cpu(warnings: list[str]) -> CpuInfo:
        try:
            info = parse_cpuinfo(Path("/proc/cpuinfo").read_text(errors="replace"))
        except OSError as error:
            warnings.append(f"Could not read /proc/cpuinfo: {error}")
            info = {}
        frequency = psutil.cpu_freq()
        return CpuInfo(
            name=info.get("name") or platform.processor() or None,
            vendor=normalize_cpu_vendor(info.get("vendor")),
            physical_cores=psutil.cpu_count(logical=False),
            logical_cores=psutil.cpu_count(logical=True),
            max_frequency_mhz=int(frequency.max) if frequency and frequency.max else None,
            features=CpuFeatures(
                avx=info.get("avx"), avx2=info.get("avx2"), avx512=info.get("avx512"), neon=info.get("neon")
            ),
        )

    @staticmethod
    def _mac_cpu() -> CpuInfo:
        output = _run(["sysctl", "-n", "machdep.cpu.brand_string"], timeout=5)
        return CpuInfo(
            name=output.strip() if output else platform.processor() or None,
            vendor="Apple" if platform.machine() == "arm64" else None,
            physical_cores=psutil.cpu_count(logical=False),
            logical_cores=psutil.cpu_count(logical=True),
            features=CpuFeatures(neon=True) if platform.machine() == "arm64" else CpuFeatures(),
        )

    def _linux_gpus(self, warnings: list[str]) -> list[_GpuFacts]:
        rocm = rocm_runtime_available()
        vulkan = _vulkan_loader_available()
        gpus = [gpu for gpu in linux_sysfs_gpus() if gpu.vendor != "nvidia"]
        for gpu in gpus:
            gpu.apis = GpuApis(rocm=rocm and gpu.vendor == "amd", vulkan=vulkan)

        if shutil.which("nvidia-smi") is not None:
            cuda_version = self._nvidia_cuda_version()
            rows = self._nvidia_smi_rows()
            if rows is None:
                warnings.append("nvidia-smi is installed but could not list GPUs")
            for index, row in enumerate(rows or []):
                gpus.append(
                    _GpuFacts(
                        name=row["name"],
                        vendor="nvidia",
                        kind="discrete",
                        vendor_id="0x10de",
                        vram_total_bytes=row["total_bytes"],
                        driver_version=row["driver_version"],
                        apis=GpuApis(cuda=cuda_version, vulkan=vulkan),
                        nvidia_smi_index=index,
                    )
                )
        if not gpus and os.path.exists("/dev/dxg"):
            warnings.append(
                "WSL passes the GPU through /dev/dxg, which doesn't say what the GPU is, so none is listed. "
                "It may still be usable, but builds will be ranked as if there were none."
            )
        return gpus

    @staticmethod
    def _nvidia_smi_rows() -> list[dict[str, Any]] | None:
        output = _run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,driver_version",
                "--format=csv,noheader,nounits",
            ]
        )
        return parse_nvidia_smi_csv(output) if output is not None else None

    @staticmethod
    def _read_memory(static: _StaticFacts) -> MemoryInfo:
        virtual = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return MemoryInfo(
            total_bytes=virtual.total,
            available_bytes=virtual.available,
            used_bytes=virtual.used,
            percent_used=virtual.percent,
            swap_total_bytes=swap.total,
            swap_used_bytes=swap.used,
            module_count=static.module_count,
            speed_mhz=static.memory_speed_mhz,
            type=static.memory_type,
        )

    def _read_gpus(self, static: _StaticFacts, warnings: list[str]) -> list[GpuInfo]:
        windows_usage: dict[str, int] = {}
        if any(gpu.luid for gpu in static.gpus):
            rows = _powershell_json(_WINDOWS_GPU_MEMORY_SCRIPT, timeout=20)
            if rows is None:
                warnings.append("Could not read GPU performance counters: VRAM in use is unknown")
            windows_usage = parse_gpu_memory_counters(rows)
        nvidia_rows = (
            self._nvidia_smi_rows() if any(gpu.nvidia_smi_index is not None for gpu in static.gpus) else None
        ) or []

        gpus: list[GpuInfo] = []
        for gpu in static.gpus:
            used: int | None = None
            if gpu.luid is not None:
                used = windows_usage.get(gpu.luid)
            elif gpu.nvidia_smi_index is not None and gpu.nvidia_smi_index < len(nvidia_rows):
                used = nvidia_rows[gpu.nvidia_smi_index]["used_bytes"]
            elif gpu.sysfs_device is not None:
                used = _read_int(gpu.sysfs_device / "mem_info_vram_used")
            elif gpu.kind == "unified":
                used = psutil.virtual_memory().used
            free = max(gpu.vram_total_bytes - used, 0) if gpu.vram_total_bytes is not None and used is not None else None
            gpus.append(
                GpuInfo(
                    name=gpu.name,
                    vendor=gpu.vendor,
                    kind=gpu.kind,
                    vendor_id=gpu.vendor_id,
                    device_id=gpu.device_id,
                    vram_total_bytes=gpu.vram_total_bytes,
                    vram_used_bytes=used,
                    vram_free_bytes=free,
                    driver_version=gpu.driver_version,
                    apis=gpu.apis,
                )
            )
        return gpus

    def _read_storage(self, static: _StaticFacts, warnings: list[str]) -> list[StorageVolume]:
        try:
            partitions = psutil.disk_partitions(all=False)
            if static.os.is_wsl:
                partitions += [
                    partition
                    for partition in psutil.disk_partitions(all=True)
                    if WSL_WINDOWS_DRIVE.match(partition.mountpoint)
                ]
        except OSError as error:
            warnings.append(f"Could not list drives: {error}")
            return []

        volumes: list[StorageVolume] = []
        for partition in usable_partitions(partitions):
            try:
                usage = psutil.disk_usage(partition.mountpoint)
            except OSError:  # an empty card reader, an offline network share, ...
                continue
            facts = static.volumes.get(_mount_key(partition.mountpoint))
            # WSL's disks are virtual and report as spinning whatever they're
            # stored on, so what they say about themselves means nothing.
            if facts is None and sys.platform == "linux" and not static.os.is_wsl:
                facts = _linux_volume_facts(partition.device)
            facts = facts or _VolumeFacts()
            volumes.append(
                StorageVolume(
                    mount=partition.mountpoint,
                    label=facts.label,
                    filesystem=partition.fstype or None,
                    total_bytes=usage.total,
                    used_bytes=usage.used,
                    free_bytes=usage.free,
                    media=facts.media,
                    bus=facts.bus,
                    disk_name=facts.disk_name,
                )
            )

        data_volume = _volume_containing(data_dir(), volumes)
        if data_volume is not None:
            data_volume.is_data_drive = True
        return volumes


def _volume_containing(path: Path, volumes: list[StorageVolume]) -> StorageVolume | None:
    containing = [volume for volume in volumes if path.is_relative_to(volume.mount)]
    return max(containing, key=lambda volume: len(volume.mount), default=None)


def budget_from_hardware(hardware: HardwareInfo) -> MemoryBudget:
    """Sizes a model against the single best GPU. llama.cpp can split across
    several, but that isn't something to promise from an estimate."""
    ram = hardware.memory.available_bytes
    unified = next((gpu for gpu in hardware.gpus if gpu.kind == "unified"), None)
    if unified is not None:
        return MemoryBudget(has_gpu=True, vram_free_bytes=ram, ram_available_bytes=0)
    discrete = [gpu for gpu in hardware.gpus if gpu.kind == "discrete"]
    if not discrete:
        return MemoryBudget(has_gpu=False, vram_free_bytes=None, ram_available_bytes=ram)
    # Unknown usage (no counters) falls back to the card's total.
    best = max(
        (gpu.vram_free_bytes if gpu.vram_free_bytes is not None else gpu.vram_total_bytes) or 0 for gpu in discrete
    )
    return MemoryBudget(has_gpu=True, vram_free_bytes=best or None, ram_available_bytes=ram)


hardware_service = HardwareService()

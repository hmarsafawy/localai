from __future__ import annotations

from collections import namedtuple
from pathlib import Path

import pytest

from models.hardware import CpuInfo, GpuInfo, HardwareInfo, MemoryInfo, OsInfo
from services.hardware import (
    budget_from_hardware,
    classify_gpu,
    linux_sysfs_gpus,
    normalize_arch,
    normalize_cpu_vendor,
    parse_cpuinfo,
    parse_cuda_version,
    parse_gpu_memory_counters,
    parse_nvidia_smi_csv,
    parse_process_gpu_memory,
    select_real_adapters,
    usable_partitions,
    windows_volumes,
)

GIB = 1024**3


@pytest.mark.parametrize(
    ("vendor", "name", "dedicated", "expected"),
    [
        ("amd", "AMD Radeon RX 9070 XT", 16 * GIB, "discrete"),
        ("amd", "AMD Radeon(TM) Graphics", 512 * 1024**2, "integrated"),
        # An APU's BIOS can reserve this much, which isn't what makes a card discrete.
        ("amd", "AMD Radeon(TM) Graphics", 16 * GIB, "integrated"),
        ("amd", "AMD Radeon 780M Graphics", 8 * GIB, "integrated"),
        ("amd", "AMD Radeon RX 6800M", 12 * GIB, "discrete"),
        ("amd", "AMD Radeon(TM) Vega 8 Graphics", 2 * GIB, "integrated"),
        ("nvidia", "NVIDIA GeForce RTX 3080", 10 * GIB, "discrete"),
        ("intel", "Intel(R) UHD Graphics 770", GIB, "integrated"),
        ("intel", "Intel(R) Arc(TM) A770 Graphics", 16 * GIB, "discrete"),
        ("intel", "Intel(R) Arc(TM) Graphics", 8 * GIB, "integrated"),
        ("apple", "Apple M3", None, "unified"),
        ("unknown", "Some GPU", 8 * GIB, "discrete"),
        ("unknown", "Some GPU", GIB, "integrated"),
        ("unknown", "Some GPU", None, "integrated"),
    ],
)
def test_classify_gpu(vendor: str, name: str, dedicated: int | None, expected: str) -> None:
    assert classify_gpu(vendor, name, dedicated) == expected


def adapter(name: str, vendor_id: int, device_id: int, luid: str, software: bool = False) -> dict:
    return {
        "name": name,
        "vendor_id": vendor_id,
        "device_id": device_id,
        "subsys_id": 1,
        "dedicated_bytes": 16 * GIB,
        "luid": luid,
        "software": software,
    }


def test_select_real_adapters_on_a_real_machine() -> None:
    """What DXGI reports on a PC with a dGPU, an iGPU and a virtual display."""
    adapters = [
        adapter("AMD Radeon RX 9070 XT", 0x1002, 0x7550, "luid_0x00000000_0x00018e31"),
        adapter("AMD Radeon(TM) Graphics", 0x1002, 0x13C0, "luid_0x00000000_0x0001b94a"),
        adapter("AMD Radeon RX 9070 XT", 0x1002, 0x7550, "luid_0x00000000_0x000272da"),
        adapter("Microsoft Basic Render Driver", 0x1414, 0x8C, "luid_0x00000000_0x0001b919", software=True),
    ]
    live = {"luid_0x00000000_0x00018e31", "luid_0x00000000_0x0001b94a", "luid_0x00000000_0x0001b919"}
    selected = select_real_adapters(adapters, live)
    assert [(item["name"], item["luid"]) for item in selected] == [
        ("AMD Radeon RX 9070 XT", "luid_0x00000000_0x00018e31"),
        ("AMD Radeon(TM) Graphics", "luid_0x00000000_0x0001b94a"),
    ]


def test_two_identical_cards_are_both_kept() -> None:
    adapters = [
        adapter("NVIDIA GeForce RTX 4090", 0x10DE, 1, "luid_a"),
        adapter("NVIDIA GeForce RTX 4090", 0x10DE, 1, "luid_b"),
    ]
    assert len(select_real_adapters(adapters, {"luid_a", "luid_b"})) == 2


def test_a_duplicate_is_dropped_even_without_counters() -> None:
    adapters = [
        adapter("NVIDIA GeForce RTX 4090", 0x10DE, 1, "luid_a"),
        adapter("NVIDIA GeForce RTX 4090", 0x10DE, 1, "luid_b"),
    ]
    assert len(select_real_adapters(adapters, set())) == 1


def test_parse_gpu_memory_counters_normalizes_and_adds_up() -> None:
    rows = [
        {"Name": "luid_0x00000000_0x00018E31_phys_0", "DedicatedUsage": 3540107264},
        {"Name": "luid_0x00000000_0x00018E31_phys_1", "DedicatedUsage": 10},
        {"Name": "something else", "DedicatedUsage": 99},
    ]
    assert parse_gpu_memory_counters(rows) == {"luid_0x00000000_0x00018e31": 3540107274}


def test_parse_gpu_memory_counters_handles_a_single_row_and_nothing() -> None:
    # PowerShell unwraps a one-item list.
    assert parse_gpu_memory_counters({"Name": "luid_0x0_0x1_phys_0", "DedicatedUsage": 5}) == {"luid_0x0_0x1": 5}
    assert parse_gpu_memory_counters(None) == {}


def test_one_process_gpu_memory_counts_only_the_gpu_asked_about() -> None:
    """Real rows for llama-server: it also opens the integrated GPU (0x1B8A7)."""
    rows = [
        {
            "Name": "pid_20044_luid_0x00000000_0x00018D8C_phys_0",
            "DedicatedUsage": 8_224_000_000,
            "SharedUsage": 2_910_000_000,
        },
        {
            "Name": "pid_20044_luid_0x00000000_0x0001B8A7_phys_0",
            "DedicatedUsage": 150_000_000,
            "SharedUsage": 816_000_000,
        },
    ]
    assert parse_process_gpu_memory(rows, {"luid_0x00000000_0x00018d8c"}) == (8_224_000_000, 2_910_000_000)
    single = {"Name": "pid_1_luid_0x0_0x2_phys_0", "DedicatedUsage": 5, "SharedUsage": None}
    assert parse_process_gpu_memory(single, {"luid_0x0_0x2"}) == (5, 0)
    assert parse_process_gpu_memory(rows, {"luid_0x0_0x9"}) is None
    assert parse_process_gpu_memory(None, {"luid_0x0_0x2"}) is None


def test_parse_nvidia_smi_csv() -> None:
    rows = parse_nvidia_smi_csv("NVIDIA GeForce RTX 3080, 10240, 512, 572.16\n")
    assert rows == [
        {
            "name": "NVIDIA GeForce RTX 3080",
            "total_bytes": 10240 * 1024**2,
            "used_bytes": 512 * 1024**2,
            "driver_version": "572.16",
        }
    ]


def test_parse_nvidia_smi_csv_skips_unavailable_memory() -> None:
    assert parse_nvidia_smi_csv("GPU, [N/A], [N/A], 1.0")[0]["total_bytes"] is None


def test_parse_cuda_version() -> None:
    banner = "| NVIDIA-SMI 572.16   Driver Version: 572.16   CUDA Version: 12.8  |"
    assert parse_cuda_version(banner) == "12.8"
    assert parse_cuda_version("no version here") is None


def test_parse_cpuinfo_x86() -> None:
    text = "vendor_id\t: AuthenticAMD\nmodel name\t: AMD Ryzen 7 9700X\nflags\t\t: fpu avx avx2 avx512f\n"
    info = parse_cpuinfo(text)
    assert info["name"] == "AMD Ryzen 7 9700X"
    assert info["avx"] and info["avx2"] and info["avx512"]
    assert not info["neon"]


def test_parse_cpuinfo_arm() -> None:
    info = parse_cpuinfo("Features\t: fp asimd sve\n")
    assert info["neon"]
    assert not info["avx2"]


@pytest.mark.parametrize(
    ("machine", "expected"), [("AMD64", "x64"), ("x86_64", "x64"), ("aarch64", "arm64"), ("arm64", "arm64"), ("s390x", "s390x")]
)
def test_normalize_arch(machine: str, expected: str) -> None:
    assert normalize_arch(machine) == expected


def test_normalize_cpu_vendor() -> None:
    assert normalize_cpu_vendor("AuthenticAMD") == "AMD"
    assert normalize_cpu_vendor("GenuineIntel") == "Intel"
    assert normalize_cpu_vendor(None) is None


def test_windows_volumes_on_a_real_machine() -> None:
    probe = {
        "partitions": [{"DriveLetter": "E", "DiskNumber": 1}, {"DriveLetter": "C", "DiskNumber": 2}],
        "disks": [
            {"DeviceId": "1", "FriendlyName": "ST4000DM004", "MediaType": "HDD", "BusType": "SATA"},
            {"DeviceId": "2", "FriendlyName": "WD Blue SN5000", "MediaType": "SSD", "BusType": "NVMe"},
        ],
        "volumes": [{"DriveLetter": "E", "FileSystemLabel": "Local Disk"}, {"DriveLetter": "C", "FileSystemLabel": ""}],
    }
    volumes = windows_volumes(probe)
    drive_c = next(facts for mount, facts in volumes.items() if mount.lower().startswith("c:"))
    drive_e = next(facts for mount, facts in volumes.items() if mount.lower().startswith("e:"))
    assert (drive_c.media, drive_c.bus, drive_c.label) == ("nvme", "NVMe", None)
    assert (drive_e.media, drive_e.disk_name, drive_e.label) == ("hdd", "ST4000DM004", "Local Disk")


def test_windows_volumes_understands_numeric_codes() -> None:
    probe = {
        "partitions": [{"DriveLetter": "C", "DiskNumber": 0}],
        "disks": [{"DeviceId": "0", "MediaType": 4, "BusType": 11}],
    }
    facts = next(iter(windows_volumes(probe).values()))
    assert (facts.media, facts.bus) == ("ssd", "SATA")


def write_card(root: Path, name: str, vendor: str, device: str, **files: str) -> None:
    folder = root / name / "device"
    folder.mkdir(parents=True)
    (folder / "vendor").write_text(vendor)
    (folder / "device").write_text(device)
    for filename, content in files.items():
        (folder / filename).write_text(content)


def test_linux_sysfs_gpus(tmp_path: Path) -> None:
    write_card(tmp_path, "card0", "0x1002\n", "0x7550\n", mem_info_vram_total=f"{16 * GIB}\n")
    write_card(tmp_path, "card1", "0x10de\n", "0x2206\n")
    # A connector, which looks like a card from its name but isn't one.
    (tmp_path / "card0-HDMI-A-1").mkdir()
    gpus = linux_sysfs_gpus(tmp_path)
    assert [(gpu.vendor, gpu.device_id) for gpu in gpus] == [("amd", "0x7550"), ("nvidia", "0x2206")]
    assert gpus[0].vram_total_bytes == 16 * GIB
    assert gpus[0].kind == "discrete"


def hardware_with(gpus: list[GpuInfo], ram: int = 32 * GIB) -> HardwareInfo:
    return HardwareInfo(
        os=OsInfo(system="windows", arch="x64"),
        cpu=CpuInfo(),
        memory=MemoryInfo(total_bytes=ram, available_bytes=ram, used_bytes=0, percent_used=0),
        gpus=gpus,
        data_dir=".",
    )


def test_budget_ignores_an_integrated_gpus_shared_memory() -> None:
    budget = budget_from_hardware(
        hardware_with(
            [
                GpuInfo(name="iGPU", vendor="amd", kind="integrated", vram_total_bytes=16 * GIB, vram_free_bytes=16 * GIB),
            ]
        )
    )
    assert budget.has_gpu is False
    assert budget.vram_free_bytes is None
    assert budget.ram_available_bytes == 32 * GIB


def test_budget_uses_the_best_discrete_gpu() -> None:
    budget = budget_from_hardware(
        hardware_with(
            [
                GpuInfo(name="small", vendor="nvidia", kind="discrete", vram_total_bytes=8 * GIB, vram_free_bytes=8 * GIB),
                GpuInfo(name="big", vendor="nvidia", kind="discrete", vram_total_bytes=24 * GIB, vram_free_bytes=20 * GIB),
                GpuInfo(name="iGPU", vendor="amd", kind="integrated", vram_total_bytes=64 * GIB, vram_free_bytes=64 * GIB),
            ]
        )
    )
    assert budget.has_gpu is True
    assert budget.vram_free_bytes == 20 * GIB


def test_budget_falls_back_to_total_when_usage_is_unknown() -> None:
    budget = budget_from_hardware(
        hardware_with([GpuInfo(name="gpu", vendor="nvidia", kind="discrete", vram_total_bytes=8 * GIB)])
    )
    assert budget.vram_free_bytes == 8 * GIB


def test_budget_on_unified_memory_counts_ram_once() -> None:
    budget = budget_from_hardware(hardware_with([GpuInfo(name="Apple M3", vendor="apple", kind="unified")]))
    assert budget.has_gpu is True
    assert budget.vram_free_bytes == 32 * GIB
    assert budget.ram_available_bytes == 0


Partition = namedtuple("Partition", "device mountpoint fstype opts")


def test_usable_partitions_on_real_wsl() -> None:
    """What psutil reports inside WSL: one system disk bound in several places,
    a read-only tools disk, and a Windows drive mounted over 9p."""
    partitions = [
        Partition("/dev/sdb", "/usr/src/linux-headers-6.18", "ext4", "ro,relatime"),
        Partition("/dev/sdb", "/usr/lib/linux-tools/6.18", "ext4", "ro,relatime"),
        Partition("/dev/sdd", "/mnt/wslg/distro", "ext4", "rw,relatime"),
        Partition("/dev/sdd", "/", "ext4", "rw,relatime"),
        Partition("C:\\", "/mnt/c", "9p", "rw,noatime"),
    ]
    assert [item.mountpoint for item in usable_partitions(partitions)] == ["/", "/mnt/c"]


def test_usable_partitions_on_windows() -> None:
    partitions = [
        Partition("C:\\", "C:\\", "NTFS", "rw,fixed"),
        Partition("E:\\", "E:\\", "NTFS", "rw,fixed"),
        Partition("F:\\", "F:\\", "", "cdrom"),
        Partition("G:\\", "G:\\", "UDF", "ro,cdrom"),
    ]
    assert [item.mountpoint for item in usable_partitions(partitions)] == ["C:\\", "E:\\"]


def test_usable_partitions_skips_snaps_and_pseudo_filesystems() -> None:
    partitions = [
        Partition("/dev/nvme0n1p2", "/", "ext4", "rw"),
        Partition("/dev/loop3", "/snap/core/1", "squashfs", "ro"),
        Partition("/dev/loop4", "/mnt/image", "ext4", "rw"),
        Partition("tmpfs", "/run", "tmpfs", "rw"),
    ]
    assert [item.mountpoint for item in usable_partitions(partitions)] == ["/"]

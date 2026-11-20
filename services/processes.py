"""Keeps the programs the app starts, such as llama-server or pip, from
outliving it. They're stopped when the app shuts down, and on Windows they're
also tied to the app's process, so they go even if the app is killed."""

from __future__ import annotations

import logging
import subprocess
import sys
import threading

logger = logging.getLogger(__name__)


class _WindowsJob:
    """A Windows job object that ends every process in it once the app's handle
    to it closes, which happens however the app exits, even when it's killed.
    Processes started by a process in it are in it too."""

    JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        class IoCounters(ctypes.Structure):
            _fields_ = [
                (name, ctypes.c_ulonglong)
                for name in (
                    "ReadOperationCount",
                    "WriteOperationCount",
                    "OtherOperationCount",
                    "ReadTransferCount",
                    "WriteTransferCount",
                    "OtherTransferCount",
                )
            ]

        class BasicLimits(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimits),
                ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        self._ctypes = ctypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        self._kernel32 = kernel32

        # Not inheritable, so a process in it can't keep its own job alive.
        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = self.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            handle, self.JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        # Deliberately never closed: closing it is what ends the processes.
        self._handle = handle

    def add(self, process: subprocess.Popen[bytes]) -> None:
        handle = int(process._handle)  # type: ignore[attr-defined]
        if not self._kernel32.AssignProcessToJobObject(self._handle, handle):
            raise self._ctypes.WinError(self._ctypes.get_last_error())


_job: _WindowsJob | None = None
_job_lock = threading.Lock()


def tie_to_app(process: subprocess.Popen[bytes]) -> None:
    """Makes the process end with the app. Only needed on Windows: elsewhere a
    killed app is rare enough to leave to the shutdown stopping everything."""
    global _job
    if sys.platform != "win32":
        return
    try:
        with _job_lock:
            if _job is None:
                _job = _WindowsJob()
        _job.add(process)
    except OSError as error:
        logger.warning("Process %s could outlive the app if it crashes: %s", process.pid, error)

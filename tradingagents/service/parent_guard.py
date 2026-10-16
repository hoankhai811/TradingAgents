"""Make service children die with the service parent process."""

from __future__ import annotations

import ctypes
import os
import signal
import sys


def _await_parent(gate, parent_pid):
    if sys.platform.startswith("linux"):
        if ctypes.CDLL(None).prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
            return False
    return gate.wait(5) and os.getppid() == parent_pid


def guarded_call(gate, parent_pid, target, args):
    """Do not start external reads until the parent owns child lifetime."""
    if _await_parent(gate, parent_pid):
        target(*args)


def guarded_run(gate, parent_pid, store, job_id, config):
    """Do not start paid work until the parent owns child lifetime."""
    if not _await_parent(gate, parent_pid):
        return
    from tradingagents.service.worker import run_job

    run_job(store, job_id, config)


class WindowsChildJob:
    def __init__(self):
        self.handle = None
        if os.name != "nt":
            if not sys.platform.startswith("linux"):
                raise OSError("parent-death protection is unavailable on this platform")
            return
        from ctypes import wintypes

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

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimits),
                ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes = (ctypes.c_void_p, ctypes.c_wchar_p)
        self.kernel.CreateJobObjectW.restype = ctypes.c_void_p
        self.kernel.SetInformationJobObject.argtypes = (
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        )
        self.kernel.SetInformationJobObject.restype = wintypes.BOOL
        self.kernel.AssignProcessToJobObject.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
        self.kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        self.kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
        self.kernel.CloseHandle.restype = wintypes.BOOL
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
        limits = ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.kernel.SetInformationJobObject(
            self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            code = ctypes.get_last_error()
            self.close()
            raise OSError(code, "SetInformationJobObject failed")

    def assign(self, process):
        if self.handle and not self.kernel.AssignProcessToJobObject(
            self.handle, process._popen._handle
        ):
            raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None

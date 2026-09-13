"""Bounded subprocess execution.

Children run without a shell, at below-normal priority, with a timeout, a capped
output buffer and cooperative cancellation. On Windows they are placed in a Job
Object so that (a) the CPU hard cap applies to the whole process tree and (b) the
entire tree dies when the job is closed - no orphaned test runners.
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field

_IS_WIN = sys.platform == "win32"
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
CREATE_NO_WINDOW = 0x08000000

# Agent-owned child processes, so the governor can reclaim them (pid -> info),
# and their job objects, so the governor can tighten CPU caps while they run.
OWNED: dict[int, dict] = {}
JOBS: dict[int, "_Job"] = {}
_owned_lock = threading.Lock()


def retune_running_jobs(cpu_rate_percent: float) -> int:
    """Apply a new CPU cap to every running child job. Returns how many were updated."""
    n = 0
    with _owned_lock:
        jobs = list(JOBS.values())
    for j in jobs:
        try:
            if j.handle and j.cpu_rate_percent != cpu_rate_percent and j.set_cpu_rate(cpu_rate_percent):
                n += 1
        except Exception:
            continue
    return n


def cap_current_process(cpu_rate_percent: float) -> bool:
    """Put this process (and every child it will start) under a CPU hard cap.

    Used by the API server so in-process inference (llama.cpp, embeddings) can never
    push the machine past the cap. No kill-on-close here: this job holds the server.
    """
    if not _IS_WIN:
        return False
    try:
        import ctypes
        from ctypes import wintypes

        job = _Job.__new__(_Job)
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        k.CreateJobObjectW.restype = wintypes.HANDLE
        k.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                              wintypes.DWORD]
        k.SetInformationJobObject.restype = wintypes.BOOL
        k.GetCurrentProcess.restype = wintypes.HANDLE
        k.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        k.AssignProcessToJobObject.restype = wintypes.BOOL

        class CPU_RATE(ctypes.Structure):
            _fields_ = [("ControlFlags", wintypes.DWORD), ("CpuRate", wintypes.DWORD)]

        job._k, job._ctypes, job._CPU_RATE = k, ctypes, CPU_RATE
        job.handle = k.CreateJobObjectW(None, None)
        job.cpu_capped, job.cpu_rate_percent = False, None
        if not job.handle or not job.set_cpu_rate(cpu_rate_percent):
            return False
        if not k.AssignProcessToJobObject(job.handle, k.GetCurrentProcess()):
            return False
        global SELF_JOB
        SELF_JOB = job
        return True
    except Exception:
        return False


SELF_JOB: "_Job | None" = None


@dataclass
class ProcResult:
    argv: list[str]
    exit_code: int | None
    output: str
    duration_ms: float
    timed_out: bool = False
    cancelled: bool = False
    truncated: bool = False
    cpu_cap_percent: float | None = None
    job_object: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and not self.cancelled

    def to_dict(self) -> dict:
        return {
            "argv": self.argv, "exit_code": self.exit_code, "output": self.output,
            "duration_ms": round(self.duration_ms, 1), "timed_out": self.timed_out,
            "cancelled": self.cancelled, "truncated": self.truncated,
            "cpu_cap_percent": self.cpu_cap_percent, "job_object": self.job_object,
        }


# ------------------------------------------------------------- windows job object

class _Job:
    """Minimal ctypes wrapper: CPU hard cap + optional memory cap + kill-on-close."""

    def __init__(self, cpu_rate_percent: float | None, memory_limit_mb: int | None):
        import ctypes
        from ctypes import wintypes

        self._k = k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        k.CreateJobObjectW.restype = wintypes.HANDLE
        k.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                              wintypes.DWORD]
        k.SetInformationJobObject.restype = wintypes.BOOL
        k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k.OpenProcess.restype = wintypes.HANDLE
        k.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        k.AssignProcessToJobObject.restype = wintypes.BOOL
        k.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        k.CloseHandle.argtypes = [wintypes.HANDLE]
        self.handle = self._k.CreateJobObjectW(None, None)
        if not self.handle:
            raise OSError(ctypes.get_last_error(), "CreateJobObject failed")

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class BASIC(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                        ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD),
                        ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t),
                        ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t),
                        ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class EXTENDED(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", IO_COUNTERS),
                        ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        class CPU_RATE(ctypes.Structure):
            _fields_ = [("ControlFlags", wintypes.DWORD), ("CpuRate", wintypes.DWORD)]

        ext = EXTENDED()
        flags = 0x00002000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if memory_limit_mb:
            flags |= 0x00000200  # JOB_OBJECT_LIMIT_JOB_MEMORY
            ext.JobMemoryLimit = int(memory_limit_mb) * 1024 * 1024
        ext.BasicLimitInformation.LimitFlags = flags
        if not self._k.SetInformationJobObject(self.handle, 9, ctypes.byref(ext), ctypes.sizeof(ext)):
            raise OSError(ctypes.get_last_error(), "SetInformationJobObject(extended) failed")

        self._ctypes, self._CPU_RATE = ctypes, CPU_RATE
        self.cpu_capped = False
        self.cpu_rate_percent: float | None = None
        if cpu_rate_percent:
            self.set_cpu_rate(cpu_rate_percent)

    def set_cpu_rate(self, percent: float) -> bool:
        """Hard cap for the whole job, as a percent of all logical CPUs. Adjustable live."""
        rate = self._CPU_RATE()
        rate.ControlFlags = 0x1 | 0x4  # ENABLE | HARD_CAP
        rate.CpuRate = max(1, min(10000, int(percent * 100)))
        ok = bool(self._k.SetInformationJobObject(
            self.handle, 15, self._ctypes.byref(rate), self._ctypes.sizeof(rate)))
        if ok:
            self.cpu_capped, self.cpu_rate_percent = True, percent
        return ok

    def assign(self, pid: int) -> bool:
        PROCESS_SET_QUOTA, PROCESS_TERMINATE = 0x0100, 0x0001
        h = self._k.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, pid)
        if not h:
            return False
        try:
            return bool(self._k.AssignProcessToJobObject(self.handle, h))
        finally:
            self._k.CloseHandle(h)

    def terminate(self) -> None:
        self._k.TerminateJobObject(self.handle, 1)

    def close(self) -> None:
        if self.handle:
            self._k.CloseHandle(self.handle)
            self.handle = None


def _kill_tree(pid: int) -> None:
    try:
        import psutil

        parent = psutil.Process(pid)
        for c in parent.children(recursive=True):
            try:
                c.kill()
            except psutil.Error:
                pass
        parent.kill()
    except Exception:
        pass


def _cap_output(chunks: list[str], limit: int) -> tuple[str, bool]:
    text = "".join(chunks)
    if len(text) <= limit:
        return text, False
    head = limit // 4
    return text[:head] + f"\n… [{len(text) - limit} chars omitted] …\n" + text[-(limit - head):], True


def run_process(
    argv: list[str],
    *,
    cwd: str,
    env: dict | None = None,
    timeout_s: float = 120.0,
    cpu_rate_percent: float | None = None,
    memory_limit_mb: int | None = None,
    cancel_event: threading.Event | None = None,
    max_output_chars: int = 40000,
    label: str = "",
) -> ProcResult:
    t0 = time.perf_counter()
    flags = (BELOW_NORMAL_PRIORITY_CLASS | CREATE_NO_WINDOW) if _IS_WIN else 0
    proc = subprocess.Popen(
        argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, creationflags=flags, text=True, encoding="utf-8",
        errors="replace",
    )
    job = None
    if _IS_WIN:
        try:
            job = _Job(cpu_rate_percent, memory_limit_mb)
            if not job.assign(proc.pid):
                job.close()
                job = None
        except Exception:
            job = None
    with _owned_lock:
        OWNED[proc.pid] = {"pid": proc.pid, "label": label or " ".join(argv[:3]),
                           "started": time.time(), "cwd": cwd}
        if job is not None:
            JOBS[proc.pid] = job

    chunks: list[str] = []
    total = [0]

    def _reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            chunks.append(line)
            total[0] += len(line)
            # Keep memory bounded on runaway output: drop the middle, keep head + tail.
            if total[0] > max_output_chars * 4 and len(chunks) > 400:
                del chunks[200:len(chunks) - 200]
                total[0] = sum(len(c) for c in chunks)

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()
    timed_out = cancelled = False
    try:
        while True:
            try:
                proc.wait(timeout=0.25)
                break
            except subprocess.TimeoutExpired:
                pass
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
            elif time.perf_counter() - t0 > timeout_s:
                timed_out = True
            if cancelled or timed_out:
                if job is not None:
                    job.terminate()
                _kill_tree(proc.pid)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                break
    finally:
        reader.join(timeout=2)
        with _owned_lock:
            OWNED.pop(proc.pid, None)
            JOBS.pop(proc.pid, None)
        if job is not None:
            capped = job.cpu_capped
            cpu_rate_percent = job.cpu_rate_percent or cpu_rate_percent
            job.close()  # kill-on-close reaps any grandchildren still alive
        else:
            capped = False
    output, truncated = _cap_output(chunks, max_output_chars)
    return ProcResult(
        argv=argv, exit_code=proc.returncode, output=output,
        duration_ms=(time.perf_counter() - t0) * 1000, timed_out=timed_out,
        cancelled=cancelled, truncated=truncated,
        cpu_cap_percent=cpu_rate_percent if capped else None, job_object=job is not None,
    )


def owned_processes() -> list[dict]:
    with _owned_lock:
        return list(OWNED.values())

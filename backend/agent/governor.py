"""Resource governor: measured CPU/RAM state that heavy work must be admitted by.

    NORMAL     system CPU < caution (60%)
    CAUTION    caution .. cap (60-70%)         reduce concurrency, smaller context
    HIGH_LOAD  > cap (70%)                     protection: no new heavy work starts,
                                               child CPU caps tightened, prefer tools
    EMERGENCY  > emergency (90%) or RAM low    stop starting work, reclaim agent RAM

Decisions use a short moving average (not one noisy sample) and hysteresis: a state
is left only after the load has been below its entry threshold minus a margin for
`recover_samples` consecutive samples. Every transition is timestamped, so the CPU
backtest can report activation latency and recovery time from real data.

Enforcement is layered and measured, not assumed:
  - heavy work calls `admit()` and waits (bounded) while load is high;
  - child processes run in Job Objects whose hard CPU cap follows the state;
  - LLM thread counts are derived from the cap;
  - the API server may place itself under a hard cap (`proc.cap_current_process`).
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Callable

import psutil

from backend.config import SETTINGS

NORMAL, CAUTION, HIGH_LOAD, EMERGENCY = "NORMAL", "CAUTION", "HIGH_LOAD", "EMERGENCY"
_RANK = {NORMAL: 0, CAUTION: 1, HIGH_LOAD: 2, EMERGENCY: 3}
HEAVY_KINDS = {"llm", "tests", "index", "embed", "build"}


@dataclass
class Sample:
    ts: float
    sys_cpu: float
    proc_cpu: float
    ram_avail_gb: float
    queue: int = 0
    llm_active: bool = False


@dataclass
class Admission:
    admitted: bool
    kind: str
    state: str
    waited_ms: float
    reason: str
    degrade: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def psutil_sampler() -> Callable[[], tuple[float, float, float]]:
    """Returns (system cpu %, this process + children cpu % of machine, RAM available GB)."""
    me = psutil.Process()
    ncpu = psutil.cpu_count() or 1
    psutil.cpu_percent(None)
    me.cpu_percent(None)
    child_prev: dict[int, psutil.Process] = {}

    def sample() -> tuple[float, float, float]:
        sys_cpu = psutil.cpu_percent(None)
        own = me.cpu_percent(None)
        try:
            kids = me.children(recursive=True)
        except psutil.Error:
            kids = []
        for k in kids:
            if k.pid not in child_prev:
                child_prev[k.pid] = k
                try:
                    k.cpu_percent(None)  # prime; counted from the next sample
                except psutil.Error:
                    pass
                continue
            try:
                own += child_prev[k.pid].cpu_percent(None)
            except psutil.Error:
                child_prev.pop(k.pid, None)
        alive = {k.pid for k in kids}
        for pid in list(child_prev):
            if pid not in alive:
                child_prev.pop(pid, None)
        return sys_cpu, min(100.0, own / ncpu), psutil.virtual_memory().available / 1e9

    return sample


class Governor:
    def __init__(self, sampler: Callable[[], tuple[float, float, float]] | None = None, *,
                 interval_s: float | None = None, window: int = 3, recover_samples: int = 3,
                 hysteresis: float = 5.0, persist: bool = True, clock=time.time):
        cfg = SETTINGS.agent
        self.caution, self.cap, self.emergency = cfg.cpu_caution_percent, cfg.cpu_cap_percent, cfg.cpu_emergency_percent
        self.ram_min_gb = cfg.ram_min_available_gb
        self.interval = interval_s if interval_s is not None else cfg.sampler_interval_s
        self._sampler = sampler
        self.window, self.recover_samples, self.hysteresis = window, recover_samples, hysteresis
        self.persist, self.clock = persist, clock
        self.logical_cpus = psutil.cpu_count() or 1
        self.state = NORMAL
        self.samples: deque[Sample] = deque(maxlen=900)
        self.transitions: deque[dict] = deque(maxlen=300)
        self.protection_events: deque[dict] = deque(maxlen=300)
        self._below = 0
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._heavy_running = 0
        self._queue = 0
        self.llm_active = False
        self._last_persist = 0.0
        self._last_child_cap: float | None = None

    # ----------------------------------------------------------- sampling loop
    def start(self) -> "Governor":
        if self._thread and self._thread.is_alive():
            return self
        if self._sampler is None:
            self._sampler = psutil_sampler()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="jla-governor", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval * 3 + 1)

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:
                pass
            self._stop.wait(self.interval)

    def tick(self, values: tuple[float, float, float] | None = None) -> Sample:
        """Take one sample (or ingest `values`) and update the state. Test-friendly."""
        if values is None:
            if self._sampler is None:
                self._sampler = psutil_sampler()
            values = self._sampler()
        sys_cpu, proc_cpu, ram = values
        s = Sample(self.clock(), round(float(sys_cpu), 1), round(float(proc_cpu), 1),
                   round(float(ram), 2), self._queue, self.llm_active)
        with self._cond:
            self.samples.append(s)
            self._update_state(s)
            self._cond.notify_all()
        self._apply_child_caps()
        if self.persist and s.ts - self._last_persist >= 5.0:
            self._persist(s)
        return s

    def smoothed_cpu(self) -> float:
        recent = list(self.samples)[-self.window:]
        return sum(x.sys_cpu for x in recent) / len(recent) if recent else 0.0

    def smoothed_own_cpu(self) -> float:
        recent = list(self.samples)[-self.window:]
        return sum(x.proc_cpu for x in recent) / len(recent) if recent else 0.0

    def _target_state(self, cpu: float, ram: float) -> str:
        if cpu >= self.emergency or ram < self.ram_min_gb:
            return EMERGENCY
        if cpu >= self.cap:
            return HIGH_LOAD
        if cpu >= self.caution:
            return CAUTION
        return NORMAL

    def _update_state(self, s: Sample) -> None:
        cpu = self.smoothed_cpu()
        # Entry reacts to the raw sample ("activate immediately when CPU exceeds the
        # cap"); exit uses the average plus hysteresis, so a spike cannot cause flapping.
        target = self._target_state(max(cpu, s.sys_cpu), s.ram_avail_gb)
        if _RANK[target] > _RANK[self.state]:
            self._transition(target, cpu, s)
            self._below = 0
            return
        if _RANK[target] < _RANK[self.state]:
            # Leave only when clearly below the entry line of the current state.
            entry = {CAUTION: self.caution, HIGH_LOAD: self.cap, EMERGENCY: self.emergency}[self.state]
            ram_ok = s.ram_avail_gb >= self.ram_min_gb * 1.25
            if cpu <= entry - self.hysteresis and ram_ok:
                self._below += 1
                if self._below >= self.recover_samples:
                    self._transition(target, cpu, s)
                    self._below = 0
            else:
                self._below = 0
        else:
            self._below = 0

    def _transition(self, new: str, cpu: float, s: Sample) -> None:
        old = self.state
        self.state = new
        ev = {"ts": s.ts, "from": old, "to": new, "cpu_avg": round(cpu, 1), "cpu_raw": s.sys_cpu,
              "ram_avail_gb": s.ram_avail_gb}
        self.transitions.append(ev)
        if _RANK[new] >= _RANK[HIGH_LOAD] > _RANK[old]:
            self.protection_events.append({"activated_at": s.ts, "state": new, "cpu": s.sys_cpu})
        if _RANK[new] < _RANK[HIGH_LOAD] <= _RANK[old] and self.protection_events:
            last = self.protection_events[-1]
            if "recovered_at" not in last:
                last["recovered_at"] = s.ts
                last["duration_s"] = round(s.ts - last["activated_at"], 2)
        if new == EMERGENCY and s.ram_avail_gb < self.ram_min_gb:
            threading.Thread(target=self._reclaim_agent_ram, daemon=True).start()

    def _reclaim_agent_ram(self) -> None:
        try:
            from backend.agent.ram import reclaim_agent_owned

            actions = reclaim_agent_owned()
            if actions:
                self.protection_events.append({"activated_at": self.clock(), "state": "RAM_RECLAIM",
                                               "actions": actions})
        except Exception:
            pass

    def _persist(self, s: Sample) -> None:
        self._last_persist = s.ts
        try:
            from backend.db import execute

            execute("INSERT INTO governor_samples(ts,sys_cpu,proc_cpu,ram_avail_gb,state,queue,llm_active) "
                    "VALUES(?,?,?,?,?,?,?)",
                    (s.ts, s.sys_cpu, s.proc_cpu, s.ram_avail_gb, self.state, s.queue, int(s.llm_active)))
            if int(s.ts) % 300 < 5:
                execute("DELETE FROM governor_samples WHERE ts < ?", (s.ts - 86400,))
        except Exception:
            pass

    # ----------------------------------------------------------- policy
    def other_load(self) -> float:
        """CPU used by everything that is not this server (percent of machine)."""
        return max(0.0, self.smoothed_cpu() - self.smoothed_own_cpu())

    def child_cpu_cap(self) -> float:
        """Hard cap for a new child process so that total stays under the cap."""
        headroom = self.cap - self.other_load() - 5.0
        if self.state == EMERGENCY:
            headroom = min(headroom, 10.0)
        elif self.state == HIGH_LOAD:
            headroom = min(headroom, 25.0)
        return round(max(10.0, min(self.cap - 5.0, headroom)), 1)

    def llm_threads(self) -> int:
        """Inference threads that keep the model's own CPU share under the cap."""
        share = max(10.0, min(self.cap - 5.0, self.cap - self.other_load() - 5.0))
        return max(2, min(self.logical_cpus - 2, math.floor(self.logical_cpus * share / 100.0)))

    def policy(self) -> dict:
        st = self.state
        return {
            "state": st,
            "cpu_avg": round(self.smoothed_cpu(), 1),
            "max_heavy_concurrency": {NORMAL: 2, CAUTION: 1, HIGH_LOAD: 1, EMERGENCY: 0}[st],
            "allow_llm": st in (NORMAL, CAUTION),
            "allow_background": st == NORMAL,
            "allow_embeddings": st == NORMAL,
            "context_scale": {NORMAL: 1.0, CAUTION: 0.75, HIGH_LOAD: 0.5, EMERGENCY: 0.5}[st],
            "max_tokens_scale": {NORMAL: 1.0, CAUTION: 0.8, HIGH_LOAD: 0.6, EMERGENCY: 0.5}[st],
            "prefer": "cheapest_reliable" if st in (NORMAL, CAUTION) else "deterministic",
            "child_cpu_cap": self.child_cpu_cap(),
            "llm_threads": self.llm_threads(),
        }

    def _apply_child_caps(self) -> None:
        cap = self.child_cpu_cap()
        if self._last_child_cap is None or abs(cap - self._last_child_cap) >= 5.0:
            self._last_child_cap = cap
            try:
                from backend.agent.proc import retune_running_jobs

                retune_running_jobs(cap)
            except Exception:
                pass

    # ----------------------------------------------------------- admission
    def admit(self, kind: str, *, critical: bool = False, timeout_s: float = 60.0,
              cancel_event: threading.Event | None = None) -> Admission:
        """Wait (bounded) until heavy work may start. Light work is always admitted."""
        t0 = time.perf_counter()
        if kind not in HEAVY_KINDS:
            return Admission(True, kind, self.state, 0.0, "light work is never throttled")
        with self._cond:
            self._queue += 1
            try:
                while True:
                    pol = self.policy()
                    st = self.state
                    limit = pol["max_heavy_concurrency"]
                    blocked_by_state = st in (HIGH_LOAD, EMERGENCY) and not (critical and st == HIGH_LOAD)
                    if not blocked_by_state and self._heavy_running < max(1, limit):
                        self._heavy_running += 1
                        waited = (time.perf_counter() - t0) * 1000
                        degrade = {"context_scale": pol["context_scale"],
                                   "max_tokens_scale": pol["max_tokens_scale"],
                                   "llm_threads": pol["llm_threads"]}
                        return Admission(True, kind, st, round(waited, 1),
                                         "admitted" if waited < 1 else f"admitted after waiting for {st} to clear",
                                         degrade)
                    elapsed = time.perf_counter() - t0
                    if elapsed >= timeout_s or (cancel_event is not None and cancel_event.is_set()):
                        reason = (f"deferred: system is {st} (cpu {pol['cpu_avg']}%)" if blocked_by_state
                                  else "deferred: heavy-work concurrency limit reached")
                        self.protection_events.append({"activated_at": self.clock(), "state": "DEFERRED",
                                                       "kind": kind, "reason": reason})
                        return Admission(False, kind, st, round(elapsed * 1000, 1), reason)
                    self._cond.wait(timeout=min(0.5, max(0.05, timeout_s - elapsed)))
            finally:
                self._queue -= 1

    def release(self, kind: str) -> None:
        if kind not in HEAVY_KINDS:
            return
        with self._cond:
            self._heavy_running = max(0, self._heavy_running - 1)
            self._cond.notify_all()

    class _Slot:
        def __init__(self, gov: "Governor", adm: Admission):
            self.gov, self.adm = gov, adm

        def __enter__(self) -> Admission:
            return self.adm

        def __exit__(self, *exc) -> None:
            if self.adm.admitted:
                self.gov.release(self.adm.kind)

    def slot(self, kind: str, **kw) -> "_Slot":
        """`with governor().slot('llm') as adm:` - releases automatically."""
        return Governor._Slot(self, self.admit(kind, **kw))

    # ----------------------------------------------------------- reporting
    def status(self, history: int = 120) -> dict:
        last = self.samples[-1] if self.samples else None
        return {
            "state": self.state, "running": self.running, "logical_cpus": self.logical_cpus,
            "thresholds": {"caution": self.caution, "cap": self.cap, "emergency": self.emergency,
                           "ram_min_gb": self.ram_min_gb},
            "last": asdict(last) if last else None,
            "policy": self.policy(),
            "heavy_running": self._heavy_running, "queue": self._queue,
            "history": [asdict(s) for s in list(self.samples)[-history:]],
            "transitions": list(self.transitions)[-50:],
            "protection_events": list(self.protection_events)[-50:],
        }


_instance: Governor | None = None
_inst_lock = threading.Lock()


def governor() -> Governor:
    global _instance
    with _inst_lock:
        if _instance is None:
            _instance = Governor()
        return _instance


def set_governor(g: Governor | None) -> None:
    """Swap the process-wide governor (tests inject a fake-sampled one)."""
    global _instance
    with _inst_lock:
        _instance = g

"""Shared result type. Every processor in the harness returns one of these."""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class ProcessorResult:
    """Uniform envelope: result + confidence + cost, from every layer."""

    result: Any
    confidence: float
    processor: str
    latency_ms: float
    layer: str = "deterministic"  # deterministic|nlp|ml|retrieval|llm|hybrid
    llm_used: bool = False
    llm_calls: int = 0
    tokens: int = 0
    error: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["latency_ms"] = round(self.latency_ms, 2)
        d["confidence"] = round(self.confidence, 4)
        return d


@contextmanager
def timed():
    """`with timed() as t: ...` then `t()` gives elapsed ms."""
    t0 = time.perf_counter()
    holder = {}

    def elapsed() -> float:
        return holder.get("ms", (time.perf_counter() - t0) * 1000)

    try:
        yield elapsed
    finally:
        holder["ms"] = (time.perf_counter() - t0) * 1000

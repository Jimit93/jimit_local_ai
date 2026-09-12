"""LLM adapter interface. Application code depends on this, never on a runtime."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    model: str
    runtime: str
    stop_reason: str = "stop"
    meta: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LocalLLM(ABC):
    """Minimal contract every local runtime must satisfy."""

    name: str = "base"
    runtime: str = "base"

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def loaded(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
    ) -> LLMResult:
        raise NotImplementedError

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        raise NotImplementedError

    def unload(self) -> None:
        return None

    def health(self) -> dict:
        """Cheap liveness probe; loads the model if needed."""
        t0 = time.perf_counter()
        try:
            r = self.generate("Reply with the single word: OK", max_tokens=5, temperature=0.0)
            return {
                "ok": bool(r.text.strip()),
                "model": self.name,
                "runtime": self.runtime,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "sample": r.text.strip()[:40],
            }
        except Exception as e:  # depends on local runtime state
            return {"ok": False, "model": self.name, "runtime": self.runtime, "error": str(e)}

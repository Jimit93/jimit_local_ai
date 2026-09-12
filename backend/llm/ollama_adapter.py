"""Ollama adapter. Used only when a suitable local Ollama model is present."""
from __future__ import annotations

import json
import time
import urllib.request

from backend.config import SETTINGS
from backend.llm.base import LLMResult, LocalLLM

BASE = "http://127.0.0.1:11434"


def list_ollama_models(timeout: float = 2.0) -> list[dict]:
    try:
        with urllib.request.urlopen(f"{BASE}/api/tags", timeout=timeout) as r:
            return json.loads(r.read()).get("models", [])
    except Exception:
        return []


class QwenOllamaAdapter(LocalLLM):
    runtime = "ollama"

    def __init__(self, model: str):
        self.name = model
        self._ok = False

    @property
    def loaded(self) -> bool:
        return self._ok

    def load(self) -> None:
        if self._ok:
            return
        names = {m.get("name", "") for m in list_ollama_models()}
        if self.name not in names:
            raise RuntimeError(f"ollama model not available: {self.name}")
        self._ok = True

    def count_tokens(self, text: str) -> int:
        # Ollama reports counts only after a call; approximate for budgeting.
        return max(1, len(text) // 4)

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
        payload = {
            "model": self.name,
            "prompt": prompt,
            "system": system or "",
            "stream": False,
            "options": {
                "temperature": SETTINGS.llm.temperature if temperature is None else temperature,
                "top_p": SETTINGS.llm.top_p if top_p is None else top_p,
                "num_predict": max_tokens if max_tokens is not None else SETTINGS.llm.max_tokens,
                "stop": stop or [],
            },
        }
        req = urllib.request.Request(
            f"{BASE}/api/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=SETTINGS.llm.timeout_s) as r:
            data = json.loads(r.read())
        return LLMResult(
            text=(data.get("response") or "").strip(),
            prompt_tokens=int(data.get("prompt_eval_count", 0)),
            completion_tokens=int(data.get("eval_count", 0)),
            latency_ms=(time.perf_counter() - t0) * 1000,
            model=self.name,
            runtime=self.runtime,
            stop_reason=data.get("done_reason", "stop"),
        )

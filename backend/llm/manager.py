"""Singleton, lazily-loaded LLM manager.

The model is loaded once per process, on first use, and reused for every task.
All LLM usage in the app goes through `llm()` so token/call accounting is central.
"""
from __future__ import annotations

import atexit
import threading

from backend.config import SETTINGS
from backend.llm.base import LLMResult, LocalLLM
from backend.llm.discovery import ModelFile, select_model

_lock = threading.Lock()
_instance: LocalLLM | None = None
_selected: ModelFile | None = None

# Process-wide usage counters (reset by the benchmark harness between runs).
USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "latency_ms": 0.0}


def reset_usage() -> None:
    USAGE.update({"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "latency_ms": 0.0})


def selected_model() -> ModelFile | None:
    return _selected


def llm(force_reload: bool = False) -> LocalLLM:
    """Return the process-wide LLM adapter, constructing it on first call."""
    global _instance, _selected
    with _lock:
        if _instance is not None and not force_reload:
            return _instance
        if SETTINGS.llm.model_path:
            from backend.llm.llamacpp_adapter import QwenLlamaCppAdapter

            _instance = QwenLlamaCppAdapter(SETTINGS.llm.model_path)
            return _instance
        mf = select_model(SETTINGS.llm.prefer_size)
        if mf is not None:
            from backend.llm.llamacpp_adapter import QwenLlamaCppAdapter

            _selected = mf
            _instance = QwenLlamaCppAdapter(mf.path)
            return _instance
        # Fall back to Ollama only if no local GGUF was discovered.
        from backend.llm.ollama_adapter import QwenOllamaAdapter, list_ollama_models

        names = [m.get("name", "") for m in list_ollama_models()]
        qwen = [n for n in names if "qwen" in n.lower()]
        if not (qwen or names):
            raise RuntimeError("No local Qwen GGUF found and Ollama is unavailable.")
        _instance = QwenOllamaAdapter((qwen or names)[0])
        return _instance


_busy_lock = threading.Lock()
_busy = {"llamacpp": 0}
_ollama_busy: dict[str, int] = {}


def _mark(key: str, delta: int, table: dict) -> None:
    with _busy_lock:
        table[key] = max(0, table.get(key, 0) + delta)
    try:
        from backend.agent.governor import governor

        governor().llm_active = any(_busy.values()) or any(_ollama_busy.values())
    except Exception:
        pass


def generate(prompt: str, **kw) -> LLMResult:
    """Single funnel for every LLM call so usage is always accounted."""
    _mark("llamacpp", +1, _busy)
    try:
        r = llm().generate(prompt, **kw)
    finally:
        _mark("llamacpp", -1, _busy)
    USAGE["calls"] += 1
    USAGE["prompt_tokens"] += r.prompt_tokens
    USAGE["completion_tokens"] += r.completion_tokens
    USAGE["latency_ms"] += r.latency_ms
    return r


def is_loaded() -> bool:
    return _instance is not None and _instance.loaded


def is_busy() -> bool:
    return _busy["llamacpp"] > 0


def ollama_busy(name: str) -> bool:
    return _ollama_busy.get(name, 0) > 0


def release() -> str | None:
    """Unload the llama.cpp model if it is idle (RAM reclaim). It reloads lazily on next use."""
    import gc

    with _lock:
        if _instance is None or not _instance.loaded or is_busy():
            return None
        name = _instance.name
        _instance.unload()
    gc.collect()
    return name


def warmup() -> dict:
    """Pay the cold-start cost at server start, not on the user's first request."""
    import time as _t

    out = {}
    t0 = _t.perf_counter()
    try:
        llm().load()
        out["llm_load_ms"] = round((_t.perf_counter() - t0) * 1000, 1)
        out["model"] = llm().name
    except Exception as e:
        out["llm_error"] = str(e)
    t0 = _t.perf_counter()
    try:
        from backend.retrieval.index import embed, index

        index().build(use_embeddings=True)
        # Vectors are cached in SQLite, so build() alone may never touch MiniLM.
        # Force it now: the first query otherwise pays ~20s of model load.
        embed(["warmup"])
        out["index_build_ms"] = round((_t.perf_counter() - t0) * 1000, 1)
    except Exception as e:
        out["index_error"] = str(e)
    return out


@atexit.register
def _release() -> None:
    """llama.cpp's destructor misbehaves during interpreter teardown; free first."""
    global _instance
    try:
        if _instance is not None:
            _instance.unload()
            _instance = None
    except Exception:
        pass

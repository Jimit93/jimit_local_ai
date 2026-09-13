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


def _finish(r: LLMResult, mode: str) -> LLMResult:
    """Never return a reply cut mid-sentence (or a half edit block) at the token limit."""
    from backend.harness.completion import finish

    r.text, meta = finish(r.text, r.stop_reason, mode)
    r.meta["completion"] = meta
    return r


def generate(prompt: str, *, finish: str = "prose", **kw) -> LLMResult:
    """Single funnel for every LLM call so usage is always accounted.

    `finish`: "prose" (default) trims a truncated reply to its last complete sentence,
    "edits" drops an unterminated edit block, "none" returns the raw text.
    """
    _mark("llamacpp", +1, _busy)
    try:
        r = _finish(llm().generate(prompt, **kw), finish)
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


# ------------------------------------------------------------ code-model tiers
#
# Coding models run in Ollama (its own process). They are constructed here and nowhere
# else. Tiers are resolved against what is actually installed, cheapest first.

CODE_USAGE: dict[str, dict] = {}
_code_adapters: dict = {}


class InsufficientRAM(RuntimeError):
    def __init__(self, model: str, report: dict):
        super().__init__(f"not enough free RAM for {model}: short by {report.get('shortfall_gb')} GB")
        self.model, self.report = model, report


def _tier_candidates() -> list[tuple[str, str]]:
    a = SETTINGS.agent
    return [("small", a.code_model_small), ("capable", a.code_model_capable),
            ("small", "qwen2.5-coder:0.5b"), ("capable", "deepseek-coder:6.7b-instruct-q4_K_M")]


def code_models() -> list[dict]:
    """Installed coding models by tier, with their on-disk size (≈ resident RAM)."""
    from backend.llm.ollama_adapter import list_ollama_models

    installed = {m.get("name"): m for m in list_ollama_models()}
    out, seen = [], set()
    for tier, name in _tier_candidates():
        m = installed.get(name) or installed.get(name + ":latest")
        if m and name not in seen:
            seen.add(name)
            out.append({"tier": tier, "name": m["name"], "size_gb": round((m.get("size") or 0) / 1e9, 2),
                        "preferred": name in (SETTINGS.agent.code_model_small, SETTINGS.agent.code_model_capable)})
    return out


def resolve_code_model(tier: str) -> dict | None:
    models = code_models()
    exact = [m for m in models if m["tier"] == tier]
    if exact:
        return exact[0]
    return models[0] if models else None


def _ensure_ram_for(model: dict, *, task_id: str | None = None) -> dict:
    from backend.agent import ram as RAM

    loaded = {(m.get("name") or m.get("model")) for m in RAM.ollama_loaded_models()}
    if model["name"] in loaded:
        return {"ok": True, "already_loaded": True}
    need = round(model["size_gb"] * 1.15 + SETTINGS.agent.ram_min_available_gb, 2)
    rep = RAM.reclaim(need, keep_models={model["name"]}, task_id=task_id)
    if not rep["ok"]:
        raise InsufficientRAM(model["name"], rep)
    return rep


def generate_code(prompt: str, *, tier: str = "small", system: str | None = None,
                  max_tokens: int = 700, temperature: float = 0.1, task_id: str | None = None,
                  keep_alive: str = "3m", finish: str = "prose") -> LLMResult:
    """Code generation through a tiered Ollama model, RAM-checked and CPU-thread-capped."""
    from backend.llm.ollama_adapter import QwenOllamaAdapter

    model = resolve_code_model(tier)
    if model is None:
        raise RuntimeError("no coding model is installed in Ollama")
    ram_report = _ensure_ram_for(model, task_id=task_id)
    ad = _code_adapters.get(model["name"])
    if ad is None:
        ad = _code_adapters[model["name"]] = QwenOllamaAdapter(model["name"])
    # A fixed thread count: changing num_thread between calls makes Ollama reload the model.
    threads = SETTINGS.llm.n_threads
    _mark(model["name"], +1, _ollama_busy)
    try:
        r = _finish(ad.generate(prompt, system=system, max_tokens=max_tokens, temperature=temperature,
                                options={"num_thread": threads, "num_ctx": 8192}, keep_alive=keep_alive), finish)
    finally:
        _mark(model["name"], -1, _ollama_busy)
    u = CODE_USAGE.setdefault(model["name"], {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "latency_ms": 0.0})
    u["calls"] += 1
    u["prompt_tokens"] += r.prompt_tokens
    u["completion_tokens"] += r.completion_tokens
    u["latency_ms"] += r.latency_ms
    USAGE["calls"] += 1
    USAGE["prompt_tokens"] += r.prompt_tokens
    USAGE["completion_tokens"] += r.completion_tokens
    USAGE["latency_ms"] += r.latency_ms
    r.meta.update({"tier": model["tier"], "requested_tier": tier, "threads": threads,
                   "ram": {k: ram_report.get(k) for k in ("already_loaded", "start_available_gb", "end_available_gb", "actions")
                           if k in ram_report}})
    return r


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

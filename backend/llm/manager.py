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


def _ensure_ram_general() -> dict:
    """Before the general llama.cpp model loads: free idle agent-owned RAM, re-select to fit.

    Found by the system backtest: with a code model resident in Ollama, the general-model
    fallback loaded a 4.4 GB GGUF chosen for RAM that had been free earlier, and free RAM
    fell to 0.01 GB.
    """
    if is_loaded():
        return {"already_loaded": True}
    from backend.agent import ram as RAM

    mf = select_model(SETTINGS.llm.prefer_size) if not SETTINGS.llm.model_path else None
    size_gb = (mf.total_size_bytes / 1e9) if mf else 1.5
    need = round(size_gb * 1.15 + SETTINGS.agent.ram_min_available_gb, 2)
    rep = RAM.reclaim(need, keep_models={"llama.cpp"}, use_allowlist=False)
    if SETTINGS.llm.prefer_size == "auto" and not SETTINGS.llm.model_path:
        chosen = select_model("auto")  # re-select with the RAM that is actually free now
        if _instance is None or (chosen and getattr(_instance, "model_path", None) != chosen.path):
            llm(force_reload=True)
        small = select_model("small")
        need_small = (small.total_size_bytes / 1e9) * 1.15 + SETTINGS.agent.ram_min_available_gb if small else 0
        if RAM.available_gb() < need_small:
            raise InsufficientRAM(small.name if small else "general model",
                                  {**rep, "shortfall_gb": round(need_small - RAM.available_gb(), 2)})
    return rep


def generate(prompt: str, *, finish: str = "prose", **kw) -> LLMResult:
    """Single funnel for every LLM call so usage is always accounted.

    `finish`: "prose" (default) trims a truncated reply to its last complete sentence,
    "edits" drops an unterminated edit block, "none" returns the raw text.
    """
    _ensure_ram_general()
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
    try:
        ram_report = _ensure_ram_for(model, task_id=task_id)
    except InsufficientRAM as e:
        # Degrade instead of failing: the largest installed code model that fits the RAM
        # that is actually free after reclaim. Verification downstream catches weak output.
        free = e.report.get("end_available_gb") or 0.0
        # Configured (preferred) models before legacy fallbacks, then larger first. Measured in the
        # system backtest: the older deepseek-coder:6.7b produced escaped-string edits where
        # qwen2.5-coder:3b did not, so a newer small model beats an older large one here.
        fits = [m for m in sorted(code_models(), key=lambda m: (not m.get("preferred"), -m["size_gb"]))
                if m["name"] != model["name"] and m["size_gb"] * 1.15 + SETTINGS.agent.ram_min_available_gb <= free]
        if not fits:
            raise
        degraded_from, model = model["name"], fits[0]
        ram_report = {**_ensure_ram_for(model, task_id=task_id), "degraded_from": degraded_from,
                      "free_gb": free, "shortfall_gb": e.report.get("shortfall_gb")}
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
                   "degraded_from": ram_report.get("degraded_from"),
                   "ram": {k: ram_report.get(k) for k in ("already_loaded", "start_available_gb", "end_available_gb", "actions",
                                                          "free_gb", "shortfall_gb") if k in ram_report}})
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

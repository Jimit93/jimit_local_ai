"""Milestone 2 verification: load the local Qwen model and run real inference."""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psutil  # noqa: E402

from backend.config import SETTINGS  # noqa: E402
from backend.llm.discovery import select_model  # noqa: E402
from backend.llm.llamacpp_adapter import QwenLlamaCppAdapter  # noqa: E402


def main() -> int:
    prefer = sys.argv[1] if len(sys.argv) > 1 else SETTINGS.llm.prefer_size
    mf = select_model(prefer)
    if mf is None:
        print("NO MODEL FOUND")
        return 1
    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    ad = QwenLlamaCppAdapter(mf.path, n_ctx=2048)
    t0 = time.perf_counter()
    ad.load()
    load_ms = (time.perf_counter() - t0) * 1000
    rss1 = proc.memory_info().rss
    r = ad.generate(
        "Hello, introduce yourself in two sentences.",
        system="You are a concise local assistant.",
        max_tokens=96,
        temperature=0.3,
    )
    rss2 = proc.memory_info().rss
    out = {
        "model": mf.name,
        "model_gb": round(mf.total_size_bytes / 1e9, 2),
        "quant": mf.quant,
        "load_ms": round(load_ms, 1),
        "gen_ms": round(r.latency_ms, 1),
        "prompt_tokens": r.prompt_tokens,
        "completion_tokens": r.completion_tokens,
        "tok_per_s": round(r.completion_tokens / (r.latency_ms / 1000), 2) if r.latency_ms else 0,
        "rss_after_load_mb": round((rss1 - rss0) / 1e6, 1),
        "rss_peak_mb": round(rss2 / 1e6, 1),
        "threads": ad.n_threads,
        "text": r.text,
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

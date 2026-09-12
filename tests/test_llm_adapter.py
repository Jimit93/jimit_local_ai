"""Adapter and discovery tests.

Model loading is slow (seconds to tens of seconds), so the expensive inference
test is opt-in via JLA_TEST_LLM=1 and skipped by default.
"""
import os

import pytest

from backend.llm import manager as M
from backend.llm.base import LLMResult, LocalLLM
from backend.llm.discovery import qwen_candidates, scan_models, select_model


def test_discovery_finds_local_gguf_models():
    models = scan_models()
    if not models:
        pytest.skip("no models on this machine")
    assert any(m.fmt == "gguf" for m in models)
    assert all(".venv" not in m.path and "site-packages" not in m.path for m in models)


def test_split_gguf_reports_combined_size():
    shards = [m for m in scan_models() if m.shard_total]
    if not shards:
        pytest.skip("no split model present")
    m = shards[0]
    assert m.total_size_bytes >= m.size_bytes
    # Only the first shard is a valid entry point.
    assert all(c.shard_index in (None, 1) for c in qwen_candidates())


def test_selection_respects_the_ram_budget():
    cands = qwen_candidates()
    if len(cands) < 2:
        pytest.skip("need two candidates to test selection")
    tiny = select_model("auto", available_ram_bytes=1)
    huge = select_model("auto", available_ram_bytes=10**12)
    assert tiny.total_size_bytes <= huge.total_size_bytes
    assert select_model("small").total_size_bytes <= select_model("large").total_size_bytes


def test_manager_is_a_singleton():
    a, b = M.llm(), M.llm()
    assert a is b
    assert isinstance(a, LocalLLM)


def test_usage_accounting_starts_clean():
    M.reset_usage()
    assert M.USAGE == {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                       "latency_ms": 0.0}


@pytest.mark.skipif(os.environ.get("JLA_TEST_LLM") != "1",
                    reason="set JLA_TEST_LLM=1 to run real inference (slow)")
def test_real_inference_and_accounting():
    M.reset_usage()
    r = M.generate("Reply with the single word: OK", max_tokens=5, temperature=0.0)
    assert isinstance(r, LLMResult)
    assert r.text.strip()
    assert r.completion_tokens > 0 and r.prompt_tokens > 0
    assert M.USAGE["calls"] == 1
    assert M.USAGE["completion_tokens"] == r.completion_tokens


@pytest.mark.skipif(os.environ.get("JLA_TEST_LLM") != "1",
                    reason="set JLA_TEST_LLM=1 to run real inference (slow)")
def test_health_check():
    h = M.llm().health()
    assert h["ok"] and h["runtime"] in ("llama.cpp", "ollama")

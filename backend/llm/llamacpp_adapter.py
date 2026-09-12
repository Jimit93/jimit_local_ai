"""llama-cpp-python adapter for local GGUF Qwen models."""
from __future__ import annotations

import os
import time

from backend.config import SETTINGS
from backend.llm.base import LLMResult, LocalLLM


class QwenLlamaCppAdapter(LocalLLM):
    runtime = "llama.cpp"

    def __init__(
        self,
        model_path: str,
        *,
        n_ctx: int | None = None,
        n_threads: int | None = None,
        n_batch: int | None = None,
        chat_format: str | None = None,
    ):
        self.model_path = model_path
        self.name = os.path.basename(model_path)
        self.n_ctx = n_ctx or SETTINGS.llm.n_ctx
        self.n_threads = n_threads or SETTINGS.llm.n_threads
        self.n_batch = n_batch or SETTINGS.llm.n_batch
        self.chat_format = chat_format
        self._llm = None
        self.load_time_ms: float | None = None

    @property
    def loaded(self) -> bool:
        return self._llm is not None

    def load(self) -> None:
        if self._llm is not None:
            return
        from llama_cpp import Llama

        t0 = time.perf_counter()
        kwargs = dict(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_batch=self.n_batch,
            seed=SETTINGS.llm.seed,
            verbose=False,
            use_mmap=True,
            use_mlock=False,
        )
        if self.chat_format:
            kwargs["chat_format"] = self.chat_format
        self._llm = Llama(**kwargs)
        self.load_time_ms = (time.perf_counter() - t0) * 1000

    def unload(self) -> None:
        self._llm = None

    def count_tokens(self, text: str) -> int:
        self.load()
        return len(self._llm.tokenize(text.encode("utf-8", "ignore"), add_bos=False))

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
        self.load()
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        t0 = time.perf_counter()
        out = self._llm.create_chat_completion(
            messages=msgs,
            max_tokens=max_tokens if max_tokens is not None else SETTINGS.llm.max_tokens,
            temperature=SETTINGS.llm.temperature if temperature is None else temperature,
            top_p=SETTINGS.llm.top_p if top_p is None else top_p,
            stop=stop or [],
        )
        dt = (time.perf_counter() - t0) * 1000
        choice = out["choices"][0]
        usage = out.get("usage", {}) or {}
        return LLMResult(
            text=(choice["message"].get("content") or "").strip(),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            latency_ms=dt,
            model=self.name,
            runtime=self.runtime,
            stop_reason=choice.get("finish_reason") or "stop",
        )

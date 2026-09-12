"""Discover local model files. Never assumes a filename."""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path

from backend.config import MODEL_SEARCH_DIR

WEIGHT_SUFFIXES = {".gguf", ".safetensors", ".onnx", ".bin", ".pt"}
SKIP_PARTS = {".venv", "venv", "site-packages", "node_modules", "__pycache__", ".git"}
_QUANT_RE = re.compile(r"(q\d[_\-][0-9kmsKMS_]*|f16|fp16|bf16|int8|q8_0)", re.I)
_SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.I)
_PARAM_RE = re.compile(r"(\d+(?:[._]\d+)?)\s*b(?![a-z])", re.I)


@dataclass
class ModelFile:
    path: str
    name: str
    fmt: str
    size_bytes: int
    quant: str | None
    family: str
    params_b: float | None
    shard_index: int | None
    shard_total: int | None
    total_size_bytes: int  # incl. sibling shards

    def to_dict(self) -> dict:
        return asdict(self)


def _family(name: str) -> str:
    low = name.lower()
    for fam in ("qwen", "llama", "mistral", "phi", "gemma", "minilm", "piper", "whisper"):
        if fam in low:
            return fam
    return "unknown"


def _params_b(name: str) -> float | None:
    m = _PARAM_RE.search(name)
    if not m:
        return None
    try:
        return float(m.group(1).replace("_", "."))
    except ValueError:
        return None


def scan_models(root: Path | str = MODEL_SEARCH_DIR, max_depth: int = 6) -> list[ModelFile]:
    """Walk `root` and return every model weight file that is not inside a venv."""
    root = Path(root)
    out: list[ModelFile] = []
    if not root.exists():
        return out
    root_depth = len(root.parts)
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in WEIGHT_SUFFIXES:
            continue
        if SKIP_PARTS & set(p.parts):
            continue
        if len(p.parts) - root_depth > max_depth:
            continue
        try:
            size = p.stat().st_size
        except OSError:
            continue
        if size < 5_000_000:  # ignore tiny aux weights
            continue
        shard = _SHARD_RE.search(p.name)
        si = int(shard.group(1)) if shard else None
        st = int(shard.group(2)) if shard else None
        total = size
        if shard:
            stem = p.name[: shard.start()]
            total = sum(
                q.stat().st_size for q in p.parent.glob(f"{stem}-*-of-{shard.group(2)}.gguf")
            )
        qm = _QUANT_RE.search(p.name)
        out.append(
            ModelFile(
                path=str(p),
                name=p.name,
                fmt=p.suffix.lower().lstrip("."),
                size_bytes=size,
                quant=qm.group(1).lower() if qm else None,
                family=_family(p.name),
                params_b=_params_b(p.name),
                shard_index=si,
                shard_total=st,
                total_size_bytes=total,
            )
        )
    return sorted(out, key=lambda m: -m.total_size_bytes)


def qwen_candidates(models: list[ModelFile] | None = None) -> list[ModelFile]:
    """Loadable Qwen GGUF entry points (first shard only for split models)."""
    models = models if models is not None else scan_models()
    return [
        m
        for m in models
        if m.family == "qwen" and m.fmt == "gguf" and (m.shard_index in (None, 1))
    ]


def select_model(prefer: str = "auto", available_ram_bytes: int | None = None) -> ModelFile | None:
    """Pick a Qwen GGUF. `auto` = largest model that fits comfortably in free RAM."""
    cands = qwen_candidates()
    if not cands:
        return None
    if prefer == "large":
        return cands[0]
    if prefer == "small":
        return cands[-1]
    if available_ram_bytes is None:
        try:
            import psutil

            available_ram_bytes = psutil.virtual_memory().available
        except Exception:
            return cands[-1]
    budget = available_ram_bytes * 0.80  # leave headroom for kv-cache + OS
    fits = [m for m in cands if m.total_size_bytes <= budget]
    return fits[0] if fits else cands[-1]

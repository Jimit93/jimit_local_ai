"""Central configuration. Paths and runtime settings only - no side effects on import."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("JLA_DATA_DIR", ROOT / "data"))
DB_PATH = Path(os.environ.get("JLA_DB_PATH", DATA_DIR / "agent.db"))
CONFIG_DIR = ROOT / "config"
REPORTS_DIR = ROOT / "reports"
DOCS_DIR = ROOT / "docs"
LOG_DIR = DATA_DIR / "logs"

# Where local GGUF/other model files live. Discovery scans this tree.
MODEL_SEARCH_DIR = Path(
    os.environ.get("JLA_MODEL_DIR", r"C:\Users\pc\Quantized_AI_agent\models")
)

# LinkedIn exports. Both the user-specified "Basic" export and any sibling
# export folders are inspected; only files that actually exist are used.
LINKEDIN_SOURCES = [
    p
    for p in (
        os.environ.get("JLA_LINKEDIN_SRC"),
        r"C:\Users\pc\Downloads\Basic_LinkedInDataExport_09-12-2026.zip",
        r"C:\Users\pc\Downloads\Basic_LinkedInDataExport_09-12-2026.zip.zip",
        r"C:\Users\pc\Downloads\Complete_LinkedInDataExport_09-06-2026.zip",
    )
    if p
]

EMBED_MODEL_NAME = os.environ.get("JLA_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


@dataclass
class LLMSettings:
    """Runtime knobs for the local LLM. Memory-aware defaults."""

    model_path: str | None = None          # explicit override
    prefer_size: str = os.environ.get("JLA_MODEL_PREFER", "auto")  # auto|small|large
    n_ctx: int = int(os.environ.get("JLA_N_CTX", 4096))
    n_threads: int = int(os.environ.get("JLA_THREADS", 0)) or max(1, (os.cpu_count() or 4) - 2)
    n_batch: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    timeout_s: float = float(os.environ.get("JLA_TIMEOUT", 300))
    seed: int = 42


@dataclass
class HarnessSettings:
    """Confidence thresholds used by the router/escalation logic."""

    deterministic_min_conf: float = 0.90
    nlp_min_conf: float = 0.70
    ml_min_conf: float = 0.60
    retrieval_min_conf: float = 0.35
    max_context_chars: int = 6000
    # Two capped examples carry the voice; a third mostly buys prompt-eval seconds.
    max_style_examples: int = 2
    style_example_max_chars: int = 700


@dataclass
class Settings:
    llm: LLMSettings = field(default_factory=LLMSettings)
    harness: HarnessSettings = field(default_factory=HarnessSettings)
    host: str = "127.0.0.1"       # local-only bind by default
    port: int = int(os.environ.get("JLA_PORT", 8077))


SETTINGS = Settings()


def ensure_dirs() -> None:
    for d in (DATA_DIR, LOG_DIR, REPORTS_DIR, DOCS_DIR, CONFIG_DIR):
        d.mkdir(parents=True, exist_ok=True)

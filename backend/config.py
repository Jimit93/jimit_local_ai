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


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    return [p for p in raw.split(";") if p.strip()] if raw else default


@dataclass
class AgentSettings:
    """Coding agent: workspace scope, loop limits, resource policy, code-model tiers."""

    # Directories the agent may operate in. More can be registered at runtime; every
    # path is resolved and checked against these, so nothing outside is reachable.
    workspace_roots: list[str] = field(
        default_factory=lambda: _env_list("JLA_WORKSPACES", [str(ROOT)]))
    # CPU policy (system-wide percent). <caution: normal; caution..cap: reduce;
    # >cap: protect; >emergency: stop starting work.
    cpu_caution_percent: float = float(os.environ.get("JLA_CPU_CAUTION", 60))
    cpu_cap_percent: float = float(os.environ.get("JLA_CPU_CAP", 70))
    cpu_emergency_percent: float = float(os.environ.get("JLA_CPU_EMERGENCY", 90))
    ram_min_available_gb: float = float(os.environ.get("JLA_RAM_MIN_GB", 1.0))
    sampler_interval_s: float = 1.0
    # Controlled-loop limits.
    max_iterations: int = 8
    max_repairs: int = 2
    step_timeout_s: float = 900.0
    task_timeout_s: float = float(os.environ.get("JLA_AGENT_TASK_TIMEOUT", 3600))
    command_timeout_s: float = 180.0
    test_timeout_s: float = 600.0
    # IO caps.
    max_read_chars: int = 20000
    max_edit_bytes: int = 1_000_000
    context_max_chars: int = 7000
    auto_resume: bool = os.environ.get("JLA_AGENT_AUTO_RESUME", "1") == "1"
    max_auto_resumes: int = 2
    # Code-model tiers (Ollama names). Cheapest first; the capable tier is only used
    # when RAM allows and a cheaper attempt was insufficient.
    code_model_small: str = os.environ.get("JLA_CODE_MODEL_SMALL", "qwen2.5-coder:0.5b")
    code_model_capable: str = os.environ.get(
        "JLA_CODE_MODEL_CAPABLE", "deepseek-coder:6.7b-instruct-q4_K_M")


@dataclass
class Settings:
    llm: LLMSettings = field(default_factory=LLMSettings)
    harness: HarnessSettings = field(default_factory=HarnessSettings)
    agent: AgentSettings = field(default_factory=AgentSettings)
    host: str = "127.0.0.1"       # local-only bind by default
    port: int = int(os.environ.get("JLA_PORT", 8077))


SETTINGS = Settings()


def ensure_dirs() -> None:
    for d in (DATA_DIR, LOG_DIR, REPORTS_DIR, DOCS_DIR, CONFIG_DIR):
        d.mkdir(parents=True, exist_ok=True)

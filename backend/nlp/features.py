"""NLP feature extraction. Rule-based and lexicon-based - no LLM calls.

Everything here runs in microseconds and is deterministic, so the same post always
yields the same features (required for reproducible benchmarks).
"""
from __future__ import annotations

import re
from collections import Counter

from backend.tools.deterministic import (
    BULLET_RE,
    EMOJI_RE,
    WORD_RE,
    paragraphs,
    sentences,
    text_stats,
)

# --------------------------------------------------------------- technical lexicon

TECH_SEED = {
    "ai", "ml", "llm", "llms", "rag", "nlp", "embedding", "embeddings", "vector", "vectors",
    "model", "models", "inference", "quantized", "quantization", "gguf", "onnx", "safetensors",
    "token", "tokens", "tokenizer", "latency", "throughput", "benchmark", "benchmarks",
    "backtest", "pipeline", "api", "apis", "sdk", "cpu", "gpu", "ram", "vram", "runtime",
    "deterministic", "heuristic", "classifier", "classification", "regression", "clustering",
    "transformer", "transformers", "attention", "fine-tune", "finetune", "fine-tuning",
    "dataset", "datasets", "corpus", "training", "validation", "accuracy", "precision",
    "recall", "f1", "confidence", "agent", "agents", "agentic", "orchestration", "harness",
    "router", "routing", "prompt", "prompts", "context", "sqlite", "database", "faiss",
    "index", "retrieval", "semantic", "search", "ocr", "tts", "stt", "whisper", "qwen",
    "llama", "mistral", "python", "fastapi", "react", "docker", "automation", "workflow",
    "integration", "deployment", "production", "offline", "local", "on-device", "edge",
    "privacy", "encryption", "architecture", "framework", "library", "optimization",
    "cache", "caching", "batch", "batching", "streaming", "parser", "parsing", "regex",
    "json", "csv", "sql", "etl", "algorithm", "algorithms", "neural", "deep", "supervised",
    "unsupervised", "feature", "features", "hyperparameter", "epoch", "gradient", "loss",
    "trading", "backtesting", "portfolio", "risk", "drawdown", "sharpe", "volatility",
    "signal", "signals", "strategy", "execution", "slippage", "invoice", "invoicing",
    "accounting", "reconciliation", "erp", "crm", "saas", "infrastructure", "server",
    "endpoint", "microservice", "scalability", "throughput",
}

ACRONYM_RE = re.compile(r"\b[A-Z]{2,6}\b")
METRIC_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:%|x|ms|s|sec|secs|seconds|min|mins|hours?|gb|mb|kb|tb|k|m|bn?|"
    r"tok/s|tokens?|fps|hz|usd|aed|inr|\$)\b",
    re.I,
)
CAMEL_RE = re.compile(r"\b[a-z]+[A-Z][a-zA-Z]*\b")
CODEISH_RE = re.compile(r"\b\w+(?:\.py|\.js|\.ts|\.json|\.csv|\.db|\(\))\b")

# --------------------------------------------------------------------- hook types

HOOK_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("greeting", re.compile(r"^(?:hi|hello|hey)\s+(?:everyone|all|there|folks|connections)\b",
                            re.I)),
    ("gratitude", re.compile(r"^(?:thank(?:s| you)|grateful|huge thanks)\b", re.I)),
    ("what_if", re.compile(r"^what if\b", re.I)),
    ("transformation", re.compile(r"^from\s+[^,.]{1,40}\s+to\s+[^,.]{1,40}[:\-–—]", re.I)),
    ("label", re.compile(r"^[A-Z][\w &'/-]{2,40}:\s+\S|^[^.!?\n]{5,60}\s\|\s[^.!?\n]{5,80}$")),
    ("question", re.compile(r"^[^.!?]{0,160}\?")),
    ("metric", re.compile(r"^[^.!?\n]{0,40}\b\d+(?:[.,]\d+)?\s*(?:%|x|ms|gb|mb|tok|hours?|days?|"
                          r"minutes?|seconds?|months?|years?|k|m)\b", re.I)),
    ("contrarian", re.compile(r"^(?:most|everyone|people|they|nobody|no one|you)\b[^.!?]{0,80}"
                              r"\b(?:think|believe|say|assume|don't|doesn't|won't|isn't|"
                              r"aren't|not)\b", re.I)),
    ("announcement", re.compile(r"^(?:i|we)\s+(?:just\s+)?(?:built|shipped|launched|released|"
                                r"created|made|finished|completed|published|deployed)\b|"
                                r"^(?:introducing|announcing|excited to)\b", re.I)),
    ("milestone", re.compile(r"^(?:thrilled|excited|happy|proud|delighted|glad)\s+to\b|"
                             r"\b(?:certificate|certification|completed the|graduated|"
                             r"joined|promoted)\b", re.I)),
    ("problem", re.compile(r"^(?:the problem|here's the problem|most \w+ (?:fail|struggle)|"
                           r"why \w+|the issue|what nobody)\b", re.I)),
    ("listicle", re.compile(r"^\d+\s+(?:lessons|things|ways|reasons|steps|mistakes|tips|"
                            r"rules|takeaways)\b", re.I)),
    ("story", re.compile(r"^(?:last|yesterday|today|this (?:week|month|morning)|when i|"
                         r"it'?s been|a few (?:days|weeks|months)|back in)\b", re.I)),
    ("link", re.compile(r"^https?://\S+$")),
    ("claim", re.compile(r"^.{10,}")),  # fallback: a plain declarative assertion
]

CTA_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("comment", re.compile(r"\b(?:drop a comment|comment below|let me know|share your|"
                           r"tell me|thoughts\?|what do you think|curious what|"
                           r"would love to hear)\b", re.I)),
    ("dm", re.compile(r"\b(?:dm me|send me a (?:dm|message)|reach out|get in touch|"
                      r"happy to chat|let's connect|lets connect|connect with me)\b", re.I)),
    ("link", re.compile(r"\b(?:check it out|link (?:in|below)|read more|see it here|"
                        r"try it|full write-?up|details here)\b", re.I)),
    ("follow", re.compile(r"\b(?:follow (?:me|for)|hit follow|repost|share this)\b", re.I)),
    ("hiring", re.compile(r"\b(?:open to|actively (?:exploring|looking)|hiring|"
                          r"opportunities|collaborat(?:e|ion))\b", re.I)),
]

# Phrases that read as generic LLM filler. Used by the validator, never generated.
GENERIC_PHRASES = [
    "in today's fast-paced world", "in the ever-evolving", "game-changer", "game changer",
    "unlock the power", "dive deep", "delve into", "leverage the power", "seamlessly",
    "revolutionize", "cutting-edge solutions", "it's not just about", "the future is here",
    "harness the power", "harnessing the power", "cutting-edge", "cutting edge",
    "take it to the next level", "at the end of the day", "look no further",
    "tired of dealing with", "powerful features", "valuable asset",
    "moving forward", "synergy", "paradigm shift", "low-hanging fruit", "circle back",
    "i'm thrilled to announce that i have", "without further ado", "buckle up",
    "let that sink in", "here's the kicker", "spoiler alert",
]

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "that", "this", "these",
    "those", "is", "are", "was", "were", "be", "been", "being", "to", "of", "in", "on",
    "for", "with", "as", "at", "by", "from", "it", "its", "i", "you", "we", "they", "he",
    "she", "my", "your", "our", "their", "me", "us", "them", "not", "no", "do", "does",
    "did", "have", "has", "had", "will", "would", "can", "could", "should", "may", "might",
    "so", "just", "more", "most", "very", "up", "out", "about", "into", "over", "after",
    "all", "any", "each", "how", "what", "when", "where", "which", "who", "why", "s", "t",
}


def technical_terms(text: str) -> list[str]:
    """Tokens that carry technical weight: lexicon hits, acronyms, metrics, code-ish."""
    words = WORD_RE.findall(text or "")
    hits = [w for w in words if w.lower() in TECH_SEED]
    hits += ACRONYM_RE.findall(text or "")
    hits += [m.group(0) for m in METRIC_RE.finditer(text or "")]
    hits += CAMEL_RE.findall(text or "") + CODEISH_RE.findall(text or "")
    return hits


def technical_density(text: str) -> float:
    """Technical tokens / total tokens, capped at 1.0. Deterministic proxy for depth."""
    words = WORD_RE.findall(text or "")
    if not words:
        return 0.0
    return round(min(1.0, len(technical_terms(text)) / len(words)), 4)


def flesch_reading_ease(text: str) -> float:
    """Standard readability formula; syllables counted by vowel-group heuristic."""
    words = WORD_RE.findall(text or "")
    sents = sentences(text or "")
    if not words or not sents:
        return 0.0
    syll = 0
    for w in words:
        groups = re.findall(r"[aeiouy]+", w.lower())
        n = len(groups)
        if w.lower().endswith("e") and n > 1:
            n -= 1
        syll += max(1, n)
    score = 206.835 - 1.015 * (len(words) / len(sents)) - 84.6 * (syll / len(words))
    return round(score, 2)


def first_line(text: str) -> str:
    for line in (text or "").split("\n"):
        if line.strip():
            return line.strip()
    return ""


_LEAD_EMOJI_RE = re.compile(r"^(?:[\s*_]*(?:" + EMOJI_RE.pattern + r")+[\s*_]*)+")


def detect_hook(text: str) -> dict:
    """Classify the opening line into an archetype.

    A leading emoji is recorded as its own flag rather than as an archetype, so
    "🚀 AI is changing X" is still recognised as a claim.
    """
    fl = first_line(text)
    stripped = _LEAD_EMOJI_RE.sub("", fl).lstrip("*_ ").strip()
    for name, rx in HOOK_PATTERNS:
        if rx.search(stripped):
            return {
                "hook_type": name,
                "hook_text": fl,
                "hook_words": len(WORD_RE.findall(fl)),
                "starts_with_emoji": bool(_LEAD_EMOJI_RE.match(fl)),
            }
    return {
        "hook_type": "none",
        "hook_text": fl,
        "hook_words": len(WORD_RE.findall(fl)),
        "starts_with_emoji": bool(_LEAD_EMOJI_RE.match(fl)),
    }


def detect_cta(text: str) -> dict:
    """CTAs live in the tail of a post; look at the last ~3 non-empty lines."""
    lines = [ln.strip() for ln in (text or "").split("\n") if ln.strip()]
    tail = "\n".join(lines[-3:]) if lines else ""
    kinds = [name for name, rx in CTA_PATTERNS if rx.search(tail)]
    if not kinds:  # some CTAs sit mid-post
        kinds = [name for name, rx in CTA_PATTERNS if rx.search(text or "")]
        tail = tail if kinds else ""
    return {"cta_types": kinds, "has_cta": bool(kinds), "cta_text": tail if kinds else ""}


def generic_phrase_hits(text: str) -> list[str]:
    low = (text or "").lower()
    return [p for p in GENERIC_PHRASES if p in low]


def repetition_score(text: str) -> float:
    """Fraction of content words that repeat. High values mean padding."""
    words = [w.lower() for w in WORD_RE.findall(text or "") if w.lower() not in STOPWORDS]
    if len(words) < 8:
        return 0.0
    c = Counter(words)
    repeated = sum(v - 1 for v in c.values() if v > 1)
    return round(repeated / len(words), 4)


def structure_signature(text: str) -> dict:
    """Shape of the post: how it is laid out, independent of wording."""
    paras = paragraphs(text or "")
    lines = [ln for ln in (text or "").split("\n") if ln.strip()]
    bullets = len(BULLET_RE.findall(text or ""))
    return {
        "paragraph_count": len(paras),
        "line_count": len(lines),
        "bullet_count": bullets,
        "has_bullets": bullets >= 2,
        "avg_para_words": round(
            sum(len(WORD_RE.findall(p)) for p in paras) / len(paras), 2
        ) if paras else 0.0,
        "one_line_paragraphs": sum(1 for p in paras if len(sentences(p)) <= 1),
        "ends_with_hashtags": bool(re.search(r"#\w+\s*$", (text or "").strip())),
    }


def extract_features(text: str) -> dict:
    """Full deterministic + NLP feature set for one document."""
    base = text_stats(text)
    hook = detect_hook(text)
    cta = detect_cta(text)
    struct = structure_signature(text)
    return {
        **base,
        **hook,
        **cta,
        "structure": struct,
        "technical_density": technical_density(text),
        "technical_terms": sorted(set(t.lower() for t in technical_terms(text)))[:40],
        "readability": flesch_reading_ease(text),
        "repetition": repetition_score(text),
        "generic_phrases": generic_phrase_hits(text),
    }

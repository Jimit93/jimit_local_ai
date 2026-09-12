"""Topic classification for LinkedIn documents.

Weak supervision + classical ML:
  1. A deterministic keyword rule set labels the documents it is sure about.
  2. A TF-IDF + Logistic Regression model is trained on those labels so it can
     generalise to posts the rules do not cover (synonyms, paraphrases).
  3. Accuracy is measured by cross-validation, never asserted.

The LLM is not involved at any point.
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path

from backend.common import ProcessorResult
from backend.config import DATA_DIR
from backend.db import execute, executemany, query

MODEL_PATH = DATA_DIR / "models" / "topic_clf.joblib"
META_PATH = DATA_DIR / "models" / "topic_clf.json"

# Topic lexicons derived from the actual corpus vocabulary (see style_profile.json).
TOPIC_RULES: dict[str, list[str]] = {
    "ai_engineering": [
        "ai", "llm", "agent", "agents", "agentic", "model", "models", "rag", "embedding",
        "machine learning", "deep learning", "neural", "inference", "gpt", "qwen", "llama",
        "prompt", "fine-tune", "automation", "python", "chatbot", "generative ai", "nlp",
        "openai", "gemini", "ocr", "vector", "dataset", "api", "local ai", "offline ai",
    ],
    "trading": [
        "trading", "trade", "forex", "fx", "pips", "supply zone", "demand zone", "candle",
        "timeframe", "bos", "break of structure", "liquidity", "drawdown", "equity",
        "portfolio", "risk management", "entry", "stop loss", "take profit", "bullish",
        "bearish", "market", "capital", "profit", "lot size", "chart", "setup",
        "head shoulders", "price action", "mastermind", "master mind",
    ],
    "career": [
        "opportunity", "opportunities", "hiring", "role", "job", "recruiter", "resume",
        "seeking", "opentowork", "open to work", "interview", "position", "internship",
        "apply", "career", "referral",
    ],
    "learning": [
        "certification", "certificate", "course", "completed", "capstone", "iit", "program",
        "simplilearn", "skillup", "bootcamp", "learned", "training program", "workshop",
        "graduated", "curriculum", "masterclass", "webinar",
    ],
    "business": [
        "business", "client", "clients", "customer", "roi", "cost", "revenue", "invoice",
        "accounting", "erp", "operations", "workflow", "productivity", "saas", "startup",
        "enterprise", "onboarding", "partnership", "brokers",
    ],
}

_COMPILED = {
    topic: [re.compile(r"\b" + re.escape(k) + r"\w{0,3}\b", re.I) for k in kws]
    for topic, kws in TOPIC_RULES.items()
}
LABELS = sorted(TOPIC_RULES)


def rule_scores(text: str) -> dict[str, int]:
    return {t: sum(1 for rx in rxs if rx.search(text or "")) for t, rxs in _COMPILED.items()}


def rule_label(text: str, min_hits: int = 2, margin: int = 1) -> tuple[str | None, float]:
    """Deterministic label, or (None, 0.0) when the rules are not confident."""
    sc = rule_scores(text)
    ranked = sorted(sc.items(), key=lambda kv: -kv[1])
    top, top_n = ranked[0]
    second_n = ranked[1][1] if len(ranked) > 1 else 0
    if top_n < min_hits or (top_n - second_n) < margin:
        return None, 0.0
    conf = min(0.95, 0.55 + 0.08 * (top_n - second_n) + 0.02 * top_n)
    return top, round(conf, 4)


class TopicClassifier:
    """TF-IDF + multinomial logistic regression. Tiny, fast, honest about confidence."""

    def __init__(self):
        self.pipe = None
        self.meta: dict = {}

    # -- persistence -------------------------------------------------------
    def save(self) -> None:
        import joblib

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipe, MODEL_PATH)
        META_PATH.write_text(json.dumps(self.meta, indent=2), encoding="utf-8")

    def load(self) -> bool:
        import joblib

        if not MODEL_PATH.exists():
            return False
        try:
            self.pipe = joblib.load(MODEL_PATH)
            self.meta = json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else {}
            return True
        except Exception:
            return False

    # -- training ----------------------------------------------------------
    def fit(self, texts: list[str], labels: list[str]) -> dict:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import Pipeline

        t0 = time.perf_counter()
        self.pipe = Pipeline(
            [
                ("tfidf", TfidfVectorizer(
                    sublinear_tf=True, ngram_range=(1, 2), min_df=1, max_df=0.9,
                    strip_accents="unicode", lowercase=True, stop_words="english",
                )),
                ("clf", LogisticRegression(
                    max_iter=2000, C=4.0, class_weight="balanced", multi_class="auto",
                )),
            ]
        )
        counts = Counter(labels)
        folds = max(2, min(5, min(counts.values())))
        cv = None
        if len(counts) > 1 and folds >= 2:
            cv = cross_val_score(self.pipe, texts, labels, cv=folds, scoring="accuracy")
        self.pipe.fit(texts, labels)
        self.meta = {
            "n_train": len(texts),
            "labels": sorted(counts),
            "label_counts": dict(counts),
            "cv_folds": folds if cv is not None else 0,
            "cv_accuracy_mean": round(float(cv.mean()), 4) if cv is not None else None,
            "cv_accuracy_std": round(float(cv.std()), 4) if cv is not None else None,
            "train_ms": round((time.perf_counter() - t0) * 1000, 1),
            "trained_at": time.time(),
        }
        return self.meta

    # -- inference ---------------------------------------------------------
    def predict(self, text: str) -> tuple[str, float]:
        if self.pipe is None and not self.load():
            raise RuntimeError("topic classifier is not trained")
        probs = self.pipe.predict_proba([text])[0]
        classes = list(self.pipe.classes_)
        i = int(probs.argmax())
        return classes[i], round(float(probs[i]), 4)

    def predict_all(self, text: str) -> dict[str, float]:
        if self.pipe is None and not self.load():
            raise RuntimeError("topic classifier is not trained")
        probs = self.pipe.predict_proba([text])[0]
        return {c: round(float(p), 4) for c, p in zip(self.pipe.classes_, probs)}


_clf: TopicClassifier | None = None


def classifier() -> TopicClassifier:
    global _clf
    if _clf is None:
        _clf = TopicClassifier()
        _clf.load()
    return _clf


def train_from_corpus(min_words: int = 15) -> dict:
    """Label the corpus with rules, train the model, store topics in SQLite."""
    from backend.ingestion.linkedin.ingest import writing_corpus

    docs = writing_corpus(types=("post", "comment"), min_words=min_words)
    labelled = []
    for d in docs:
        lab, conf = rule_label(d["text"])
        if lab:
            labelled.append((d, lab, conf))
    if len({l for _, l, _ in labelled}) < 2:
        return {"error": "not enough labelled classes", "n_labelled": len(labelled)}

    clf = classifier()
    meta = clf.fit([d["text"] for d, _, _ in labelled], [l for _, l, _ in labelled])
    meta["n_docs"] = len(docs)
    meta["n_rule_labelled"] = len(labelled)
    meta["rule_coverage"] = round(len(labelled) / max(1, len(docs)), 4)
    clf.save()

    # Persist topics for every document: rules where confident, model elsewhere.
    execute("DELETE FROM topics WHERE source IN ('rules','tfidf_lr')")
    rows = []
    for d in docs:
        lab, conf = rule_label(d["text"])
        if lab:
            rows.append((d["id"], lab, conf, "rules"))
        else:
            plab, pconf = clf.predict(d["text"])
            rows.append((d["id"], plab, pconf, "tfidf_lr"))
    executemany(
        "INSERT OR REPLACE INTO topics(doc_id,topic,confidence,source) VALUES(?,?,?,?)", rows
    )
    meta["topic_distribution"] = dict(Counter(r[1] for r in rows))
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def classify(text: str, min_conf: float = 0.6) -> ProcessorResult:
    """Rules first (free, exact); model second; escalate if neither is confident."""
    t0 = time.perf_counter()
    lab, conf = rule_label(text)
    if lab:
        return ProcessorResult(
            result=lab, confidence=conf, processor="rule_keyword_topics",
            latency_ms=(time.perf_counter() - t0) * 1000, layer="nlp",
            meta={"scores": rule_scores(text)},
        )
    try:
        clf = classifier()
        plab, pconf = clf.predict(text)
        return ProcessorResult(
            result=plab, confidence=pconf, processor="tfidf_logistic_regression",
            latency_ms=(time.perf_counter() - t0) * 1000, layer="ml",
            meta={"probabilities": clf.predict_all(text), "escalate": pconf < min_conf},
        )
    except Exception as e:
        return ProcessorResult(
            result=None, confidence=0.0, processor="tfidf_logistic_regression",
            latency_ms=(time.perf_counter() - t0) * 1000, layer="ml", error=str(e),
        )


def topic_counts() -> dict[str, int]:
    rows = query("SELECT topic, COUNT(*) c FROM topics GROUP BY topic ORDER BY c DESC")
    return {r["topic"]: r["c"] for r in rows}

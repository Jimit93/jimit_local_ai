"""Task classification: decide what kind of work a request is, cheaply.

Two stages, both free:
  1. High-precision regex rules (microseconds). Most real requests stop here.
  2. A TF-IDF + Logistic Regression intent model trained on the seed examples below,
     for phrasings the rules do not cover.

Only if both are unsure does the caller pay for an LLM.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from backend.common import ProcessorResult
from backend.config import DATA_DIR
from backend.tools.deterministic import looks_like_math

MODEL_PATH = DATA_DIR / "models" / "intent_clf.joblib"
META_PATH = DATA_DIR / "models" / "intent_clf.json"

CATEGORIES = [
    "deterministic",
    "text_processing",
    "classification",
    "retrieval",
    "semantic_search",
    "ranking",
    "structured_data",
    "coding",
    "creative_generation",
    "reasoning",
    "workspace",
]

# Requests about the LinkedIn corpus never go to the coding rules, whatever verbs they use.
_CORPUS_WORDS = re.compile(r"\b(?:linkedin|posts?|hashtags?|comments?|reactions?|my writing)\b", re.I)


def _code_rule(t: str) -> tuple[str | None, float, str]:
    """Workspace/coding requests, recognised by the coding-intent rules (no ML here)."""
    if _CORPUS_WORDS.search(t):
        return None, 0.0, ""
    from backend.agent.intents import DETERMINISTIC, rule_intent

    intent, conf, _how = rule_intent(t)
    if intent is None or intent == "calculate":
        return None, 0.0, ""
    cat = "workspace" if intent in DETERMINISTIC else "coding"
    return cat, conf, f"rule:{cat}/{intent}"

# --- stage 1: high-precision rules -----------------------------------------

RULES: list[tuple[str, re.Pattern, float]] = [
    # Classification is checked first: "classify this LinkedIn post" also contains
    # the words that would otherwise look like a generation request.
    ("classification", re.compile(
        r"\b(?:classify|categori[sz]e|what (?:topic|category|type))\b|"
        r"\bwhich (?:pillar|bucket|category)\b", re.I), 0.93),
    # Text-level counting/extraction is checked before corpus counting and before
    # generation: "how many words in this text" is not a database question, and a URL
    # ending in "/post" must not read as a request to write a post.
    ("text_processing", re.compile(
        r"\b(?:word count|character count|how many words|how many characters|"
        r"how many sentences|count the (?:words|characters|hashtags|sentences)|"
        r"extract (?:all )?(?:the )?(?:urls?|links?|hashtags?|mentions?|emails?)|"
        r"pull the (?:urls?|hashtags?|links?)|readability|reading level)\b", re.I), 0.94),
    ("creative_generation", re.compile(
        r"\b(?:write|draft|compose|generate|create)\b(?!-up)\W{1,3}"
        r"(?:\w+\W+){0,3}?\b(?:post|linkedin|article|caption|copy|content|newsletter)\b|"
        r"^\W*(?:write|draft|compose|generate|create|turn this into)\b.{0,60}"
        r"\blinkedin\b", re.I), 0.95),
    ("structured_data", re.compile(
        r"\bhow many\b.{0,40}\b(?:posts?|comments?|documents?|records?|reactions?)\b|"
        r"\b(?:count|total number of)\b.{0,30}\b(?:posts?|comments?|docs?|documents?)\b|"
        r"\bwhat(?:'s| is) in (?:my|the) (?:database|knowledge base)\b", re.I), 0.92),
    ("retrieval", re.compile(
        r"\b(?:find|show|list|search|get)\b.{0,40}\b(?:posts?|comments?|documents?)\b"
        r".{0,30}\b(?:mentioning|containing|with the word|that mention|that say|"
        r"including the word)\b|\bexact(?:ly)? match\b", re.I), 0.93),
    ("semantic_search", re.compile(
        r"\b(?:similar|related|about|relevant|like)\b.{0,30}\b(?:posts?|content|writing)\b|"
        r"\b(?:posts?|writing)\b.{0,25}\b(?:semantically|conceptually)\b|"
        r"\bwhat have i (?:written|posted) about\b", re.I), 0.88),
    ("ranking", re.compile(
        r"\b(?:rank|top \d+|best performing|most (?:common|frequent|used))\b", re.I), 0.82),
    ("coding", re.compile(
        r"\b(?:write|fix|debug|refactor|implement)\b.{0,25}\b(?:code|function|script|"
        r"class|bug|python|javascript|sql query|regex)\b|\bstack trace\b", re.I), 0.9),
    ("reasoning", re.compile(
        r"\b(?:why|explain|analy[sz]e|compare|evaluate|what makes|how does|should i|"
        r"pros and cons|trade-?offs?)\b", re.I), 0.7),
]


def rule_classify(text: str) -> tuple[str | None, float, str]:
    t = (text or "").strip()
    if looks_like_math(t):
        return "deterministic", 0.97, "math_gate"
    if re.search(r"\b(?:what(?:'s| is) (?:the )?(?:date|time)|today'?s date|current time|"
                 r"days? between|how many days (?:until|since))\b", t, re.I):
        return "deterministic", 0.95, "datetime_gate"
    cat, conf, how = _code_rule(t)
    if cat:
        return cat, conf, how
    for cat, rx, conf in RULES:
        if rx.search(t):
            return cat, conf, f"rule:{cat}"
    return None, 0.0, "no_rule"


# --- stage 2: seed-trained intent model ------------------------------------

SEED: dict[str, list[str]] = {
    "deterministic": [
        "what is 25 * 40", "calculate 15% of 2400", "compute the average of 3 5 9",
        "how many days between 2026-01-01 and 2026-06-30", "what is the date today",
        "sum these numbers 10 20 30", "convert 5000 divided by 8", "round 3.14159 to 2 places",
        "what is 2 to the power of 10", "subtract 450 from 1200",
    ],
    "text_processing": [
        "count the words in this paragraph", "how many characters is this text",
        "extract all the urls from this", "pull the hashtags out of this post",
        "what is the readability score of this", "how many sentences are in this text",
        "strip the emojis from this text", "list the mentions in this comment",
        "give me the character count", "how many hashtags did i use here",
    ],
    "classification": [
        "classify this post as ai or trading", "what category does this text belong to",
        "which content pillar is this", "categorise this comment",
        "is this post about trading or engineering", "tag this document with a topic",
        "what topic is this about", "label this post",
    ],
    "retrieval": [
        "find my posts mentioning rag", "show posts containing the word qwen",
        "list every comment that mentions forex", "search my posts for the word benchmark",
        "which posts include the phrase supply zone", "get posts with the word offline in them",
        "find documents containing python", "show me posts that say local ai",
    ],
    "semantic_search": [
        "find posts related to local ai", "what have i written about agents",
        "show me writing similar to edge deployment", "posts about running models offline",
        "anything i posted conceptually close to automation",
        "which of my posts are about risk management", "find similar content to this idea",
        "what did i say about building agents",
    ],
    "ranking": [
        "top 5 hashtags i use", "rank my most common phrases", "what words do i use most",
        "most frequent topics in my posts", "top 10 technical terms in my writing",
        "rank these options by relevance", "which hooks do i use most often",
    ],
    "structured_data": [
        "how many posts do i have", "count the documents in the knowledge base",
        "how many comments were ingested", "what is in my database",
        "number of embeddings stored", "how many posts per topic",
        "give me the totals by document type", "how many reactions are recorded",
    ],
    "coding": [
        "write a python function to parse csv", "fix this traceback", "debug this regex",
        "refactor this class", "write a sql query for the posts table",
        "implement a binary search in python", "why does this code throw a keyerror",
    ],
    "creative_generation": [
        "write a linkedin post about my local ai agent", "draft a post on quantized models",
        "generate linkedin content about benchmarking", "compose a post announcing the project",
        "write me a post about cpu inference", "create a linkedin update about the harness",
        "draft something for linkedin about rag", "write a post in my style about agents",
    ],
    "reasoning": [
        "explain what makes my writing distinctive", "why is cpu inference slow",
        "compare bm25 and embeddings", "should i use a 7b or a 1.8b model",
        "analyse the tradeoffs of local inference", "what makes this architecture efficient",
        "evaluate whether the harness is worth it", "how does the router decide",
    ],
}


class IntentModel:
    def __init__(self):
        self.pipe = None
        self.meta: dict = {}

    def load(self) -> bool:
        import joblib

        if not MODEL_PATH.exists():
            return False
        try:
            self.pipe = joblib.load(MODEL_PATH)
            if META_PATH.exists():
                self.meta = json.loads(META_PATH.read_text(encoding="utf-8"))
            return True
        except Exception:
            return False

    def fit(self) -> dict:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import FeatureUnion, Pipeline
        import joblib

        texts = [t for cat in SEED for t in SEED[cat]]
        labels = [cat for cat in SEED for _ in SEED[cat]]
        t0 = time.perf_counter()
        self.pipe = Pipeline([
            # Word + character n-grams: the seed set is small, char n-grams add
            # robustness to morphology and typos without needing more examples.
            ("tfidf", FeatureUnion([
                ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                         strip_accents="unicode", min_df=1)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                         sublinear_tf=True, min_df=1)),
            ])),
            ("clf", LogisticRegression(max_iter=1500, C=6.0, class_weight="balanced")),
        ])
        cv = cross_val_score(self.pipe, texts, labels, cv=4, scoring="accuracy")
        self.pipe.fit(texts, labels)
        self.meta = {
            "n_train": len(texts),
            "categories": sorted(set(labels)),
            "cv_accuracy_mean": round(float(cv.mean()), 4),
            "cv_accuracy_std": round(float(cv.std()), 4),
            "train_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipe, MODEL_PATH)
        META_PATH.write_text(json.dumps(self.meta, indent=2), encoding="utf-8")
        return self.meta

    def predict(self, text: str) -> tuple[str, float]:
        if self.pipe is None and not self.load():
            self.fit()
        probs = self.pipe.predict_proba([text])[0]
        classes = list(self.pipe.classes_)
        i = int(probs.argmax())
        return classes[i], round(float(probs[i]), 4)


_model: IntentModel | None = None


def model() -> IntentModel:
    global _model
    if _model is None:
        _model = IntentModel()
        if not _model.load():
            _model.fit()
    return _model


def classify_task(text: str, ml_min_conf: float = 0.45) -> ProcessorResult:
    """Return the task category with a confidence and the stage that decided it."""
    t0 = time.perf_counter()
    cat, conf, how = rule_classify(text)
    if cat:
        return ProcessorResult(
            result=cat, confidence=conf, processor=f"task_classifier/{how}",
            latency_ms=(time.perf_counter() - t0) * 1000, layer="deterministic",
            meta={"stage": "rules"},
        )
    try:
        mcat, mconf = model().predict(text)
        return ProcessorResult(
            result=mcat, confidence=mconf, processor="task_classifier/tfidf_lr",
            latency_ms=(time.perf_counter() - t0) * 1000, layer="ml",
            meta={"stage": "ml", "needs_llm": mconf < ml_min_conf},
        )
    except Exception as e:
        return ProcessorResult(
            result="reasoning", confidence=0.2, processor="task_classifier/fallback",
            latency_ms=(time.perf_counter() - t0) * 1000, layer="deterministic",
            meta={"stage": "fallback"}, error=str(e),
        )

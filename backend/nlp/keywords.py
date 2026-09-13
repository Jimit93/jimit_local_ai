"""Keyword extraction, n-grams and lexical search (TF-IDF / BM25).

Used for: exact-ish search, topic keywords, style phrase mining, and as the cheap
first stage of hybrid retrieval.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache

from backend.nlp.features import STOPWORDS
from backend.tools.deterministic import WORD_RE

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]{1,}")


def tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    toks = [t.lower().strip("'-") for t in _TOKEN_RE.findall(text or "")]
    if keep_stopwords:
        return [t for t in toks if t]
    return [t for t in toks if t and t not in STOPWORDS and len(t) > 1]


def ngrams(tokens: list[str], n: int = 2) -> list[str]:
    return [" ".join(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]


def term_frequencies(texts: list[str], n: int = 1, top: int = 50) -> list[tuple[str, int]]:
    c: Counter = Counter()
    for t in texts:
        toks = tokenize(t)
        c.update(toks if n == 1 else ngrams(toks, n))
    return c.most_common(top)


def phrase_frequencies(texts: list[str], top: int = 30, min_count: int = 3) -> list[dict]:
    """Recurring 2-4 word phrases, longest-first, with shorter subsumed phrases dropped."""
    out: list[dict] = []
    seen: set[str] = set()
    for n in (4, 3, 2):
        for phrase, count in term_frequencies(texts, n=n, top=200):
            if count < min_count:
                continue
            if any(phrase in s for s in seen):
                continue
            seen.add(phrase)
            out.append({"phrase": phrase, "count": count, "n": n})
    return sorted(out, key=lambda d: -d["count"])[:top]


def keywords_tfidf(text: str, corpus: list[str], top: int = 10) -> list[tuple[str, float]]:
    """Classic TF-IDF keyword extraction against a background corpus."""
    doc = tokenize(text)
    if not doc:
        return []
    tf = Counter(doc)
    n_docs = max(1, len(corpus))
    df: Counter = Counter()
    for c in corpus:
        df.update(set(tokenize(c)))
    scores = {
        w: (cnt / len(doc)) * math.log((n_docs + 1) / (df.get(w, 0) + 1)) + 1e-9
        for w, cnt in tf.items()
    }
    return sorted(scores.items(), key=lambda kv: -kv[1])[:top]


def keywords_simple(text: str, top: int = 8) -> list[str]:
    """No-corpus fallback: frequency of non-stopword terms, bigrams boosted."""
    toks = tokenize(text)
    c = Counter(toks)
    bi = Counter(ngrams(toks, 2))
    ranked = [(w, n * (1 + len(w) / 20)) for w, n in c.items()]
    ranked += [(p, n * 1.6) for p, n in bi.items() if n > 1]
    return [w for w, _ in sorted(ranked, key=lambda kv: -kv[1])[:top]]


class BM25Index:
    """Okapi BM25 over a document list. Pure lexical, no model, no network."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75, tokenizer=None):
        self.k1, self.b = k1, b
        # Prose by default; code search passes an identifier-aware tokenizer.
        self._tok = tokenizer or tokenize
        self.docs_tokens = [self._tok(d) for d in docs]
        self.N = len(self.docs_tokens)
        self.doc_len = [len(t) for t in self.docs_tokens]
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0
        self.tf: list[Counter] = [Counter(t) for t in self.docs_tokens]
        df: Counter = Counter()
        for t in self.docs_tokens:
            df.update(set(t))
        self.idf = {
            w: math.log(1 + (self.N - c + 0.5) / (c + 0.5)) for w, c in df.items()
        }

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        q = self._tok(query)
        if not q or not self.N:
            return []
        scores = []
        for i in range(self.N):
            s = 0.0
            dl = self.doc_len[i] or 1
            for w in q:
                f = self.tf[i].get(w, 0)
                if not f:
                    continue
                idf = self.idf.get(w, 0.0)
                s += idf * (f * (self.k1 + 1)) / (
                    f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                )
            if s > 0:
                scores.append((i, s))
        scores.sort(key=lambda kv: -kv[1])
        return scores[:top_k]

    def normalised_search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        """Scores rescaled to 0..1 so they can be fused with cosine similarity."""
        hits = self.search(query, top_k)
        if not hits:
            return []
        top = hits[0][1] or 1.0
        return [(i, round(s / top, 4)) for i, s in hits]


@lru_cache(maxsize=8)
def _cached_stop() -> frozenset:
    return frozenset(STOPWORDS)


def shared_ngram(a: str, b: str, n: int = 6) -> str | None:
    """Longest-first check for a copied word run. Jaccard overlap misses these:
    a short verbatim line inside a long document scores low but is still a copy."""
    aw = re.findall(r"[\w'#]+", (a or "").lower())
    bw = re.findall(r"[\w'#]+", (b or "").lower())
    if len(aw) < n or len(bw) < n:
        n = min(len(aw), len(bw), n)
        if n < 3:
            return None
    bgrams = {" ".join(bw[i: i + n]) for i in range(len(bw) - n + 1)}
    for i in range(len(aw) - n + 1):
        g = " ".join(aw[i: i + n])
        if g in bgrams:
            return g
    return None


def lexical_overlap(a: str, b: str) -> float:
    """Jaccard overlap of content words. Used for style/topic similarity scoring."""
    sa, sb = set(tokenize(a)), set(tokenize(b))
    if not sa or not sb:
        return 0.0
    return round(len(sa & sb) / len(sa | sb), 4)


def vocabulary_profile(texts: list[str], top: int = 60) -> dict:
    """Vocabulary fingerprint: favourite content words, verbs and openers."""
    toks: list[str] = []
    verbs: Counter = Counter()
    openers: Counter = Counter()
    VERB_RE = re.compile(r"\b(\w+(?:ed|ing|es|s))\b", re.I)
    for t in texts:
        toks += tokenize(t)
        for line in (t or "").split("\n"):
            w = WORD_RE.findall(line.strip())
            if w:
                openers[w[0].lower()] += 1
        verbs.update(
            v.lower() for v in VERB_RE.findall(t or "")
            if v.lower() not in STOPWORDS and len(v) > 3
        )
    c = Counter(toks)
    return {
        "top_terms": c.most_common(top),
        "top_verbs": verbs.most_common(25),
        "top_line_openers": openers.most_common(20),
        "vocabulary_size": len(c),
        "total_tokens": len(toks),
    }

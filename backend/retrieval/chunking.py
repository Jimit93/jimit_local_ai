"""Semantic chunking for the RAG pipeline.

Chunks are cut where the *meaning* shifts, not every N characters:

  1. split into sentences (deterministic),
  2. embed each sentence with the local MiniLM,
  3. measure cosine distance between consecutive sentences,
  4. cut where a distance exceeds a percentile of that document's own
     distribution (an adaptive threshold - a dense technical post and a rambling
     one get different cut points),
  5. merge the resulting groups so every chunk respects min/max word bounds.

Documents shorter than `min_chunk_words * 2` are returned as a single chunk: an
80-word post has no internal structure worth splitting, and chunking it would only
scatter its context across rows.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.tools.deterministic import paragraphs, sentences, word_count


@dataclass
class Chunk:
    index: int
    text: str
    word_count: int
    sentence_span: tuple[int, int]
    boundary_score: float  # semantic distance at the cut that opened this chunk

    def to_dict(self) -> dict:
        return {
            "index": self.index, "text": self.text, "word_count": self.word_count,
            "sentence_span": list(self.sentence_span),
            "boundary_score": round(self.boundary_score, 4),
        }


def _sentence_units(text: str) -> list[str]:
    """Sentences, but never crossing a paragraph break.

    In this corpus a blank line is a hard semantic boundary the author put there
    deliberately, so it is respected before any embedding is computed.
    """
    units: list[str] = []
    for para in paragraphs(text) or [text]:
        sents = sentences(para) or [para.strip()]
        units.extend(s for s in sents if s.strip())
    return units


def semantic_chunks(
    text: str,
    *,
    min_chunk_words: int = 45,
    max_chunk_words: int = 180,
    percentile: float = 80.0,
    min_boundary_distance: float = 0.15,
    embed_fn=None,
) -> list[Chunk]:
    """Split one document into semantically coherent chunks."""
    text = (text or "").strip()
    if not text:
        return []
    total = word_count(text)
    if total <= min_chunk_words * 2:
        return [Chunk(0, text, total, (0, 0), 0.0)]

    units = _sentence_units(text)
    if len(units) < 3:
        return [Chunk(0, text, total, (0, len(units)), 0.0)]

    if embed_fn is None:
        from backend.retrieval.index import embed as embed_fn  # local, cached model

    vecs = embed_fn(units)                       # already L2-normalised
    sims = np.sum(vecs[:-1] * vecs[1:], axis=1)  # cosine similarity of neighbours
    dists = 1.0 - sims

    # Adaptive threshold from the document's own distribution, with an absolute
    # floor. Without the floor a document whose sentences are all on one topic has
    # a percentile of ~0, which would make every sentence a boundary.
    pct_value = float(np.percentile(dists, percentile)) if len(dists) > 1 else 1.0
    threshold = max(pct_value, min_boundary_distance)

    groups: list[list[int]] = [[0]]
    scores: list[float] = [0.0]
    for i, d in enumerate(dists, start=1):
        cur_words = sum(word_count(units[j]) for j in groups[-1])
        next_words = word_count(units[i])
        if cur_words + next_words > max_chunk_words:
            groups.append([i])            # hard cap reached
            scores.append(float(d))
            continue
        if d >= threshold and cur_words >= min_chunk_words:
            groups.append([i])            # meaning shifted and the chunk is big enough
            scores.append(float(d))
            continue
        groups[-1].append(i)

    # Merge any group left below the minimum into its neighbour.
    merged: list[list[int]] = []
    merged_scores: list[float] = []
    for g, s in zip(groups, scores):
        gw = sum(word_count(units[j]) for j in g)
        if merged and gw < min_chunk_words:
            prev_words = sum(word_count(units[j]) for j in merged[-1])
            if prev_words + gw <= max_chunk_words:
                merged[-1].extend(g)
                continue
        merged.append(list(g))
        merged_scores.append(s)

    out: list[Chunk] = []
    for idx, (g, s) in enumerate(zip(merged, merged_scores)):
        body = " ".join(units[j] for j in g).strip()
        out.append(Chunk(idx, body, word_count(body), (g[0], g[-1]), s))
    return out


def chunk_stats(chunks: list[Chunk]) -> dict:
    if not chunks:
        return {"n": 0}
    w = [c.word_count for c in chunks]
    return {
        "n": len(chunks),
        "mean_words": round(sum(w) / len(w), 1),
        "min_words": min(w),
        "max_words": max(w),
    }

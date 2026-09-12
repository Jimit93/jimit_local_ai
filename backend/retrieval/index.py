"""Hybrid local retrieval over semantic chunks.

    metadata filter -> BM25 -> MiniLM embeddings -> fusion -> collapse to documents

Order matters for cost. Cheap exact filters run first and shrink the candidate set;
the embedding model only ever sees what survives. Retrieval happens at *chunk*
granularity (see `chunking.py`) so a long post matches on the passage that is
actually relevant, then results are collapsed back to one hit per document.

Everything is local: MiniLM is loaded from the existing HuggingFace cache with
offline mode forced.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np

from backend.common import ProcessorResult
from backend.config import EMBED_MODEL_NAME
from backend.db import connect, execute, query
from backend.nlp.keywords import BM25Index, tokenize

# Never reach for the network: the model is already in the local HF cache.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_embedder = None
_index: "DocIndex | None" = None


def embedder():
    """Lazily load MiniLM once per process."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
    return _embedder


def embed(texts: list[str]) -> np.ndarray:
    vecs = embedder().encode(
        texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vecs.astype("float32")


@dataclass
class Hit:
    doc_id: int
    score: float
    text: str                 # the matching chunk
    doc_type: str
    created_at: str | None
    url: str | None
    topic: str | None
    lexical: float = 0.0
    semantic: float = 0.0
    chunk_index: int = 0
    n_chunks: int = 1
    doc_text: str = ""        # the full document, for callers that need structure

    def to_dict(self, snippet: int = 400) -> dict:
        return {
            "doc_id": self.doc_id, "score": round(self.score, 4),
            "lexical": round(self.lexical, 4), "semantic": round(self.semantic, 4),
            "doc_type": self.doc_type, "created_at": self.created_at,
            "url": self.url, "topic": self.topic,
            "chunk_index": self.chunk_index, "n_chunks": self.n_chunks,
            "text": self.text[:snippet] + ("…" if len(self.text) > snippet else ""),
        }


class DocIndex:
    """In-memory index over semantic chunks; vectors are cached in SQLite."""

    def __init__(self, doc_types: tuple[str, ...] = ("post", "comment"), min_words: int = 15):
        self.doc_types = doc_types
        self.min_words = min_words
        self.rows: list[dict] = []        # one entry per chunk
        self.bm25: BM25Index | None = None
        self.vectors: np.ndarray | None = None
        self.faiss_index = None
        self.built_at: float | None = None
        self.build_ms: float = 0.0
        self.chunk_ms: float = 0.0

    # -- documents ---------------------------------------------------------
    def load_docs(self) -> list[dict]:
        ph = ",".join("?" * len(self.doc_types))
        rows = query(
            f"SELECT d.id, d.doc_type, d.created_at, d.text, d.url, "
            f"(SELECT t.topic FROM topics t WHERE t.doc_id = d.id LIMIT 1) AS topic "
            f"FROM linkedin_documents d WHERE d.doc_type IN ({ph}) "
            f"ORDER BY d.created_at DESC",
            self.doc_types,
        )
        return [
            dict(r) for r in rows
            if len(tokenize(r["text"], keep_stopwords=True)) >= self.min_words
        ]

    # -- chunking ----------------------------------------------------------
    def ensure_chunks(self, docs: list[dict], rechunk: bool = False) -> list[dict]:
        """Return persisted chunks for these documents, computing what is missing."""
        from backend.retrieval.chunking import semantic_chunks

        t0 = time.perf_counter()
        conn = connect()
        if rechunk:
            execute("DELETE FROM chunks")

        have = {
            r["doc_id"] for r in query(
                "SELECT DISTINCT doc_id FROM chunks WHERE doc_id IN (%s)"
                % ",".join(str(int(d["id"])) for d in docs)
            )
        } if docs else set()

        new_rows = []
        for d in docs:
            if d["id"] in have:
                continue
            for c in semantic_chunks(d["text"]):
                new_rows.append((d["id"], c.index, c.text, c.word_count,
                                 c.boundary_score, time.time()))
        if new_rows:
            conn.executemany(
                "INSERT OR REPLACE INTO chunks(doc_id,chunk_index,text,word_count,"
                "boundary_score,created_at) VALUES(?,?,?,?,?,?)", new_rows)
            conn.commit()
        self.chunk_ms = (time.perf_counter() - t0) * 1000

        by_doc = {d["id"]: d for d in docs}
        if not by_doc:
            return []
        crows = query(
            "SELECT id, doc_id, chunk_index, text, word_count FROM chunks "
            "WHERE doc_id IN (%s) ORDER BY doc_id, chunk_index"
            % ",".join(str(int(i)) for i in by_doc)
        )
        counts: dict[int, int] = {}
        for r in crows:
            counts[r["doc_id"]] = counts.get(r["doc_id"], 0) + 1
        out = []
        for r in crows:
            d = by_doc[r["doc_id"]]
            out.append({
                "chunk_id": r["id"], "doc_id": r["doc_id"], "chunk_index": r["chunk_index"],
                "text": r["text"], "n_chunks": counts[r["doc_id"]],
                "doc_type": d["doc_type"], "created_at": d["created_at"],
                "url": d["url"], "topic": d.get("topic"), "doc_text": d["text"],
            })
        return out

    # -- build -------------------------------------------------------------
    def _cached_vectors(self, chunk_ids: list[int]) -> dict[int, np.ndarray]:
        if not chunk_ids:
            return {}
        ph = ",".join("?" * len(chunk_ids))
        rows = query(
            f"SELECT chunk_id, dim, vector FROM chunk_embeddings "
            f"WHERE model=? AND chunk_id IN ({ph})",
            [EMBED_MODEL_NAME, *chunk_ids],
        )
        return {
            r["chunk_id"]: np.frombuffer(r["vector"], dtype="float32", count=r["dim"])
            for r in rows
        }

    def build(self, use_embeddings: bool = True, rechunk: bool = False) -> dict:
        t0 = time.perf_counter()
        docs = self.load_docs()
        self.rows = self.ensure_chunks(docs, rechunk=rechunk)
        texts = [r["text"] for r in self.rows]
        self.bm25 = BM25Index(texts)

        if use_embeddings and self.rows:
            ids = [r["chunk_id"] for r in self.rows]
            cached = self._cached_vectors(ids)
            missing = [i for i, r in enumerate(self.rows) if r["chunk_id"] not in cached]
            if missing:
                new_vecs = embed([texts[i] for i in missing])
                conn = connect()
                conn.executemany(
                    "INSERT OR REPLACE INTO chunk_embeddings"
                    "(chunk_id,model,dim,vector,created_at) VALUES(?,?,?,?,?)",
                    [(self.rows[i]["chunk_id"], EMBED_MODEL_NAME, int(v.shape[0]),
                      v.tobytes(), time.time()) for i, v in zip(missing, new_vecs)],
                )
                conn.commit()
                for i, v in zip(missing, new_vecs):
                    cached[self.rows[i]["chunk_id"]] = v
            self.vectors = np.vstack([cached[r["chunk_id"]] for r in self.rows])
            self._build_faiss()

        self.built_at = time.time()
        self.build_ms = (time.perf_counter() - t0) * 1000
        return {
            "documents": len({r["doc_id"] for r in self.rows}),
            "chunks": len(self.rows),
            "embedded": 0 if self.vectors is None else int(self.vectors.shape[0]),
            "dim": 0 if self.vectors is None else int(self.vectors.shape[1]),
            "build_ms": round(self.build_ms, 1),
            "chunk_ms": round(self.chunk_ms, 1),
            "faiss": self.faiss_index is not None,
        }

    def _build_faiss(self) -> None:
        if self.vectors is None or not len(self.vectors):
            return
        try:
            import faiss

            idx = faiss.IndexFlatIP(self.vectors.shape[1])  # vectors are L2-normalised
            idx.add(self.vectors)
            self.faiss_index = idx
        except Exception:
            self.faiss_index = None  # numpy fallback below

    # -- search ------------------------------------------------------------
    @property
    def docs(self) -> list[dict]:
        """Distinct documents currently indexed (kept for callers and tests)."""
        seen: dict[int, dict] = {}
        for r in self.rows:
            if r["doc_id"] not in seen:
                seen[r["doc_id"]] = {
                    "id": r["doc_id"], "doc_type": r["doc_type"],
                    "created_at": r["created_at"], "text": r["doc_text"],
                    "url": r["url"], "topic": r["topic"],
                }
        return list(seen.values())

    def _candidate_mask(
        self, doc_type: str | None, topic: str | None, since: str | None, until: str | None,
        exclude_ids: set[int] | None = None,
    ) -> list[int]:
        """Metadata filter first - the cheapest possible narrowing."""
        out = []
        for i, r in enumerate(self.rows):
            if exclude_ids and r["doc_id"] in exclude_ids:
                continue
            if doc_type and r["doc_type"] != doc_type:
                continue
            if topic and (r.get("topic") or "") != topic:
                continue
            if since and (r.get("created_at") or "") < since:
                continue
            if until and (r.get("created_at") or "") > until:
                continue
            out.append(i)
        return out

    def search(
        self,
        q: str,
        top_k: int = 5,
        *,
        doc_type: str | None = None,
        topic: str | None = None,
        since: str | None = None,
        until: str | None = None,
        alpha: float = 0.5,          # 0 = lexical only, 1 = semantic only
        use_embeddings: bool = True,
        exclude_ids: set[int] | None = None,
        group_by_doc: bool = True,
    ) -> list[Hit]:
        if not self.rows:
            return []
        cand = self._candidate_mask(doc_type, topic, since, until, exclude_ids)
        if not cand:
            return []
        cand_set = set(cand)
        pool = max(top_k * 6, 30)

        lex: dict[int, float] = {}
        if self.bm25 is not None and alpha < 1.0:
            for i, s in self.bm25.normalised_search(q, top_k=pool):
                if i in cand_set:
                    lex[i] = s

        sem: dict[int, float] = {}
        if use_embeddings and self.vectors is not None and alpha > 0.0:
            qv = embed([q])[0]
            # FAISS covers the whole corpus, so it is only valid when nothing is
            # filtered out; otherwise use a masked numpy dot product.
            if self.faiss_index is not None and len(cand) == len(self.rows):
                D, I = self.faiss_index.search(qv.reshape(1, -1), min(len(self.rows), pool))
                sem = {int(i): float(d) for i, d in zip(I[0], D[0]) if i >= 0}
            else:
                sims = self.vectors[cand] @ qv
                order = np.argsort(-sims)[:pool]
                sem = {cand[int(j)]: float(sims[int(j)]) for j in order}

        fused: dict[int, float] = {}
        for i in set(lex) | set(sem):
            fused[i] = (1 - alpha) * lex.get(i, 0.0) + alpha * max(0.0, sem.get(i, 0.0))

        ranked = sorted(fused.items(), key=lambda kv: -kv[1])
        hits: list[Hit] = []
        seen_docs: set[int] = set()
        for i, score in ranked:
            r = self.rows[i]
            if group_by_doc and r["doc_id"] in seen_docs:
                continue          # keep only each document's best-matching chunk
            seen_docs.add(r["doc_id"])
            hits.append(Hit(
                doc_id=r["doc_id"], score=score, text=r["text"], doc_type=r["doc_type"],
                created_at=r["created_at"], url=r["url"], topic=r.get("topic"),
                lexical=lex.get(i, 0.0), semantic=max(0.0, sem.get(i, 0.0)),
                chunk_index=r["chunk_index"], n_chunks=r["n_chunks"],
                doc_text=r["doc_text"],
            ))
            if len(hits) >= top_k:
                break
        return hits

    def keyword_search(self, pattern: str, top_k: int = 20, regex: bool = False) -> list[Hit]:
        """Exact / regex match over documents. No model, no embeddings."""
        import re as _re

        rx = _re.compile(pattern if regex else _re.escape(pattern), _re.I)
        hits = []
        for d in self.docs:
            n = len(rx.findall(d["text"]))
            if n:
                hits.append(Hit(
                    doc_id=d["id"], score=float(n), text=d["text"], doc_type=d["doc_type"],
                    created_at=d["created_at"], url=d["url"], topic=d.get("topic"),
                    lexical=float(n), doc_text=d["text"],
                ))
        return sorted(hits, key=lambda h: -h.score)[:top_k]


def index(rebuild: bool = False, use_embeddings: bool = True, rechunk: bool = False) -> DocIndex:
    global _index
    if _index is None or rebuild:
        _index = DocIndex()
        _index.build(use_embeddings=use_embeddings, rechunk=rechunk)
    return _index


def retrieve(q: str, top_k: int = 5, **kw) -> ProcessorResult:
    """Retrieval as a harness processor: confidence is the top fused score."""
    t0 = time.perf_counter()
    try:
        hits = index().search(q, top_k=top_k, **kw)
    except Exception as e:
        return ProcessorResult(
            result=[], confidence=0.0, processor="hybrid_retrieval",
            latency_ms=(time.perf_counter() - t0) * 1000, layer="retrieval", error=str(e)
        )
    conf = float(hits[0].score) if hits else 0.0
    return ProcessorResult(
        result=[h.to_dict() for h in hits], confidence=min(1.0, conf),
        processor="hybrid_retrieval(bm25+minilm)",
        latency_ms=(time.perf_counter() - t0) * 1000, layer="retrieval",
        meta={"n_hits": len(hits), "corpus": len(index().rows)},
    )


def stats() -> dict:
    idx = index()
    words = [r for r in idx.rows]
    return {
        "documents_indexed": len({r["doc_id"] for r in idx.rows}),
        "chunks": len(idx.rows),
        "chunks_per_doc": round(len(idx.rows) / max(1, len({r["doc_id"] for r in idx.rows})), 2),
        "multi_chunk_docs": len({r["doc_id"] for r in words if r["n_chunks"] > 1}),
        "embeddings": 0 if idx.vectors is None else int(idx.vectors.shape[0]),
        "dim": 0 if idx.vectors is None else int(idx.vectors.shape[1]),
        "faiss": idx.faiss_index is not None,
        "model": EMBED_MODEL_NAME,
        "chunking": "semantic",
        "build_ms": round(idx.build_ms, 1),
    }

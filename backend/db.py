"""SQLite storage (WAL). One file: data/agent.db."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from backend.config import DB_PATH, ensure_dirs

_local = threading.local()

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

-- Raw ingested records, one row per source document (posts, comments, profile rows).
CREATE TABLE IF NOT EXISTS linkedin_documents (
    id            INTEGER PRIMARY KEY,
    doc_uid       TEXT UNIQUE NOT NULL,      -- stable hash of source+type+content
    doc_type      TEXT NOT NULL,             -- post|comment|article|profile|position|skill|project|education
    source_file   TEXT NOT NULL,
    created_at    TEXT,                      -- original timestamp, ISO, may be NULL
    text          TEXT NOT NULL,
    url           TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    ingested_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_docs_type ON linkedin_documents(doc_type);
CREATE INDEX IF NOT EXISTS idx_docs_date ON linkedin_documents(created_at);

-- Derived per-document features (deterministic + NLP). One row per document.
CREATE TABLE IF NOT EXISTS style_features (
    doc_id            INTEGER PRIMARY KEY REFERENCES linkedin_documents(id) ON DELETE CASCADE,
    char_count        INTEGER, word_count INTEGER, sentence_count INTEGER,
    paragraph_count   INTEGER, avg_sentence_len REAL, avg_word_len REAL,
    hashtag_count     INTEGER, emoji_count INTEGER, url_count INTEGER,
    question_count    INTEGER, exclamation_count INTEGER, number_count INTEGER,
    bullet_count      INTEGER, caps_word_count INTEGER,
    technical_density REAL, readability REAL,
    hook              TEXT, cta TEXT, has_cta INTEGER,
    features_json     TEXT NOT NULL DEFAULT '{}'
);

-- Topic assignment per document (ML/keyword derived, with confidence).
CREATE TABLE IF NOT EXISTS topics (
    doc_id     INTEGER NOT NULL REFERENCES linkedin_documents(id) ON DELETE CASCADE,
    topic      TEXT NOT NULL,
    confidence REAL NOT NULL,
    source     TEXT NOT NULL,               -- rules|tfidf_lr|kmeans
    PRIMARY KEY (doc_id, topic, source)
);

-- Semantic chunks: the unit the RAG pipeline actually retrieves over.
-- Short documents produce exactly one chunk, so nothing is fragmented needlessly.
CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY,
    doc_id      INTEGER NOT NULL REFERENCES linkedin_documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    text        TEXT NOT NULL,
    word_count  INTEGER NOT NULL,
    boundary_score REAL NOT NULL DEFAULT 0.0,
    created_at  REAL NOT NULL,
    UNIQUE(doc_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);

CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id   INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    model      TEXT NOT NULL,
    dim        INTEGER NOT NULL,
    vector     BLOB NOT NULL,
    created_at REAL NOT NULL
);

-- Retrieval moved to chunk granularity; the document-level vector table it replaced
-- is dropped so nothing can read stale counts from it.
DROP TABLE IF EXISTS embeddings;

-- Persistent agent memory, kept separate by kind and never auto-trusted.
CREATE TABLE IF NOT EXISTS agent_memory (
    id         INTEGER PRIMARY KEY,
    kind       TEXT NOT NULL,               -- preference|project|task_history|style|fact|optimization
    key        TEXT NOT NULL,
    value      TEXT NOT NULL,
    source     TEXT NOT NULL,               -- user|ingestion|derived|llm
    confidence REAL NOT NULL DEFAULT 1.0,
    created_at REAL NOT NULL,
    UNIQUE(kind, key)
);

-- Observability: one row per executed task.
CREATE TABLE IF NOT EXISTS task_runs (
    id             INTEGER PRIMARY KEY,
    task_id        TEXT NOT NULL,
    ts             REAL NOT NULL,
    task_type      TEXT,
    processor      TEXT,
    layer          TEXT,
    confidence     REAL,
    latency_ms     REAL,
    llm_used       INTEGER,
    llm_calls      INTEGER,
    tokens         INTEGER,
    escalations    INTEGER,
    success        INTEGER,
    error_type     TEXT,
    input_chars    INTEGER
);
CREATE INDEX IF NOT EXISTS idx_taskruns_ts ON task_runs(ts);

-- Benchmark storage: baseline (LLM-only) vs harness, per task.
CREATE TABLE IF NOT EXISTS benchmark_runs (
    id          INTEGER PRIMARY KEY,
    run_id      TEXT NOT NULL,
    ts          REAL NOT NULL,
    suite       TEXT NOT NULL,
    task_id     TEXT NOT NULL,
    mode        TEXT NOT NULL,              -- baseline|harness
    correct     INTEGER,
    score       REAL,
    latency_ms  REAL,
    llm_calls   INTEGER,
    tokens      INTEGER,
    processor   TEXT,
    cpu_percent REAL,
    rss_mb      REAL,
    error       TEXT
);
CREATE INDEX IF NOT EXISTS idx_bench_run ON benchmark_runs(run_id);
"""


def connect() -> sqlite3.Connection:
    """Thread-local connection; schema is applied once per connection."""
    conn = getattr(_local, "conn", None)
    if conn is not None:
        return conn
    ensure_dirs()
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    _local.conn = conn
    return conn


def query(sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
    return connect().execute(sql, tuple(params)).fetchall()


def execute(sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
    c = connect()
    cur = c.execute(sql, tuple(params))
    c.commit()
    return cur


def executemany(sql: str, rows: Iterable[Iterable[Any]]) -> None:
    c = connect()
    c.executemany(sql, rows)
    c.commit()


def counts() -> dict[str, int]:
    out = {}
    for t in (
        "linkedin_documents",
        "style_features",
        "topics",
        "chunks",
        "chunk_embeddings",
        "agent_memory",
        "task_runs",
        "benchmark_runs",
    ):
        try:
            out[t] = query(f"SELECT COUNT(*) c FROM {t}")[0]["c"]
        except sqlite3.Error:
            out[t] = 0
    return out


def remember(kind: str, key: str, value: Any, source: str, confidence: float = 1.0) -> None:
    """Upsert one memory entry. Generated content must be stored with source='llm'."""
    execute(
        "INSERT INTO agent_memory(kind,key,value,source,confidence,created_at) "
        "VALUES(?,?,?,?,?,?) ON CONFLICT(kind,key) DO UPDATE SET "
        "value=excluded.value, source=excluded.source, confidence=excluded.confidence, "
        "created_at=excluded.created_at",
        (kind, key, json.dumps(value) if not isinstance(value, str) else value,
         source, confidence, time.time()),
    )


def recall(kind: str, key: str | None = None) -> list[dict]:
    if key is None:
        rows = query("SELECT * FROM agent_memory WHERE kind=? ORDER BY created_at DESC", (kind,))
    else:
        rows = query("SELECT * FROM agent_memory WHERE kind=? AND key=?", (kind, key))
    return [dict(r) for r in rows]

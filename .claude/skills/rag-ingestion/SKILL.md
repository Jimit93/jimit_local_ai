---
name: rag-ingestion
description: Ingest files into the local RAG pipeline with semantic chunking, embeddings and hybrid retrieval. Use when adding a new document source, changing chunking, or debugging why retrieval returned the wrong passage.
---

# RAG ingestion

## Pipeline

```
file → header/format detection → normalise → dedupe → SQLite (linkedin_documents)
     → semantic chunking → chunk embeddings (MiniLM, cached) → BM25 + FAISS
```

## Semantic chunking is the default — never fixed-size windows

`backend/retrieval/chunking.py`:

1. Sentences, never crossing a paragraph break (a blank line is an author-placed
   boundary and is respected before any embedding is computed).
2. Embed each sentence with the local MiniLM (`retrieval.index.embed`).
3. Cosine distance between consecutive sentences.
4. Cut where distance ≥ `max(percentile(dists, 80), 0.15)`.
   **The floor matters:** on a single-topic document the percentile collapses to 0
   and every sentence becomes a boundary.
5. Merge groups to respect `min_chunk_words=45` / `max_chunk_words=180`.

Documents ≤ `min_chunk_words * 2` return one chunk. Do not "fix" this — splitting an
80-word post scatters its context for no gain.

Invariant enforced by `tests/test_chunking.py`: chunking never drops or duplicates a
word.

## Adding a new source

1. Parse to `{doc_uid, doc_type, source_file, created_at, text, url, metadata_json}`.
   Detect the format by **content signature** (CSV header, magic bytes), not by
   filename.
2. `doc_uid` = sha1 of type + timestamp + whitespace-normalised lowercase text, so
   re-ingestion is idempotent.
3. Insert with `INSERT OR IGNORE`, then call
   `index(rebuild=True, rechunk=True).build(rechunk=True)`.
4. Chunks and vectors are cached in `chunks` / `chunk_embeddings`; only new rows are
   embedded on a rebuild.

## Retrieval

- Order: metadata filter → BM25 → vectors. Never embed before filtering.
- `alpha` blends lexical and semantic (0 = BM25 only, 1 = vectors only; 0.5 default).
- FAISS is only valid when nothing was filtered out; the masked numpy path handles
  filtered searches.
- `search(group_by_doc=True)` keeps each document's best chunk. Pass `False` when you
  genuinely want multiple passages from one document.
- **`Hit.text` is the chunk; `Hit.doc_text` is the whole document.** Use the chunk
  for grounding (less context), the document for anything about structure or voice.

## Privacy

Ingestion is local and offline. `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` are set
before the embedder loads. Never add a source that fetches over the network without
saying so explicitly.

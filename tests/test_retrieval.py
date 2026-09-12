import pytest

from backend.retrieval.index import index, retrieve, stats


@pytest.fixture(scope="module")
def idx():
    i = index()
    if not i.docs:
        i.build(use_embeddings=True)
    if not i.docs:
        pytest.skip("no documents indexed - run ingestion first")
    return i


def test_index_is_built_from_the_local_corpus(idx):
    s = stats()
    assert s["documents_indexed"] == len(idx.docs) > 0
    # Retrieval is chunk-level: one vector per chunk, at least one chunk per document.
    assert s["chunks"] >= s["documents_indexed"]
    assert s["embeddings"] == s["chunks"] == len(idx.rows)
    assert s["chunking"] == "semantic"
    assert s["dim"] == 384          # MiniLM-L6-v2
    assert "MiniLM" in s["model"]


def test_long_documents_are_split_and_short_ones_are_not(idx):
    by_doc: dict[int, int] = {}
    for r in idx.rows:
        by_doc[r["doc_id"]] = by_doc.get(r["doc_id"], 0) + 1
    assert min(by_doc.values()) == 1
    for r in idx.rows:
        if r["n_chunks"] == 1:
            continue
        assert len(r["doc_text"].split()) > len(r["text"].split())


def test_hits_carry_chunk_and_document_text(idx):
    hits = idx.search("local ai agent", top_k=3)
    assert hits
    for h in hits:
        assert h.doc_text and len(h.doc_text) >= len(h.text)
        assert 0 <= h.chunk_index < h.n_chunks


def test_results_are_collapsed_to_one_hit_per_document(idx):
    hits = idx.search("ai automation business", top_k=5)
    assert len({h.doc_id for h in hits}) == len(hits)


def test_exact_keyword_search_is_exact(idx):
    hits = idx.keyword_search("zzzznotpresentzzz")
    assert hits == []
    any_word = idx.docs[0]["text"].split()[0]
    assert idx.keyword_search(any_word)


def test_hybrid_search_returns_scored_hits(idx):
    hits = idx.search("local ai agent", top_k=3)
    assert hits and len(hits) <= 3
    assert all(0.0 <= h.score for h in hits)
    assert hits == sorted(hits, key=lambda h: -h.score)


def test_metadata_filter_runs_before_scoring(idx):
    hits = idx.search("trading", top_k=5, doc_type="post")
    assert all(h.doc_type == "post" for h in hits)


def test_exclude_ids_prevents_leakage(idx):
    first = idx.search("local ai", top_k=1)
    if not first:
        pytest.skip("no hits")
    excluded = {first[0].doc_id}
    hits = idx.search("local ai", top_k=5, exclude_ids=excluded)
    assert all(h.doc_id not in excluded for h in hits)


def test_lexical_only_and_semantic_only_modes_differ(idx):
    lex = idx.search("offline inference", top_k=5, alpha=0.0)
    sem = idx.search("offline inference", top_k=5, alpha=1.0)
    assert all(h.semantic == 0.0 for h in lex)
    assert all(h.lexical == 0.0 for h in sem)


def test_retrieve_envelope_costs_nothing(idx):
    r = retrieve("local ai agent", top_k=3)
    assert r.ok and r.layer == "retrieval"
    assert r.llm_used is False and r.tokens == 0
    assert r.latency_ms < 5000

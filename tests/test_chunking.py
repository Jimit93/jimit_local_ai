"""Semantic chunking: boundaries follow meaning, not character counts."""
import numpy as np
import pytest

from backend.retrieval.chunking import Chunk, chunk_stats, semantic_chunks


def fake_embed(texts):
    """Deterministic stand-in: sentences sharing a keyword point the same way.

    Lets the boundary logic be tested without loading MiniLM.
    """
    vecs = []
    for t in texts:
        low = t.lower()
        v = np.array([
            1.0 if "trading" in low or "chart" in low else 0.0,
            1.0 if "model" in low or "inference" in low else 0.0,
            0.1,
        ], dtype="float32")
        vecs.append(v / np.linalg.norm(v))
    return np.vstack(vecs)


def test_short_document_is_never_split():
    text = "A short post about local models. It runs on CPU."
    chunks = semantic_chunks(text, embed_fn=fake_embed)
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_empty_text_yields_nothing():
    assert semantic_chunks("", embed_fn=fake_embed) == []


def test_boundary_follows_a_topic_change():
    trading = ("The chart shows a clean trading setup today. " * 6)
    models = ("The model runs local inference on CPU. " * 6)
    chunks = semantic_chunks(trading + models, min_chunk_words=20, max_chunk_words=200,
                             embed_fn=fake_embed)
    assert len(chunks) >= 2
    # The cut must land where the subject changes, not in the middle of a topic.
    first, second = chunks[0].text.lower(), chunks[1].text.lower()
    assert "chart" in first and "inference" in second
    assert chunks[1].boundary_score > 0.5


def test_max_words_is_respected():
    text = "The model runs local inference on CPU today. " * 40
    chunks = semantic_chunks(text, min_chunk_words=20, max_chunk_words=60,
                             embed_fn=fake_embed)
    assert all(c.word_count <= 70 for c in chunks), [c.word_count for c in chunks]
    assert len(chunks) > 1


def test_no_chunk_below_the_minimum_except_the_last_resort():
    text = ("Alpha beta gamma delta epsilon zeta eta theta. " * 12)
    chunks = semantic_chunks(text, min_chunk_words=30, max_chunk_words=90,
                             embed_fn=fake_embed)
    assert all(c.word_count >= 20 for c in chunks[:-1])


def test_chunks_cover_the_document():
    text = ("The chart shows a trading setup. " * 8) + ("The model runs inference. " * 8)
    chunks = semantic_chunks(text, min_chunk_words=15, max_chunk_words=120,
                             embed_fn=fake_embed)
    joined = " ".join(c.text for c in chunks).split()
    assert len(joined) == len(text.split()), "chunking must not drop or duplicate words"


def test_paragraph_breaks_are_respected_as_hard_boundaries():
    text = "First idea here with several words in it.\n\nSecond idea, quite different.\n\n" \
           "Third idea about the model and inference on CPU hardware today."
    chunks = semantic_chunks(text, min_chunk_words=5, max_chunk_words=25,
                             embed_fn=fake_embed)
    assert len(chunks) >= 2


def test_stats():
    cs = [Chunk(0, "a b c", 3, (0, 0), 0.0), Chunk(1, "d e", 2, (1, 1), 0.5)]
    s = chunk_stats(cs)
    assert s == {"n": 2, "mean_words": 2.5, "min_words": 2, "max_words": 3}
    assert chunk_stats([]) == {"n": 0}


@pytest.mark.parametrize("pct", [50.0, 80.0, 95.0])
def test_percentile_controls_granularity(pct):
    text = ("The chart shows a trading setup. " * 6) + ("The model runs inference. " * 6)
    chunks = semantic_chunks(text, min_chunk_words=10, max_chunk_words=200,
                             percentile=pct, embed_fn=fake_embed)
    assert 1 <= len(chunks) <= 12

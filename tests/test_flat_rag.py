import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.flat_rag import (
    load_corpus,
    split_corpus_into_chunks,
    load_index_and_chunks,
    retrieve_chunks,
    build_context_from_chunks,
)


def test_load_corpus():
    corpus = load_corpus()

    assert isinstance(corpus, str)
    assert len(corpus) > 0


def test_split_corpus_into_chunks():
    corpus = load_corpus()
    chunks = split_corpus_into_chunks(corpus)

    assert isinstance(chunks, list)
    assert len(chunks) > 0

    for chunk in chunks:
        assert isinstance(chunk, str)
        assert len(chunk) > 0


def test_load_index_and_chunks():
    index, chunks = load_index_and_chunks()

    assert index.ntotal > 0
    assert isinstance(chunks, list)
    assert len(chunks) > 0


def test_retrieve_chunks():
    results = retrieve_chunks(
        "Who founded OpenAI?",
        top_k=3
    )

    assert isinstance(results, list)
    assert len(results) > 0

    first = results[0]

    assert "chunk_id" in first
    assert "score" in first
    assert "text" in first


def test_build_context_from_chunks():
    fake_chunks = [
        {
            "chunk_id": 0,
            "score": 0.9,
            "text": "OpenAI was founded by Sam Altman."
        }
    ]

    context = build_context_from_chunks(fake_chunks)

    assert "OpenAI was founded by Sam Altman." in context
    assert "Score" in context
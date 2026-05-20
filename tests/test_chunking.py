"""Tests for the chunking strategies."""

from __future__ import annotations

from src.rag_engine.chunking import HierarchicalChunker, SentenceWindowChunker


_LONG_DOC = (
    "Retrieval-Augmented Generation augments an LM with a retriever. "
    "The retriever indexes a corpus and returns the top-k most relevant chunks. "
    "These chunks are concatenated into the prompt for the generator.\n\n"
    "A common failure mode is chunk geometry mismatch. Embedding models prefer "
    "short inputs but the LM needs sufficient context to answer. "
    "Hierarchical chunking decouples these objectives.\n\n"
    "Parent chunks store the full context; child chunks are indexed for retrieval. "
    "At retrieval time the child hit triggers parent expansion. "
    "This is the parent-child pattern."
)


def test_hierarchical_chunker_produces_parents_and_children() -> None:
    chunker = HierarchicalChunker(parent_chars=300, child_chars=120, child_overlap_chars=20)
    parents, children = chunker.chunk(_LONG_DOC, doc_metadata={"doc_id": "d1"})
    assert len(parents) >= 1
    assert len(children) >= len(parents)
    for child in children:
        assert child.parent_id is not None
        assert any(p.chunk_id == child.parent_id for p in parents)


def test_hierarchical_chunker_rejects_bad_sizes() -> None:
    import pytest
    with pytest.raises(ValueError):
        HierarchicalChunker(parent_chars=100, child_chars=200)


def test_sentence_window_indexes_sentences_independently() -> None:
    chunker = SentenceWindowChunker(window=2)
    chunks = chunker.chunk(_LONG_DOC, doc_metadata={"doc_id": "d1"})
    # The toy doc has ~10 sentences.
    assert len(chunks) >= 5
    for idx, chunk in enumerate(chunks):
        assert chunk.metadata["sentence_idx"] == idx


def test_sentence_window_expansion_returns_neighbours() -> None:
    chunker = SentenceWindowChunker(window=2)
    chunks = chunker.chunk(_LONG_DOC, doc_metadata={"doc_id": "d1"})
    expanded = chunker.expand_window(chunks[3], {"d1": chunks})
    # Window 2 around index 3 = sentences 1..5 inclusive
    for idx in (1, 2, 3, 4, 5):
        if idx < len(chunks):
            assert chunks[idx].text[:30] in expanded

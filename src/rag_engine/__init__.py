"""Week 2 — Hierarchical Retrieval-Augmented Generation.

Components:

- :mod:`chunking`      — parent/child decomposition and sentence-window chunks.
- :mod:`embeddings`    — sentence-transformer encoder wrapper.
- :mod:`hnsw_native`   — a from-scratch HNSW index (used in the notebook to
                          expose the mathematics; production uses Chroma).
- :mod:`vector_stores` — pluggable Chroma / Milvus adapters.
- :mod:`pipeline`      — :class:`HierarchicalRAG` orchestrating ingest + retrieve.
"""

from src.rag_engine.chunking import (
    Chunk,
    HierarchicalChunker,
    SentenceWindowChunker,
)
from src.rag_engine.pipeline import HierarchicalRAG

__all__ = [
    "Chunk",
    "HierarchicalChunker",
    "SentenceWindowChunker",
    "HierarchicalRAG",
]

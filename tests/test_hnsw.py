"""Tests for the from-scratch HNSW index."""

from __future__ import annotations

import numpy as np

from src.rag_engine.hnsw_native import HNSWIndex


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def test_hnsw_recovers_nearest_neighbor() -> None:
    rng = np.random.default_rng(42)
    n, d = 200, 32
    data = _normalize(rng.standard_normal((n, d)).astype(np.float32))

    index = HNSWIndex(dim=d, M=8, ef_construction=64, ef_search=64, seed=1)
    for i, vec in enumerate(data):
        index.add(vec, payload={"id": i})

    # Query each known point. Recall@1 against ground truth should be near-perfect.
    correct = 0
    for i, vec in enumerate(data[:50]):
        results = index.search(vec, k=1)
        if results and results[0][1]["id"] == i:
            correct += 1
    assert correct >= 48  # ≥96% recall@1 on the in-sample query


def test_hnsw_returns_k_results() -> None:
    rng = np.random.default_rng(0)
    data = _normalize(rng.standard_normal((50, 16)).astype(np.float32))
    index = HNSWIndex(dim=16, M=4, ef_construction=32, ef_search=32, seed=0)
    for i, vec in enumerate(data):
        index.add(vec, payload={"id": i})
    results = index.search(data[0], k=10)
    assert len(results) == 10


def test_empty_index_returns_empty() -> None:
    index = HNSWIndex(dim=8)
    assert index.search(np.zeros(8, dtype=np.float32), k=5) == []

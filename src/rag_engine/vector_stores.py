"""Vector-store adapters: a common Protocol with Chroma and Milvus backends.

The pipeline depends only on :class:`VectorStore`; concrete backends are
selected at config time. Embeddings are passed in pre-computed so the
backend never owns the encoder — this keeps the embedding model swap-able
and avoids backend-specific encoder coupling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(slots=True)
class ScoredHit:
    chunk_id: str
    score: float
    text: str
    metadata: dict[str, Any]


class VectorStore(Protocol):
    """Backend-agnostic interface for an embedding store."""

    def upsert(
        self,
        ids: list[str],
        vectors: np.ndarray,
        texts: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None: ...

    def query(
        self, vector: np.ndarray, k: int, where: dict[str, Any] | None = None
    ) -> list[ScoredHit]: ...

    def get_by_id(self, chunk_id: str) -> ScoredHit | None: ...


class ChromaStore:
    """Chroma backend (recommended for local development)."""

    def __init__(self, collection: str, persist_dir: str) -> None:
        import chromadb

        self._client = chromadb.PersistentClient(path=persist_dir)
        self._coll = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(
        self,
        ids: list[str],
        vectors: np.ndarray,
        texts: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        self._coll.upsert(
            ids=ids,
            embeddings=vectors.tolist(),
            documents=texts,
            metadatas=metadatas,
        )

    def query(
        self, vector: np.ndarray, k: int, where: dict[str, Any] | None = None
    ) -> list[ScoredHit]:
        res = self._coll.query(
            query_embeddings=[vector.tolist()],
            n_results=k,
            where=where,
        )
        ids = res["ids"][0]
        dists = res["distances"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        return [
            ScoredHit(chunk_id=i, score=1.0 - d, text=t, metadata=m or {})
            for i, d, t, m in zip(ids, dists, docs, metas, strict=True)
        ]

    def get_by_id(self, chunk_id: str) -> ScoredHit | None:
        res = self._coll.get(ids=[chunk_id])
        if not res["ids"]:
            return None
        return ScoredHit(
            chunk_id=res["ids"][0],
            score=1.0,
            text=res["documents"][0],
            metadata=res["metadatas"][0] or {},
        )


class MilvusStore:
    """Milvus backend (recommended for scale).

    Implementation note: kept intentionally minimal here; the production
    deployment uses :class:`ChromaStore`. This adapter is the proof-of-
    concept for the pluggable-backend pattern.
    """

    def __init__(self, uri: str, token: str, collection: str, dim: int) -> None:
        from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

        connections.connect("default", uri=uri, token=token)
        if not utility.has_collection(collection):
            fields = [
                FieldSchema("chunk_id", dtype=DataType.VARCHAR, max_length=128, is_primary=True, auto_id=False),
                FieldSchema("vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
                FieldSchema("text", dtype=DataType.VARCHAR, max_length=8192),
                FieldSchema("metadata", dtype=DataType.JSON),
            ]
            schema = CollectionSchema(fields=fields, description="hierarchical RAG chunks")
            Collection(name=collection, schema=schema)
        self._coll = Collection(collection)
        self._coll.load()

    def upsert(
        self,
        ids: list[str],
        vectors: np.ndarray,
        texts: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        self._coll.upsert([ids, vectors.tolist(), texts, metadatas])

    def query(
        self, vector: np.ndarray, k: int, where: dict[str, Any] | None = None
    ) -> list[ScoredHit]:
        res = self._coll.search(
            data=[vector.tolist()],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=k,
            output_fields=["text", "metadata"],
        )[0]
        return [
            ScoredHit(
                chunk_id=hit.entity.get("chunk_id") or hit.id,
                score=float(hit.score),
                text=hit.entity.get("text", ""),
                metadata=hit.entity.get("metadata") or {},
            )
            for hit in res
        ]

    def get_by_id(self, chunk_id: str) -> ScoredHit | None:
        rows = self._coll.query(
            expr=f'chunk_id == "{chunk_id}"',
            output_fields=["text", "metadata"],
        )
        if not rows:
            return None
        return ScoredHit(
            chunk_id=chunk_id,
            score=1.0,
            text=rows[0].get("text", ""),
            metadata=rows[0].get("metadata") or {},
        )


def build_store(kind: str, **kwargs: Any) -> VectorStore:
    if kind == "chroma":
        return ChromaStore(**kwargs)
    if kind == "milvus":
        return MilvusStore(**kwargs)
    raise ValueError(f"Unknown vector store: {kind!r}")

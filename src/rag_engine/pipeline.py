"""End-to-end hierarchical RAG pipeline.

Ties together: chunker → encoder → vector store, with retrieval-time
expansion from child chunks back to their parents (or, in the
sentence-window strategy, back to the surrounding window).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.rag_engine.chunking import (
    Chunk,
    HierarchicalChunker,
    SentenceWindowChunker,
)
from src.rag_engine.embeddings import SentenceEncoder
from src.rag_engine.vector_stores import ScoredHit, VectorStore, build_store
from src.utils.config import load_yaml_config
from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class RetrievedDocument:
    """Document handed to the reranker / LLM, with parent-expanded text."""

    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any]


class HierarchicalRAG:
    """Parent-child hierarchical retrieval pipeline."""

    def __init__(
        self,
        chunker: HierarchicalChunker,
        encoder: SentenceEncoder,
        child_store: VectorStore,
        parent_lookup: dict[str, str],
    ) -> None:
        self.chunker = chunker
        self.encoder = encoder
        self.child_store = child_store
        # In-memory map child.parent_id -> parent_text. For production, persist
        # this in a key-value store; for this curriculum, in-memory is fine.
        self.parent_lookup = parent_lookup

    @classmethod
    def from_config(cls, path: str) -> HierarchicalRAG:
        cfg = load_yaml_config(path)
        chunker = HierarchicalChunker(
            parent_chars=cfg["chunker"].get("parent_chars", 4000),
            child_chars=cfg["chunker"].get("child_chars", 1000),
            child_overlap_chars=cfg["chunker"].get("child_overlap_chars", 100),
        )
        encoder = SentenceEncoder(model_name=cfg["encoder"]["model"])
        child_store = build_store(**cfg["vector_store"])
        return cls(chunker=chunker, encoder=encoder, child_store=child_store, parent_lookup={})

    # ------------------------------------------------------------------ #
    # Ingest                                                             #
    # ------------------------------------------------------------------ #

    def ingest_document(self, text: str, metadata: dict[str, Any]) -> int:
        parents, children = self.chunker.chunk(text, metadata)
        for p in parents:
            self.parent_lookup[p.chunk_id] = p.text
        if not children:
            return 0
        vectors = self.encoder.encode([c.text for c in children])
        self.child_store.upsert(
            ids=[c.chunk_id for c in children],
            vectors=vectors,
            texts=[c.text for c in children],
            metadatas=[c.metadata | {"parent_id": c.parent_id or ""} for c in children],
        )
        log.info("rag_ingest_done", parents=len(parents), children=len(children))
        return len(children)

    # ------------------------------------------------------------------ #
    # Retrieve                                                           #
    # ------------------------------------------------------------------ #

    def retrieve(
        self,
        query: str,
        k: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedDocument]:
        qvec = self.encoder.encode_one(query)
        hits = self.child_store.query(qvec, k=k, where=where)
        return self._expand_to_parents(hits)

    def _expand_to_parents(self, hits: list[ScoredHit]) -> list[RetrievedDocument]:
        seen_parents: set[str] = set()
        out: list[RetrievedDocument] = []
        for hit in hits:
            parent_id = hit.metadata.get("parent_id") or ""
            # Deduplicate: a single parent often contributes several child hits.
            if parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)
            parent_text = self.parent_lookup.get(parent_id, hit.text)
            out.append(
                RetrievedDocument(
                    chunk_id=parent_id or hit.chunk_id,
                    text=parent_text,
                    score=hit.score,
                    metadata=hit.metadata,
                )
            )
        return out


class SentenceWindowRAG:
    """Sentence-window retrieval pipeline.

    Sentences are indexed individually for retrieval precision; the
    ±window context is expanded at retrieval time so the LLM receives
    sufficient context.
    """

    def __init__(
        self,
        chunker: SentenceWindowChunker,
        encoder: SentenceEncoder,
        store: VectorStore,
    ) -> None:
        self.chunker = chunker
        self.encoder = encoder
        self.store = store
        self._by_doc: dict[str, list[Chunk]] = {}

    def ingest_document(self, text: str, metadata: dict[str, Any]) -> int:
        chunks = self.chunker.chunk(text, metadata)
        doc_id = chunks[0].metadata["doc_id"] if chunks else None
        if doc_id is not None:
            self._by_doc[doc_id] = chunks
        vectors = self.encoder.encode([c.text for c in chunks])
        self.store.upsert(
            ids=[c.chunk_id for c in chunks],
            vectors=vectors,
            texts=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )
        return len(chunks)

    def retrieve(self, query: str, k: int = 10) -> list[RetrievedDocument]:
        qvec = self.encoder.encode_one(query)
        hits = self.store.query(qvec, k=k)
        out: list[RetrievedDocument] = []
        for hit in hits:
            doc_id = hit.metadata.get("doc_id")
            if doc_id and doc_id in self._by_doc:
                idx = hit.metadata.get("sentence_idx", 0)
                window_chunk = Chunk(
                    chunk_id=hit.chunk_id, text=hit.text,
                    metadata={"doc_id": doc_id, "sentence_idx": idx},
                )
                expanded = self.chunker.expand_window(window_chunk, self._by_doc)
            else:
                expanded = hit.text
            out.append(
                RetrievedDocument(
                    chunk_id=hit.chunk_id, text=expanded,
                    score=hit.score, metadata=hit.metadata,
                )
            )
        return out

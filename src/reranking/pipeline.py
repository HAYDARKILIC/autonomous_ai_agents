"""End-to-end re-ranking pipeline composing query rewriter + retriever + re-ranker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.rag_engine.pipeline import RetrievedDocument
from src.reranking.cross_encoder import (
    CohereReranker,
    CrossEncoderReranker,
    RerankedDocument,
)
from src.reranking.query_rewriter import QueryRewriter
from src.utils.config import load_yaml_config
from src.utils.logging import get_logger

log = get_logger(__name__)


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        candidates: list[RetrievedDocument],
        k: int | None = None,
    ) -> list[RerankedDocument]: ...


@dataclass(slots=True)
class RerankingPipeline:
    """Orchestrates: (optional rewrite) → retrieve → re-rank → top-k.

    The retriever is injected; only the re-ranker is owned by this object.
    """

    reranker: Reranker
    query_rewriter: QueryRewriter | None = None
    candidate_k: int = 50

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedDocument],
        k: int = 5,
    ) -> list[RerankedDocument]:
        if self.query_rewriter is not None:
            query = self.query_rewriter.rewrite(query)
        return self.reranker.rerank(query, candidates, k=k)

    @classmethod
    def from_config(cls, path: str) -> RerankingPipeline:
        cfg = load_yaml_config(path)
        backend = cfg["reranker"]["backend"]
        if backend == "cross_encoder":
            rer: Reranker = CrossEncoderReranker(
                model_name=cfg["reranker"].get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            )
        elif backend == "cohere":
            rer = CohereReranker(model=cfg["reranker"].get("model", "rerank-english-v3.0"))
        else:
            raise ValueError(f"Unknown reranker backend: {backend!r}")
        return cls(
            reranker=rer,
            query_rewriter=None,
            candidate_k=cfg["reranker"].get("candidate_k", 50),
        )

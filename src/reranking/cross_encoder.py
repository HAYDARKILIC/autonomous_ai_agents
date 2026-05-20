"""Cross-Encoder re-ranking with training-objective controls.

Mathematical contrast with Bi-Encoders
--------------------------------------
A Bi-Encoder defines :math:`s(q, d) = \\phi(q)^\\top \\phi(d)`. Because
the encoder is shared and applied independently, document embeddings
can be pre-computed and cached, making retrieval :math:`O(N d)` per
query — feasible for millions of documents.

A Cross-Encoder concatenates the pair into a single input,
:math:`[CLS]\\,q\\,[SEP]\\,d`, and predicts a scalar :math:`s(q, d) \\in
\\mathbb{R}`. Document embeddings cannot be cached, so re-ranking
:math:`N` documents costs :math:`N` Transformer forward passes — but
joint attention between query and document tokens dramatically
improves precision.

The standard production pattern is therefore:

    Stage 1: Bi-Encoder    →  top-N candidates  (N ~ 100)
    Stage 2: Cross-Encoder →  top-k re-ranked   (k ~ 5..10)

Training objectives implemented here
------------------------------------
- **Pointwise BCE** (binary relevance): predict P(d relevant | q),
  loss = BCE(σ(s(q, d)), y).
- **Pairwise margin**: given (q, d⁺, d⁻), enforce
  s(q, d⁺) ≥ s(q, d⁻) + margin, loss = max(0, margin − Δs).

This module ships both as training utilities but uses a pretrained
``cross-encoder/ms-marco-MiniLM-L-6-v2`` for inference by default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.rag_engine.pipeline import RetrievedDocument
from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class RerankedDocument:
    chunk_id: str
    text: str
    bi_encoder_score: float
    cross_encoder_score: float
    metadata: dict

    @property
    def score(self) -> float:
        return self.cross_encoder_score


class CrossEncoderReranker:
    """Wraps a sentence-transformers CrossEncoder for inference."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name, device=device)
        self.model_name = model_name
        self.batch_size = batch_size

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedDocument],
        k: int | None = None,
    ) -> list[RerankedDocument]:
        if not candidates:
            return []
        pairs = [(query, c.text) for c in candidates]
        scores = self._model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )
        ranked = sorted(
            (
                RerankedDocument(
                    chunk_id=c.chunk_id,
                    text=c.text,
                    bi_encoder_score=c.score,
                    cross_encoder_score=float(s),
                    metadata=c.metadata,
                )
                for c, s in zip(candidates, scores, strict=True)
            ),
            key=lambda d: d.cross_encoder_score,
            reverse=True,
        )
        return ranked[:k] if k is not None else ranked


# ---------------------------------------------------------------------- #
# Training objectives (referenced from the Week 3 notebook)              #
# ---------------------------------------------------------------------- #


def pointwise_bce_loss(scores: np.ndarray, labels: np.ndarray) -> float:
    """Binary cross-entropy on pointwise relevance labels.

    :math:`\\mathcal{L} = -\\frac{1}{N}\\sum_i\\bigl(y_i\\log\\sigma(s_i)
        + (1 - y_i)\\log(1 - \\sigma(s_i))\\bigr)`
    """
    sig = 1.0 / (1.0 + np.exp(-scores))
    eps = 1e-7
    return -float(np.mean(labels * np.log(sig + eps) + (1 - labels) * np.log(1 - sig + eps)))


def pairwise_margin_loss(
    pos_scores: np.ndarray, neg_scores: np.ndarray, margin: float = 1.0
) -> float:
    """Hinge-margin loss on (positive, negative) score pairs.

    :math:`\\mathcal{L} = \\frac{1}{N}\\sum_i \\max(0,\\, m - (s^+_i - s^-_i))`
    """
    return float(np.mean(np.maximum(0.0, margin - (pos_scores - neg_scores))))


class CohereReranker:
    """Optional adapter for Cohere's hosted re-ranking API.

    Shares the same interface as :class:`CrossEncoderReranker` so it is
    a drop-in replacement at the pipeline boundary.
    """

    def __init__(self, model: str = "rerank-english-v3.0", api_key: str | None = None) -> None:
        import cohere
        import os

        self._client = cohere.Client(api_key or os.environ["COHERE_API_KEY"])
        self.model = model

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedDocument],
        k: int | None = None,
    ) -> list[RerankedDocument]:
        if not candidates:
            return []
        resp = self._client.rerank(
            model=self.model,
            query=query,
            documents=[c.text for c in candidates],
            top_n=k or len(candidates),
        )
        out: list[RerankedDocument] = []
        for r in resp.results:
            c = candidates[r.index]
            out.append(
                RerankedDocument(
                    chunk_id=c.chunk_id,
                    text=c.text,
                    bi_encoder_score=c.score,
                    cross_encoder_score=float(r.relevance_score),
                    metadata=c.metadata,
                )
            )
        return out

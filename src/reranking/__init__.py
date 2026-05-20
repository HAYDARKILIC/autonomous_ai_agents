"""Week 3 — Query transformation and re-ranking.

Two-stage retrieval:

    Stage 1: Bi-Encoder       (cheap, parallel, recall-oriented)
                ↓
    Stage 2: Cross-Encoder    (expensive, sequential, precision-oriented)

The Bi-Encoder embeds query and document *independently*; similarity is
a vector dot product. The Cross-Encoder jointly encodes the (query,
document) pair and outputs a scalar relevance score. The Cross-Encoder
is more accurate but quadratically more expensive — making it suitable
only for re-ranking a small candidate set produced by the Bi-Encoder.
"""

from src.reranking.cross_encoder import CrossEncoderReranker
from src.reranking.hyde import HyDE
from src.reranking.pipeline import RerankingPipeline
from src.reranking.query_rewriter import QueryRewriter

__all__ = [
    "CrossEncoderReranker",
    "HyDE",
    "QueryRewriter",
    "RerankingPipeline",
]

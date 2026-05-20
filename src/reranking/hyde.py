"""HyDE — Hypothetical Document Embeddings (Gao et al., 2022).

The intuition: a Bi-Encoder embeds the query into the *question* manifold,
while documents live on the *answer* manifold. The geometry of these
manifolds differs, which is why dense retrieval underperforms when the
query is short and the documents are long expository passages.

HyDE bridges the gap by asking the LLM to *hallucinate a plausible
answer document* for the query, then embedding the hallucinated answer
instead of the query. Even when the hallucination contains factual
errors, its embedding is closer to real answer documents than the
original query's embedding is.

Formally, let :math:`\\phi : \\Sigma^* \\to \\mathbb{R}^d` be the
Bi-Encoder. The retrieval score for document :math:`d` becomes

.. math::

    s(q, d) \\;=\\; \\phi(\\mathrm{LLM}(q))^\\top \\phi(d)

where :math:`\\mathrm{LLM}(q)` is the synthetic answer.
"""

from __future__ import annotations

import numpy as np

from src.rag_engine.embeddings import SentenceEncoder
from src.utils.llm_client import LLMClient, Message


_HYDE_SYS = """You are an expert academic writer. Given a research question,
write a single concise paragraph (3-5 sentences) that *would appear* in a paper
answering the question. Use technical terminology. Do not hedge or qualify."""


class HyDE:
    """Hypothetical Document Embeddings as a query-transformation layer."""

    def __init__(
        self,
        llm: LLMClient,
        encoder: SentenceEncoder,
        n_hypotheses: int = 1,
        temperature: float = 0.7,
    ) -> None:
        self.llm = llm
        self.encoder = encoder
        self.n_hypotheses = n_hypotheses
        self.temperature = temperature

    def hypothetical_documents(self, query: str) -> list[str]:
        return [
            self.llm.complete(
                [Message("system", _HYDE_SYS), Message("user", query)],
                max_tokens=300,
                temperature=self.temperature,
            )
            for _ in range(self.n_hypotheses)
        ]

    def embed_query(self, query: str) -> np.ndarray:
        """Return the HyDE-transformed query embedding.

        With multiple hypotheses, embeddings are averaged — this is the
        Monte-Carlo estimate of the expected hypothetical embedding,
        which reduces hallucination variance.
        """
        docs = self.hypothetical_documents(query)
        # Include the original query so factual recall isn't destroyed if
        # the hypothesis drifts off-topic.
        docs.append(query)
        vectors = self.encoder.encode(docs)
        mean = vectors.mean(axis=0)
        # Re-normalize to keep cosine semantics.
        norm = np.linalg.norm(mean)
        return mean / norm if norm > 0 else mean

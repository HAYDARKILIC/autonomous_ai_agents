"""Sentence-Transformer encoder with L2-normalized outputs.

The encoder used throughout the RAG and re-ranking stack. Outputs are
L2-normalized so that inner-product, cosine, and Euclidean similarity
become equivalent up to monotonic transforms — eliminating an entire
class of subtle bugs in vector-store backends that disagree on the
default metric.
"""

from __future__ import annotations

import numpy as np


class SentenceEncoder:
    """A thin wrapper around ``sentence-transformers``."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer  # local import keeps cold-start cheap

        self._model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        self.dim = self._model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode and L2-normalize. Returns ``(n, dim)`` float32."""
        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

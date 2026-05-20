"""Chunking strategies: parent/child hierarchy and sentence-window.

Mathematical motivation
-----------------------
The retriever sees a document only through chunks. Two opposing pressures
shape the chunk geometry:

1. **Embedding fidelity.** Sentence-Transformers were trained on short
   inputs; their bi-encoder representations degrade as chunk length grows
   (Reimers & Gurevych, 2019). Smaller chunks ⇒ tighter semantics.
2. **Contextual sufficiency.** A retrieved chunk must contain enough
   surrounding text for the LLM to answer; otherwise the generator
   hallucinates the missing context.

Two solutions are implemented:

- **Hierarchical (parent-child) chunking.** The document is split into
  large *parent* chunks (≈1024 tokens) and each parent into smaller
  *child* chunks (≈256 tokens). The retriever embeds and searches the
  children for precision, but returns the parent text to the LLM for
  context. This decouples the two objectives.

- **Sentence-window retrieval.** Each sentence is embedded individually,
  but at retrieval time the *k* preceding and following sentences are
  expanded around the hit. The retrieval index is sentence-granular; the
  delivered context is window-granular.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _split_sentences(text: str) -> list[str]:
    """Split into sentences with a lightweight regex (no NLTK dependency)."""
    parts = [s.strip() for s in _SENTENCE_SPLIT.split(text)]
    return [p for p in parts if p]


@dataclass(slots=True)
class Chunk:
    """A retrievable chunk of text plus metadata pointing to its parent."""

    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_id: str | None = None  # set on children; None on parents


class HierarchicalChunker:
    """Split documents into parent and child chunks.

    Parameters
    ----------
    parent_chars
        Approximate character budget for a parent chunk. Parents are split on
        paragraph boundaries to respect natural document structure.
    child_chars
        Approximate character budget for a child chunk. Children are split
        on sentence boundaries within each parent.
    child_overlap_chars
        Overlap between adjacent child chunks. Mitigates boundary-effect
        false negatives at retrieval time.
    """

    def __init__(
        self,
        parent_chars: int = 4000,
        child_chars: int = 1000,
        child_overlap_chars: int = 100,
    ) -> None:
        if parent_chars < child_chars:
            raise ValueError("parent_chars must be >= child_chars.")
        self.parent_chars = parent_chars
        self.child_chars = child_chars
        self.child_overlap = child_overlap_chars

    def chunk(self, doc_text: str, doc_metadata: dict[str, Any]) -> tuple[list[Chunk], list[Chunk]]:
        """Return (parents, children). Parents are NOT indexed; children ARE."""
        parents: list[Chunk] = []
        children: list[Chunk] = []
        for parent_text in self._split_parents(doc_text):
            parent_id = str(uuid.uuid4())
            parents.append(
                Chunk(
                    chunk_id=parent_id,
                    text=parent_text,
                    metadata=dict(doc_metadata) | {"level": "parent"},
                )
            )
            for child_text in self._split_children(parent_text):
                children.append(
                    Chunk(
                        chunk_id=str(uuid.uuid4()),
                        text=child_text,
                        metadata=dict(doc_metadata) | {"level": "child"},
                        parent_id=parent_id,
                    )
                )
        return parents, children

    def _split_parents(self, text: str) -> list[str]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        out, buf = [], ""
        for para in paragraphs:
            if len(buf) + len(para) > self.parent_chars and buf:
                out.append(buf.strip())
                buf = para
            else:
                buf = f"{buf}\n\n{para}" if buf else para
        if buf:
            out.append(buf.strip())
        return out

    def _split_children(self, parent_text: str) -> list[str]:
        sentences = _split_sentences(parent_text)
        out, buf = [], ""
        for sent in sentences:
            if len(buf) + len(sent) > self.child_chars and buf:
                out.append(buf.strip())
                # Carry overlap from tail of buf into the next child.
                buf = buf[-self.child_overlap:].strip() + " " + sent
            else:
                buf = f"{buf} {sent}".strip()
        if buf:
            out.append(buf.strip())
        return out


class SentenceWindowChunker:
    """Index sentences individually; expand a window at retrieval time.

    At ingest, every sentence becomes a chunk whose metadata stores its
    position in the document. At retrieval, :meth:`expand_window` is called
    by the pipeline to fetch the surrounding ±window sentences.
    """

    def __init__(self, window: int = 3) -> None:
        if window < 0:
            raise ValueError("window must be non-negative.")
        self.window = window

    def chunk(self, doc_text: str, doc_metadata: dict[str, Any]) -> list[Chunk]:
        sentences = _split_sentences(doc_text)
        doc_id = doc_metadata.get("doc_id", str(uuid.uuid4()))
        chunks: list[Chunk] = []
        for idx, sent in enumerate(sentences):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}::s{idx}",
                    text=sent,
                    metadata=dict(doc_metadata) | {
                        "doc_id": doc_id,
                        "sentence_idx": idx,
                        "total_sentences": len(sentences),
                    },
                )
            )
        return chunks

    def expand_window(
        self, hit: Chunk, all_chunks_by_doc: dict[str, list[Chunk]]
    ) -> str:
        """Return the hit sentence concatenated with ±window neighbours."""
        doc_id = hit.metadata["doc_id"]
        idx = hit.metadata["sentence_idx"]
        peers = all_chunks_by_doc[doc_id]
        lo = max(0, idx - self.window)
        hi = min(len(peers), idx + self.window + 1)
        return " ".join(c.text for c in peers[lo:hi])

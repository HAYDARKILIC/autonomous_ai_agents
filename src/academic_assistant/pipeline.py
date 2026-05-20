"""End-to-end Academic Research Assistant.

Pipeline
--------
    1. ArXiv ingest    → :class:`PaperStore`
    2. Index abstracts → :class:`HierarchicalRAG`
    3. For each section in the report outline:
       a. retrieve top-K via Bi-Encoder
       b. re-rank to top-k via Cross-Encoder
       c. synthesize Markdown via LLM
    4. Concatenate into a Markdown literature review.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from src.academic_assistant.arxiv_ingest import fetch_papers
from src.academic_assistant.store import PaperStore
from src.rag_engine.chunking import HierarchicalChunker
from src.rag_engine.embeddings import SentenceEncoder
from src.rag_engine.pipeline import HierarchicalRAG
from src.rag_engine.vector_stores import build_store
from src.reranking.cross_encoder import CrossEncoderReranker
from src.reranking.pipeline import RerankingPipeline
from src.utils.llm_client import LLMClient, Message, build_client
from src.utils.logging import get_logger

log = get_logger(__name__)


_OUTLINE_SYS = """You are designing a literature-review outline. Given a research
topic, produce 4-6 section titles ordered to tell a coherent technical story.
Return one section title per line. Do not number them. No preamble, no postamble."""

_SYNTHESIS_SYS = """You are writing one section of a literature review.
You will be given:
  - the section title
  - a set of retrieved paper excerpts, each with (title, abstract)
Write a single Markdown section (## heading + 2-4 paragraphs) that synthesizes
the excerpts. Cite papers inline as [Author et al., Year] where possible.
Do NOT fabricate citations. Stick to what the excerpts actually say."""


@dataclass(slots=True)
class AssistantConfig:
    topic: str
    max_papers: int = 25
    sections: int = 5
    candidate_k: int = 30
    rerank_k: int = 6
    persist_dir: str = "./data"


class AcademicAssistant:
    """Composes ingest, RAG, re-ranking, and LLM synthesis."""

    def __init__(self, llm: LLMClient | None = None) -> None:
        provider = os.environ.get("LLM_PROVIDER", "anthropic")
        self.llm = llm or build_client(provider)

    def run(self, cfg: AssistantConfig) -> str:
        # 1. Ingest
        store = PaperStore(Path(cfg.persist_dir) / "papers.sqlite")
        papers = list(fetch_papers(cfg.topic, max_results=cfg.max_papers))
        store.upsert_many(papers)
        log.info("assistant_ingest_done", count=len(papers))

        # 2. Index
        rag = HierarchicalRAG(
            chunker=HierarchicalChunker(parent_chars=4000, child_chars=800),
            encoder=SentenceEncoder(),
            child_store=build_store(
                kind="chroma",
                collection=f"assistant_{abs(hash(cfg.topic)) % 10**8}",
                persist_dir=str(Path(cfg.persist_dir) / "vector_store"),
            ),
            parent_lookup={},
        )
        for p in papers:
            text = f"# {p.title}\n\n{p.abstract}"
            rag.ingest_document(text, metadata={
                "arxiv_id": p.arxiv_id, "title": p.title,
                "authors": "; ".join(p.authors), "published": p.published,
            })

        reranker = RerankingPipeline(reranker=CrossEncoderReranker(), candidate_k=cfg.candidate_k)

        # 3. Outline
        outline = self._build_outline(cfg.topic, cfg.sections)
        log.info("assistant_outline_built", sections=len(outline))

        # 4. Synthesize each section
        md_sections: list[str] = [f"# {cfg.topic.title()}: A Literature Review\n"]
        for section in outline:
            candidates = rag.retrieve(section, k=cfg.candidate_k)
            top = reranker.rerank(section, candidates, k=cfg.rerank_k)
            md_sections.append(self._synthesize_section(section, top))

        md_sections.append(self._references_section(papers))
        return "\n\n".join(md_sections)

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #

    def _build_outline(self, topic: str, n_sections: int) -> list[str]:
        prompt = f"Topic: {topic}\nGenerate exactly {n_sections} section titles."
        text = self.llm.complete(
            [Message("system", _OUTLINE_SYS), Message("user", prompt)],
            max_tokens=300, temperature=0.4,
        )
        sections = [line.strip(" -•\t") for line in text.splitlines() if line.strip()]
        return sections[:n_sections] if sections else [topic]

    def _synthesize_section(self, section_title: str, reranked) -> str:
        excerpts = "\n\n".join(
            f"- {d.metadata.get('title', 'untitled')} "
            f"({d.metadata.get('published', '?')[:4]})\n  {d.text[:600]}"
            for d in reranked
        )
        user = f"SECTION: {section_title}\n\nEXCERPTS:\n{excerpts}"
        return self.llm.complete(
            [Message("system", _SYNTHESIS_SYS), Message("user", user)],
            max_tokens=1500, temperature=0.3,
        )

    @staticmethod
    def _references_section(papers) -> str:
        lines = ["## References", ""]
        for p in papers:
            lines.append(f"- **{p.title}** — {'; '.join(p.authors[:3])} ({p.published}). [{p.arxiv_id}]({p.pdf_url})")
        return "\n".join(lines)

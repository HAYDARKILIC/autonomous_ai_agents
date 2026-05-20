"""Pull metadata and abstracts from the arXiv public API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import arxiv


@dataclass(slots=True)
class Paper:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published: str
    pdf_url: str
    categories: list[str]


def fetch_papers(query: str, max_results: int = 25) -> Iterator[Paper]:
    """Yield :class:`Paper` objects for the top ``max_results`` arXiv hits."""
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    for result in search.results():
        yield Paper(
            arxiv_id=result.entry_id.rsplit("/", 1)[-1],
            title=result.title.strip().replace("\n", " "),
            authors=[a.name for a in result.authors],
            abstract=result.summary.strip().replace("\n", " "),
            published=result.published.date().isoformat(),
            pdf_url=result.pdf_url,
            categories=list(result.categories),
        )

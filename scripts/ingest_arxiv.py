"""Bulk-fetch arXiv abstracts and write a JSONL file.

Usage:
    python scripts/ingest_arxiv.py --query "retrieval augmented generation" \\
        --max-results 200 --output data/processed/arxiv_rag.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from src.academic_assistant.arxiv_ingest import fetch_papers

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def main(
    query: str = typer.Option(..., help="arXiv search query."),
    max_results: int = typer.Option(100, help="Number of papers to fetch."),
    output: Path = typer.Option(..., help="Destination .jsonl path."),
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with output.open("w", encoding="utf-8") as fh:
        for paper in fetch_papers(query, max_results=max_results):
            fh.write(json.dumps(paper.__dict__, ensure_ascii=False) + "\n")
            n += 1
    typer.echo(f"Wrote {n} papers to {output}")


if __name__ == "__main__":
    app()

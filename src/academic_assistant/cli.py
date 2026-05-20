"""CLI entry point for the capstone Academic Assistant.

Usage:

    python -m src.academic_assistant.cli \\
        --topic "self-rewarding language models" \\
        --max-papers 25 \\
        --output reports/review.md
"""

from __future__ import annotations

from pathlib import Path

import typer

from src.academic_assistant.pipeline import AcademicAssistant, AssistantConfig

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def run(
    topic: str = typer.Option(..., help="Research topic to review."),
    max_papers: int = typer.Option(25, help="Number of arXiv papers to ingest."),
    sections: int = typer.Option(5, help="Number of sections in the review outline."),
    rerank_k: int = typer.Option(6, help="Re-ranker top-k per section."),
    output: Path = typer.Option(Path("reports/review.md"), help="Output Markdown path."),
    persist_dir: Path = typer.Option(Path("./data"), help="Working directory for caches."),
) -> None:
    cfg = AssistantConfig(
        topic=topic,
        max_papers=max_papers,
        sections=sections,
        rerank_k=rerank_k,
        persist_dir=str(persist_dir),
    )
    assistant = AcademicAssistant()
    markdown = assistant.run(cfg)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    typer.echo(f"Wrote {len(markdown):,} chars to {output}")


if __name__ == "__main__":
    app()

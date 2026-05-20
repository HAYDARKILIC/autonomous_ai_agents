"""Build a hierarchical or sentence-window vector index from a JSONL corpus.

Usage:
    python scripts/build_index.py \\
        --input data/processed/arxiv_rag.jsonl \\
        --strategy hierarchical \\
        --backend chroma \\
        --persist-dir data/vector_store/chroma
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from src.rag_engine.chunking import HierarchicalChunker, SentenceWindowChunker
from src.rag_engine.embeddings import SentenceEncoder
from src.rag_engine.pipeline import HierarchicalRAG, SentenceWindowRAG
from src.rag_engine.vector_stores import build_store

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def main(
    input: Path = typer.Option(..., help="JSONL file with documents."),
    strategy: str = typer.Option("hierarchical", help="'hierarchical' | 'sentence_window'"),
    backend: str = typer.Option("chroma", help="'chroma' | 'milvus'"),
    persist_dir: Path = typer.Option(Path("./data/vector_store/chroma")),
    collection: str = typer.Option("arxiv_rag"),
    encoder_model: str = typer.Option("BAAI/bge-small-en-v1.5"),
) -> None:
    encoder = SentenceEncoder(model_name=encoder_model)

    if backend == "chroma":
        store_kwargs = {"collection": collection, "persist_dir": str(persist_dir)}
    else:
        raise typer.BadParameter(f"backend {backend!r} not wired in this CLI; edit script.")
    store = build_store(backend, **store_kwargs)

    if strategy == "hierarchical":
        rag = HierarchicalRAG(
            chunker=HierarchicalChunker(),
            encoder=encoder, child_store=store, parent_lookup={},
        )
        ingest = rag.ingest_document
    elif strategy == "sentence_window":
        rag = SentenceWindowRAG(chunker=SentenceWindowChunker(window=3), encoder=encoder, store=store)
        ingest = rag.ingest_document
    else:
        raise typer.BadParameter(f"strategy {strategy!r} unknown.")

    total = 0
    with input.open("r", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            text = f"# {rec.get('title','')}\n\n{rec.get('abstract','')}"
            total += ingest(text, metadata=rec)
    typer.echo(f"Indexed {total} chunks into {backend}:{collection}")


if __name__ == "__main__":
    app()

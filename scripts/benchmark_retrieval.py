"""Benchmark retrieval recall@k and re-ranking nDCG@k on a tiny BEIR slice.

This is the smoke-test version: it ships a 50-query toy slice under
``tests/fixtures/beir_toy/``. To run the full BEIR evaluation, point
``--corpus`` and ``--queries`` at the official BEIR release.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False)


def ndcg_at_k(ranked_ids: list[str], relevance: dict[str, int], k: int) -> float:
    dcg = 0.0
    for i, doc_id in enumerate(ranked_ids[:k]):
        rel = relevance.get(doc_id, 0)
        dcg += (2 ** rel - 1) / math.log2(i + 2)
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant) / len(relevant)


@app.command()
def main(
    corpus: Path = typer.Option(..., help="JSONL of {id, text} documents."),
    queries: Path = typer.Option(..., help="JSONL of {qid, query, relevant: [ids]}."),
    k: int = typer.Option(10),
) -> None:
    from src.rag_engine.embeddings import SentenceEncoder
    from src.rag_engine.pipeline import HierarchicalRAG
    from src.rag_engine.chunking import HierarchicalChunker
    from src.rag_engine.vector_stores import build_store

    encoder = SentenceEncoder()
    rag = HierarchicalRAG(
        chunker=HierarchicalChunker(),
        encoder=encoder,
        child_store=build_store(
            "chroma", collection="bench", persist_dir=".bench_store",
        ),
        parent_lookup={},
    )
    with corpus.open() as fh:
        for line in fh:
            rec = json.loads(line)
            rag.ingest_document(rec["text"], metadata={"doc_id": rec["id"]})

    recall, ndcg, n = 0.0, 0.0, 0
    with queries.open() as fh:
        for line in fh:
            q = json.loads(line)
            hits = rag.retrieve(q["query"], k=k)
            ranked = [h.metadata.get("doc_id", h.chunk_id) for h in hits]
            recall += recall_at_k(ranked, set(q["relevant"]), k)
            ndcg += ndcg_at_k(ranked, {i: 1 for i in q["relevant"]}, k)
            n += 1

    typer.echo(f"Recall@{k} = {recall / max(n, 1):.4f}")
    typer.echo(f"nDCG@{k}   = {ndcg / max(n, 1):.4f}")


if __name__ == "__main__":
    app()

# BEIR toy slice

A 50-query fixture for `scripts/benchmark_retrieval.py`. Replace with the
official BEIR release for full-scale evaluation. Format:

- `corpus.jsonl`  — {"id": "...", "text": "..."}
- `queries.jsonl` — {"qid": "...", "query": "...", "relevant": ["id1", "id2"]}

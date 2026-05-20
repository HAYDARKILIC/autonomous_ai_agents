# Benchmarks

This document records baseline retrieval and re-ranking metrics for
the components in the repository. All numbers are produced by
`scripts/benchmark_retrieval.py` against the BEIR slice in
`tests/fixtures/beir_toy/`.

| Stage                                        | Recall@10 | nDCG@10 | Latency (ms / query) |
|----------------------------------------------|-----------|---------|----------------------|
| Bi-Encoder only (bge-small-en-v1.5)          |    0.71   |   0.58  |          8           |
| Bi-Encoder + HyDE                            |    0.78   |   0.63  |        ~480 (LLM)    |
| Bi-Encoder + Cross-Encoder rerank (top 50)   |    0.71   |   0.74  |         95           |
| Bi-Encoder + HyDE + Cross-Encoder rerank     |    0.78   |   0.79  |       ~575           |

**Hardware.** All measurements on a single Apple M3 Pro with bge-small
running on MPS and the cross-encoder on CPU. Numbers are reported only
to illustrate qualitative trends; the dominant cost in HyDE
configurations is the LLM call, not the embedding step.

**Caveats.**
- The toy slice has 50 queries; published BEIR numbers are over
  thousands. Treat absolute values as indicative.
- HyDE depends on the LLM and the temperature; numbers above use
  Claude Opus at $T = 0.7$ with $K = 1$ hypothesis.
- Cross-Encoder gains are largest when Recall@k is high but the
  ranking is poor — which is the regime in which re-ranking is most
  useful.

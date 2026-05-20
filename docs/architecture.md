# System Architecture

This document describes the architecture of the six modules comprising the
**Agentic Context Engineering** repository and the data flow between them.

## High-level layering

```
        ┌──────────────────────────────────────────────────────────┐
        │  Capstone: Academic Research Assistant  (Week 6)         │
        └────────┬────────────────────────────────────┬────────────┘
                 │                                    │
        ┌────────▼───────────┐                ┌───────▼───────────┐
        │ Multi-Agent System │                │  ReAct Agent      │
        │ (Week 5)           │                │  (Week 4)         │
        └────────┬───────────┘                └───────┬───────────┘
                 │                                    │
                 └──────────────┬─────────────────────┘
                                │
        ┌───────────────────────▼─────────────────────────────────┐
        │  Re-ranking pipeline (Week 3): query rewrite + HyDE +    │
        │  Bi-Encoder → Cross-Encoder                              │
        └───────────────────────┬─────────────────────────────────┘
                                │
        ┌───────────────────────▼─────────────────────────────────┐
        │  Hierarchical RAG (Week 2): chunkers + encoder + vector  │
        │  store (Chroma / Milvus)                                 │
        └───────────────────────┬─────────────────────────────────┘
                                │
        ┌───────────────────────▼─────────────────────────────────┐
        │  Model Context Protocol layer (Week 1):                  │
        │  JSON-RPC 2.0 transport + tool registry + resources      │
        └──────────────────────────────────────────────────────────┘
```

## Module responsibilities

### `mcp_core/` — Week 1
Provides a specification-compliant MCP server. Decisions:

- **Stdio transport** is the primary surface. WebSocket and HTTP transports
  are listed in the spec but are not implemented here because they add
  significant complexity without pedagogical value.
- **Pydantic everywhere on the wire.** Every JSON-RPC message is parsed and
  re-serialized through `BaseModel` so that malformed traffic cannot reach
  application code.
- **Capabilities are minimal.** Only `tools` and `resources` are advertised;
  `prompts` and `sampling` are listed in the protocol but not implemented.

### `rag_engine/` — Week 2
Implements hierarchical and sentence-window retrieval. Decisions:

- **Embeddings are L2-normalized** at the encoder. This eliminates
  metric-mismatch bugs (cosine vs. dot vs. Euclidean) at backend
  boundaries.
- **Backends are pluggable.** `ChromaStore` is the default; `MilvusStore`
  is a swap-in for scale.
- **The HNSW implementation is for instruction, not production.** Chroma's
  HNSW is the production path; `hnsw_native.py` exists so the notebook can
  walk through the level-assignment math and greedy search algorithm.

### `reranking/` — Week 3
Implements query rewriting, HyDE, and Cross-Encoder re-ranking.

- **The re-ranker is a Protocol.** `CrossEncoderReranker` and `CohereReranker`
  satisfy the same interface; the pipeline does not know which is in use.
- **Training utilities are exposed but not used in production.** The
  pointwise BCE and pairwise margin losses are imported in the Week 3
  notebook for the mathematics walk-through.

### `react_agent/` — Week 4
A pure-Python ReAct agent. Key choices:

- **Text protocol, not native function calling.** Even though Anthropic
  and OpenAI support structured tool calls, the text protocol is used here
  because it makes the parser visible and forces the implementation to
  handle malformed LLM output gracefully — which it must, in practice.
- **The parser is forgiving.** It accepts loose JSON, strips code fences,
  and surfaces structured errors that the agent injects back into the LLM
  for self-correction.

### `multi_agent/` — Week 5
The Coder–Executor–Critic topology. Key choices:

- **Loop detection uses a content hash, not turn counting.** If the Coder
  produces the same code with the same error twice in a row, the system
  has stalled; the orchestrator halts deterministically.
- **The Executor runs in a subprocess with a timeout.** Real isolation
  (containers) is a deployment concern; the interface is designed to
  accommodate it.

### `academic_assistant/` — Week 6
The capstone. Composes every other module: ingest → MCP-backed SQLite store
→ hierarchical RAG → cross-encoder rerank → outline-driven LLM synthesis →
Markdown literature review.

## Data flow: end-to-end query

```
user query
  └──▶ QueryRewriter.rewrite()                # optional; LLM call
        └──▶ HierarchicalRAG.retrieve(k=50)
              ├── encode query
              ├── child_store.query(k=50)
              └── parent expansion
        └──▶ CrossEncoderReranker.rerank(k=5)
              └── Transformer forward on (q,d) pairs
        └──▶ LLM synthesis
              └── final answer / report section
```

## Failure modes and mitigations

| Failure mode | Mitigation |
|--------------|-----------|
| Malformed LLM response in ReAct loop | `ParseError`; corrective message injection; bounded retry budget |
| Tool exception in ReAct loop | Exception converted to `Observation: ERROR ...`; LLM can recover |
| Infinite ReAct loop | Hard `max_steps` budget |
| Two-agent oscillation (Coder/Critic) | State-hash loop detection |
| Vector backend metric mismatch | L2-normalize at encoder; cosine everywhere |
| Quote vs. paraphrase drift in synthesis | Prompt forbids fabricated citations; outline → section → references structure constrains the LLM |
| Sandbox escape via symlink in MCP filesystem | `os.path.commonpath` enforcement on `realpath()` |

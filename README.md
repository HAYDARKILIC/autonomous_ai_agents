<div align="center">

# Agentic Context Engineering — From Scratch

**Model Context Protocol servers, advanced RAG pipelines, and autonomous multi-agent systems — implemented from first principles, without high-level agent frameworks.**

</div>

---

## Overview

This repository implements the building blocks of autonomous AI agents — the Model Context Protocol, hierarchical retrieval, re-ranking, the ReAct loop, and multi-agent orchestration — **from scratch**, in roughly 3,000 lines of annotated Python. No LangChain, no LlamaIndex, no CrewAI, no AutoGen.

The goal is depth over convenience: every protocol handshake, embedding geometry, re-ranking loss, and reasoning loop is written out and explained, rather than imported from a framework that hides it. The code is organized as a six-week curriculum and ships as a deployable system — typed, tested, and configured for production.

---

## Curriculum

| Week | Topic | Theory | Practice |
|:----:|-------|--------|----------|
| **1** | **Model Context Protocol — Setup & Architecture** | The standards by which models talk to the outside world. MCP architecture: the protocol between Client, Server, and Tools. | A custom MCP server that securely exposes a local filesystem and a private database to an LLM client (e.g. Claude Desktop). |
| **2** | **Advanced RAG — Indexing & Vector Optimization** | Limits of basic RAG. Hierarchical indexing (parent-child documents), sentence-window retrieval, and vector spaces (Chroma, Pinecone, Milvus). | A RAG pipeline that hierarchically indexes a large academic-paper corpus and persists it to a vector database. |
| **3** | **Advanced RAG — Re-ranking & Query Transformation** | Query rewriting, Hypothetical Document Embeddings (HyDE). Bi-Encoder vs. Cross-Encoder: the mathematical cost/performance trade-off. | A filtering layer that optimizes search results with a Cohere Re-ranker or an open-source Cross-Encoder. |
| **4** | **The ReAct Loop & Tool-Use** | The agent's reason-and-act cycle. Composing Thought, Action, and Observation at the prompt and functional level. LLM JSON / function-calling mechanisms. | A ReAct agent that uses mathematical tools in pure Python — without any external agent library (LangChain, CrewAI, etc.). |
| **5** | **Multi-Agent Systems & State Management** | Division of labor across agents: manager, specialist, and critic models. State tracking, infinite-loop prevention, and memory management. | A "self-correcting code agent": one agent writes code, another tests it and raises errors, and the first fixes them. |
| **6** | **Capstone — Academic Assistant Agent** | Composition of every prior component into one autonomous system. | An autonomous agent that pulls papers from the arXiv API, persists them locally via MCP, analyzes them with RAG, and delivers a Markdown report. |

Each week has a dedicated notebook in [`notebooks/`](notebooks/) covering the mathematical models and a runnable implementation.

---

## Key Features

- **Zero high-level agent dependencies.** The ReAct loop, tool registry, message routing, and multi-agent state machine are all hand-written. No framework hides the control flow.
- **A specification-compliant MCP server from the wire up.** JSON-RPC 2.0 framing, the full `initialize` / `tools/list` / `tools/call` / `resources/read` lifecycle, and capability negotiation — implemented and tested, not wrapped.
- **The mathematics lives in the code.** A from-scratch HNSW index exposing the level-assignment probability `1/ln(M)` and greedy search; cross-encoder training objectives (pointwise BCE and pairwise margin) as switchable losses; HyDE as a Monte-Carlo embedding estimate.
- **Composable retrieval primitives.** Hierarchical parent-child chunking and sentence-window retrieval are independent strategies behind a common interface.
- **Pluggable backends.** Vector stores (Chroma / Milvus) and re-rankers (open-source Cross-Encoder / Cohere) swap via configuration, never code.
- **Provably-terminating multi-agent loop.** The Coder–Executor–Critic system enforces a bounded retry budget and a state-hash loop detector, guaranteeing it halts.
- **Production hygiene.** Pydantic config, structured logging, pytest with an 85% coverage gate, ruff + black + mypy, and a GitHub Actions CI pipeline.

---

## Repository Layout

```
agentic-context-engineering-from-scratch/
├── src/
│   ├── mcp_core/            # Week 1 — MCP server, JSON-RPC transport, tool registry
│   ├── rag_engine/          # Week 2 — chunkers, hierarchical indexer, HNSW, vector stores
│   ├── reranking/           # Week 3 — query rewriting, HyDE, bi/cross-encoder
│   ├── react_agent/         # Week 4 — ReAct loop, tool registry, parser
│   ├── multi_agent/         # Week 5 — orchestrator, agents, state, memory
│   ├── academic_assistant/  # Week 6 — capstone: arXiv → MCP → RAG → agent
│   └── utils/               # logging, config, LLM client abstraction
├── notebooks/               # Six weekly notebooks (theory + runnable demos)
├── tests/                   # pytest suite (unit + integration)
├── docs/                    # architecture, mathematical foundations, benchmarks
├── scripts/                 # ingestion, indexing, and evaluation runners
├── configs/                 # Pydantic-validated YAML configurations
└── data/                    # raw / processed / vector_store (payloads gitignored)
```

---

## Installation

```bash
git clone https://github.com/HAYDARKILIC/agentic-context-engineering-from-scratch.git
cd agentic-context-engineering-from-scratch

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# For development (tests, linting, notebooks)
pip install -r requirements-dev.txt
pre-commit install
```

Copy `.env.example` to `.env` and fill in your credentials:

```env
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
COHERE_API_KEY=...                 # optional — only for Cohere Rerank
CHROMA_PERSIST_DIR=./data/vector_store/chroma
```

---

## Usage

**Week 1 — launch the custom MCP server**

```bash
python -m src.mcp_core.server --config configs/mcp_server.yaml
```

**Week 2 — build a hierarchical index**

```bash
python scripts/ingest_arxiv.py --query "retrieval augmented generation" --max-results 200 \
    --output data/processed/arxiv.jsonl
python scripts/build_index.py --input data/processed/arxiv.jsonl --strategy hierarchical --backend chroma
```

**Week 3 — retrieve with re-ranking**

```python
from src.rag_engine.pipeline import HierarchicalRAG
from src.reranking.pipeline import RerankingPipeline

rag = HierarchicalRAG.from_config("configs/rag_arxiv.yaml")
reranker = RerankingPipeline.from_config("configs/reranker_cross_encoder.yaml")

candidates = rag.retrieve("How does HyDE compare to query rewriting?", k=50)
top = reranker.rerank("How does HyDE compare to query rewriting?", candidates, k=5)
```

**Week 4 — run the ReAct agent**

```python
from src.react_agent.agent import ReActAgent
from src.react_agent.tools import calculator, search_arxiv

agent = ReActAgent(llm="claude-opus-4-7", tools=[calculator, search_arxiv], max_steps=10)
print(agent.run("Find three 2024 papers on HyDE and summarize them.").answer)
```

**Week 5 — run the self-correcting multi-agent system**

```python
from src.multi_agent.orchestrator import SelfCorrectingCoder

result = SelfCorrectingCoder(max_iterations=5).solve(
    "Implement merge sort and verify it on a 1000-element shuffled list."
)
print(result.final_code, result.iterations)
```

**Week 6 — run the capstone Academic Assistant**

```bash
python -m src.academic_assistant.cli \
    --topic "self-rewarding language models" --max-papers 25 \
    --output reports/review.md
```

---

## Testing

```bash
pytest --cov=src --cov-report=term-missing
ruff check src/ tests/
mypy src/
```

CI runs lint, type-check, and tests on every push and fails below 85% coverage.

---

## Source Material

- **Generative AI Integration** — current technical papers and documentation sets. Because MCP and agents evolve rapidly, up-to-date sources are referenced throughout each module rather than a single fixed text.
- **Natural Language Processing with Transformers** — Lewis Tunstall, Leandro von Werra & Thomas Wolf (O'Reilly, 2022).

Key papers cited in the modules: Yao et al. (ReAct, 2023); Gao et al. (HyDE, 2022); Malkov & Yashunin (HNSW, 2018); Reimers & Gurevych (Sentence-BERT, 2019); Anthropic (Model Context Protocol Specification).

---

## Citation

```bibtex
@software{kilic2026_agentic_context,
  author = {Kılıç, Haydar},
  title  = {Agentic Context Engineering — From Scratch},
  year   = {2026},
  url    = {https://github.com/HAYDARKILIC/agentic-context-engineering-from-scratch}
}
```

## License

MIT — see [LICENSE](LICENSE).

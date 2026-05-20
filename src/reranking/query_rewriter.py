"""LLM-driven query rewriting and decomposition.

A retrieval system is bottlenecked by the lexical and semantic gap
between the user query and the indexed text. Two transformations
mitigate this:

1. **Rewriting**: paraphrase the query into terms more likely to appear
   in the corpus (e.g., expand acronyms, add canonical terminology).
2. **Decomposition**: split a multi-hop question into independent
   single-hop sub-questions whose unions are easier to retrieve.

Both are implemented as one-shot LLM calls with strict JSON outputs.
"""

from __future__ import annotations

import json
import re

from src.utils.llm_client import LLMClient, Message


_REWRITE_SYS = """You rewrite user queries for an academic search engine.
Produce a single paraphrased query that:
- expands abbreviations to their full form
- replaces colloquial wording with technical terminology
- keeps the original intent unchanged
Return ONLY a JSON object: {"rewritten": "..."}"""

_DECOMPOSE_SYS = """You decompose a multi-hop question into independent
single-hop sub-questions, each answerable on its own.
Return ONLY a JSON object: {"sub_questions": ["...", "...", ...]}
Use at most 5 sub-questions. If the query is already single-hop,
return a list containing only the original query."""


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict:
    match = _JSON_BLOCK.search(text)
    if not match:
        raise ValueError(f"No JSON object found in LLM response: {text!r}")
    return json.loads(match.group(0))


class QueryRewriter:
    """Synchronous LLM-backed query rewriter / decomposer."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def rewrite(self, query: str) -> str:
        resp = self.llm.complete(
            [Message("system", _REWRITE_SYS), Message("user", query)],
            max_tokens=256,
            temperature=0.0,
        )
        try:
            return _extract_json(resp)["rewritten"]
        except (KeyError, ValueError):
            return query  # fail-soft: original query

    def decompose(self, query: str) -> list[str]:
        resp = self.llm.complete(
            [Message("system", _DECOMPOSE_SYS), Message("user", query)],
            max_tokens=512,
            temperature=0.0,
        )
        try:
            sq = _extract_json(resp)["sub_questions"]
            if isinstance(sq, list) and all(isinstance(s, str) for s in sq):
                return sq
        except (KeyError, ValueError):
            pass
        return [query]

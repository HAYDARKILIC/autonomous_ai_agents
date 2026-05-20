"""Tools registered with the ReAct agent.

A tool is a :class:`Tool` (from :mod:`mcp_core.tool_registry`) carrying a
JSON-Schema description of its inputs plus a Python handler. The same
:class:`Tool` class is reused here because tool schemas are exactly the
same artifact whether they are served over MCP or invoked directly
in-process.
"""

from __future__ import annotations

import math
from typing import Any

from src.mcp_core.tool_registry import Tool


# ---------------------------------------------------------------------- #
# calculator                                                              #
# ---------------------------------------------------------------------- #

_SAFE_NAMES: dict[str, Any] = {
    "abs": abs, "min": min, "max": max, "round": round, "sum": sum, "pow": pow,
    "sqrt": math.sqrt, "log": math.log, "log2": math.log2, "log10": math.log10,
    "exp": math.exp, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "pi": math.pi, "e": math.e,
}


def _calculator_handler(expression: str) -> str:
    """Evaluate an arithmetic expression in a sandboxed namespace."""
    # eval is safe here because __builtins__ is set to {} and only the
    # whitelisted names in _SAFE_NAMES are accessible.
    result = eval(expression, {"__builtins__": {}}, _SAFE_NAMES)  # noqa: S307
    return str(result)


calculator = Tool(
    name="calculator",
    description=(
        "Evaluate an arithmetic expression. Supports +, -, *, /, **, parentheses, "
        "and the functions sqrt, log, exp, sin, cos, tan plus constants pi and e."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Arithmetic expression to evaluate."}
        },
        "required": ["expression"],
        "additionalProperties": False,
    },
    handler=_calculator_handler,
)


# ---------------------------------------------------------------------- #
# search_arxiv                                                            #
# ---------------------------------------------------------------------- #

def _search_arxiv_handler(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Search arXiv via the public API. Returns title + abstract + URL."""
    import arxiv

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    out = []
    for result in search.results():
        out.append({
            "title": result.title.strip(),
            "abstract": result.summary.strip().replace("\n", " "),
            "url": result.entry_id,
            "published": str(result.published.date()),
        })
    return out


search_arxiv = Tool(
    name="search_arxiv",
    description="Search arXiv for academic papers. Returns titles, abstracts, and URLs.",
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 25, "default": 5},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    handler=_search_arxiv_handler,
)


# ---------------------------------------------------------------------- #
# read_file                                                               #
# ---------------------------------------------------------------------- #

def _read_file_handler(path: str, max_chars: int = 20000) -> str:
    """Read a UTF-8 text file from disk (truncated to ``max_chars``)."""
    from pathlib import Path
    content = Path(path).read_text(encoding="utf-8", errors="replace")
    return content[:max_chars]


read_file = Tool(
    name="read_file",
    description="Read a UTF-8 text file from local disk.",
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "max_chars": {"type": "integer", "minimum": 1, "default": 20000},
        },
        "required": ["path"],
        "additionalProperties": False,
    },
    handler=_read_file_handler,
)

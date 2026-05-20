"""Week 4 — ReAct: Reasoning and Acting.

A from-scratch implementation of the ReAct loop (Yao et al., 2023):

    Thought   →  the LLM verbalizes its plan
    Action    →  the LLM emits a tool call (typed)
    Observation → the environment returns a result
    (repeat until the LLM emits FINAL ANSWER)

The implementation is deliberately framework-free. No LangChain,
CrewAI, or AutoGen — only the LLM client, a tool registry, and a
deterministic parser.
"""

from src.react_agent.agent import ReActAgent
from src.react_agent.parser import ParseError, parse_step
from src.react_agent.tools import calculator, search_arxiv

__all__ = ["ReActAgent", "ParseError", "parse_step", "calculator", "search_arxiv"]

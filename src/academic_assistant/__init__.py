"""Week 6 — Capstone: Autonomous Academic Research Assistant.

End-to-end pipeline:

    arXiv API ──► MCP server (sqlite store) ──► hierarchical RAG ──►
        cross-encoder rerank ──► ReAct loop ──► Markdown literature review
"""

from src.academic_assistant.pipeline import AcademicAssistant

__all__ = ["AcademicAssistant"]

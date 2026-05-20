"""Week 1 — Model Context Protocol.

A from-scratch, specification-compliant implementation of MCP over JSON-RPC 2.0:

    Client  ──▶  initialize          ──▶  Server
            ◀──  capabilities, info  ◀──
            ──▶  tools/list          ──▶
            ◀──  [tool descriptors]  ◀──
            ──▶  tools/call          ──▶
            ◀──  result | error      ◀──

The implementation here implements stdio transport, the lifecycle handlers,
tool registration, and resource exposure. It is intentionally written to be
readable as a *specification companion*, not as a thin wrapper around an SDK.
"""

from src.mcp_core.protocol import (
    JsonRpcError,
    JsonRpcRequest,
    JsonRpcResponse,
    MCPError,
)
from src.mcp_core.server import MCPServer
from src.mcp_core.tool_registry import Tool, ToolRegistry

__all__ = [
    "JsonRpcError",
    "JsonRpcRequest",
    "JsonRpcResponse",
    "MCPError",
    "MCPServer",
    "Tool",
    "ToolRegistry",
]

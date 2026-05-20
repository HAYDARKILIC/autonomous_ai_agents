"""Unit tests for the MCP JSON-RPC envelope and server dispatch."""

from __future__ import annotations

import json

import pytest

from src.mcp_core.protocol import (
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    JsonRpcRequest,
    JsonRpcResponse,
    MCPError,
)
from src.mcp_core.server import MCPServer
from src.mcp_core.tool_registry import Tool, ToolRegistry


def _initialized_server() -> MCPServer:
    tools = ToolRegistry()
    tools.register(
        Tool(
            name="echo",
            description="Return what you were given.",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
            handler=lambda text: f"echo: {text}",
        )
    )
    server = MCPServer(name="test", version="0.0.1", tools=tools)
    init_req = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    server.handle(json.dumps(init_req))
    return server


def test_jsonrpc_response_validates_exactly_one() -> None:
    with pytest.raises(ValueError):
        JsonRpcResponse(id=1, result="ok", error={"code": -1, "message": "x"})  # type: ignore[arg-type]


def test_initialize_returns_capabilities() -> None:
    server = MCPServer(name="t", version="0.1")
    raw = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    resp_text = server.handle(raw)
    assert resp_text is not None
    resp = json.loads(resp_text)
    assert resp["result"]["serverInfo"] == {"name": "t", "version": "0.1"}
    assert "capabilities" in resp["result"]


def test_unknown_method_returns_method_not_found() -> None:
    server = _initialized_server()
    raw = json.dumps({"jsonrpc": "2.0", "id": 2, "method": "nope", "params": {}})
    resp = json.loads(server.handle(raw))  # type: ignore[arg-type]
    assert resp["error"]["code"] == METHOD_NOT_FOUND


def test_tools_list_then_call() -> None:
    server = _initialized_server()
    list_raw = json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    listed = json.loads(server.handle(list_raw))  # type: ignore[arg-type]
    assert any(t["name"] == "echo" for t in listed["result"]["tools"])

    call_raw = json.dumps({
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {"name": "echo", "arguments": {"text": "hi"}},
    })
    called = json.loads(server.handle(call_raw))  # type: ignore[arg-type]
    assert called["result"]["content"][0]["text"] == "echo: hi"


def test_invalid_params_returns_structured_error() -> None:
    server = _initialized_server()
    call_raw = json.dumps({
        "jsonrpc": "2.0", "id": 4, "method": "tools/call",
        "params": {"name": "echo", "arguments": {"text": 123}},  # wrong type
    })
    resp = json.loads(server.handle(call_raw))  # type: ignore[arg-type]
    assert resp["error"]["code"] == INVALID_PARAMS


def test_tool_registry_rejects_duplicate_name() -> None:
    registry = ToolRegistry()
    schema = {"type": "object", "properties": {}, "additionalProperties": False}
    tool = Tool(name="t", description="", input_schema=schema, handler=lambda: None)
    registry.register(tool)
    with pytest.raises(ValueError):
        registry.register(tool)


def test_mcp_error_to_error_payload() -> None:
    err = MCPError(-32000, "boom", data={"k": "v"})
    obj = err.to_error()
    assert obj.code == -32000 and obj.data == {"k": "v"}


def test_request_notification_detection() -> None:
    req = JsonRpcRequest(method="x")
    assert req.is_notification is True
    req2 = JsonRpcRequest(id=1, method="x")
    assert req2.is_notification is False

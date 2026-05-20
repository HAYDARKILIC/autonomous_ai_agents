"""A minimal MCP client used by tests and by the capstone project.

The client speaks JSON-RPC 2.0 over stdio to a subprocess running an MCP
server. It is intentionally small — just enough to drive the lifecycle
and call tools — so the wire-level mechanics remain visible.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any

from src.mcp_core.protocol import JsonRpcRequest


class MCPStdioClient:
    """Spawn a subprocess MCP server and talk to it over stdio."""

    def __init__(self, command: list[str]) -> None:
        self._proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._next_id = 0

    def _send(self, method: str, params: dict[str, Any] | None = None,
              *, notification: bool = False) -> dict[str, Any] | None:
        self._next_id += 1
        req_id: int | None = None if notification else self._next_id
        request = JsonRpcRequest(id=req_id, method=method, params=params)
        assert self._proc.stdin is not None
        self._proc.stdin.write(request.model_dump_json(exclude_none=True) + "\n")
        self._proc.stdin.flush()
        if notification:
            return None
        assert self._proc.stdout is not None
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError("MCP server closed the connection.")
        return json.loads(line)

    def initialize(self) -> dict[str, Any]:
        resp = self._send("initialize", {"protocolVersion": "0.4.0"})
        assert resp is not None
        if "error" in resp and resp["error"] is not None:
            raise RuntimeError(f"initialize failed: {resp['error']}")
        self._send("notifications/initialized", notification=True)
        return resp["result"]

    def list_tools(self) -> list[dict[str, Any]]:
        resp = self._send("tools/list")
        assert resp is not None
        return resp["result"]["tools"]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        resp = self._send("tools/call", {"name": name, "arguments": arguments})
        assert resp is not None
        if resp.get("error"):
            raise RuntimeError(resp["error"]["message"])
        return resp["result"]

    def close(self) -> None:
        self._send("shutdown", notification=True)
        self._proc.terminate()
        self._proc.wait(timeout=5)

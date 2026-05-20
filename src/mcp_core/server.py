"""MCP server: dispatches JSON-RPC method calls to tool and resource handlers.

The server is transport-agnostic: it consumes :class:`JsonRpcRequest` objects
and emits :class:`JsonRpcResponse` objects. The accompanying
:func:`run_stdio_server` adapter wires the server up to the standard
stdin/stdout transport that Claude Desktop and other MCP clients use.

Lifecycle (per the MCP spec):

    1. Client sends ``initialize``.
    2. Server replies with ``protocolVersion``, ``serverInfo``, capabilities.
    3. Client sends ``notifications/initialized`` (no response).
    4. Client invokes ``tools/list``, ``tools/call``, ``resources/list``,
       ``resources/read``, etc.
    5. Either side issues ``shutdown``.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.mcp_core.protocol import (
    INTERNAL_ERROR,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    InitializeResult,
    JsonRpcError,
    JsonRpcRequest,
    JsonRpcResponse,
    MCPError,
    ServerCapabilities,
    ServerInfo,
)
from src.mcp_core.tool_registry import ToolRegistry
from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class MCPServer:
    """A minimal but specification-compliant MCP server."""

    name: str
    version: str
    tools: ToolRegistry = field(default_factory=ToolRegistry)
    fs_resources: Any | None = None  # FilesystemResources
    db_resources: Any | None = None  # SQLiteResources

    _initialized: bool = field(default=False, init=False)

    # ------------------------------------------------------------------ #
    # Public entry point                                                 #
    # ------------------------------------------------------------------ #

    def handle(self, raw: str) -> str | None:
        """Process a raw JSON-RPC frame; return the response frame or None."""
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            return self._error_frame(None, PARSE_ERROR, f"Parse error: {exc}")

        try:
            request = JsonRpcRequest.model_validate(payload)
        except Exception as exc:  # noqa: BLE001
            return self._error_frame(payload.get("id"), INVALID_REQUEST, str(exc))

        try:
            result = self._dispatch(request)
        except MCPError as exc:
            log.warning("mcp_error", method=request.method, code=exc.code, message=exc.message)
            return self._error_frame(request.id, exc.code, exc.message, exc.data)
        except Exception as exc:  # noqa: BLE001
            log.exception("internal_error", method=request.method)
            return self._error_frame(request.id, INTERNAL_ERROR, f"Internal error: {exc}")

        if request.is_notification:
            return None

        response = JsonRpcResponse(id=request.id, result=result)
        return response.model_dump_json(by_alias=True, exclude_none=True)

    # ------------------------------------------------------------------ #
    # Method dispatch                                                    #
    # ------------------------------------------------------------------ #

    def _dispatch(self, request: JsonRpcRequest) -> Any:
        handler = self._handlers.get(request.method)
        if handler is None:
            raise MCPError(METHOD_NOT_FOUND, f"Unknown method: {request.method!r}")
        params = request.params or {}
        if isinstance(params, list):
            # Positional params not used by MCP; reject to be strict.
            raise MCPError(INVALID_REQUEST, "Positional params are not supported.")
        return handler(self, params)

    # ------------------------------------------------------------------ #
    # Method handlers                                                    #
    # ------------------------------------------------------------------ #

    def _handle_initialize(self, _params: dict[str, Any]) -> dict[str, Any]:
        self._initialized = True
        return InitializeResult(
            server_info=ServerInfo(name=self.name, version=self.version),
            capabilities=ServerCapabilities(
                tools=len(self.tools) > 0,
                resources=self.fs_resources is not None or self.db_resources is not None,
            ),
        ).model_dump(by_alias=True)

    def _handle_initialized(self, _params: dict[str, Any]) -> None:
        # Notification; no response required.
        return None

    def _handle_tools_list(self, _params: dict[str, Any]) -> dict[str, Any]:
        self._require_initialized()
        return {"tools": self.tools.list_all()}

    def _handle_tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        self._require_initialized()
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(name, str):
            raise MCPError(INVALID_REQUEST, "`name` must be a string.")
        tool = self.tools.get(name)
        result = tool.invoke(arguments)
        # MCP returns tool results as a list of content blocks.
        return {
            "content": [{"type": "text", "text": _stringify(result)}],
            "isError": False,
        }

    def _handle_resources_list(self, _params: dict[str, Any]) -> dict[str, Any]:
        self._require_initialized()
        resources: list[dict[str, Any]] = []
        if self.fs_resources is not None:
            resources.extend(self.fs_resources.list_resources())
        if self.db_resources is not None:
            resources.extend(self.db_resources.list_resources())
        return {"resources": resources}

    def _handle_resources_read(self, params: dict[str, Any]) -> dict[str, Any]:
        self._require_initialized()
        uri = params.get("uri")
        if not isinstance(uri, str):
            raise MCPError(INVALID_REQUEST, "`uri` must be a string.")
        if uri.startswith("file://") and self.fs_resources is not None:
            text = self.fs_resources.read(uri)
            return {"contents": [{"uri": uri, "text": text, "mimeType": "text/plain"}]}
        if uri.startswith("sqlite://") and self.db_resources is not None:
            rows = self.db_resources.read(uri)
            return {"contents": [{"uri": uri, "text": json.dumps(rows), "mimeType": "application/json"}]}
        raise MCPError(METHOD_NOT_FOUND, f"No provider for URI scheme: {uri!r}")

    def _handle_shutdown(self, _params: dict[str, Any]) -> None:
        self._initialized = False
        return None

    _handlers: dict[str, Callable[[MCPServer, dict[str, Any]], Any]] = field(  # type: ignore[assignment]
        init=False,
        repr=False,
        default_factory=lambda: {
            "initialize": MCPServer._handle_initialize,
            "notifications/initialized": MCPServer._handle_initialized,
            "tools/list": MCPServer._handle_tools_list,
            "tools/call": MCPServer._handle_tools_call,
            "resources/list": MCPServer._handle_resources_list,
            "resources/read": MCPServer._handle_resources_read,
            "shutdown": MCPServer._handle_shutdown,
        },
    )

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise MCPError(INVALID_REQUEST, "Server has not been initialized.")

    @staticmethod
    def _error_frame(
        request_id: int | str | None,
        code: int,
        message: str,
        data: Any | None = None,
    ) -> str:
        resp = JsonRpcResponse(id=request_id, error=JsonRpcError(code=code, message=message, data=data))
        return resp.model_dump_json(by_alias=True, exclude_none=True)


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except TypeError:
        return str(value)


def run_stdio_server(server: MCPServer) -> None:
    """Run an :class:`MCPServer` over stdin/stdout newline-delimited JSON."""
    log.info("mcp_server_starting", name=server.name, version=server.version)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        response = server.handle(line)
        if response is not None:
            sys.stdout.write(response + "\n")
            sys.stdout.flush()

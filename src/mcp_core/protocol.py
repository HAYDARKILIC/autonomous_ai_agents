"""JSON-RPC 2.0 framing as required by the MCP specification.

The Model Context Protocol mandates JSON-RPC 2.0 as the message envelope.
This module models the request, response, and error objects as Pydantic
schemas so that every message crossing the wire is validated. The MCP layer
sits *on top* of this transport and adds method-specific payload schemas
(``initialize``, ``tools/list``, ``tools/call``, ``resources/read``, …).

References
----------
- JSON-RPC 2.0 specification:  https://www.jsonrpc.org/specification
- Model Context Protocol:      https://modelcontextprotocol.io
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


# JSON-RPC 2.0 reserved error codes (see §5.1 of the spec).
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# MCP-defined application error codes (-32000 to -32099 reserved for impl-defined).
TOOL_NOT_FOUND = -32001
TOOL_EXECUTION_ERROR = -32002
PERMISSION_DENIED = -32003


class JsonRpcRequest(BaseModel):
    """A single JSON-RPC 2.0 request object."""

    jsonrpc: Literal["2.0"] = "2.0"
    id: int | str | None = None
    method: str
    params: dict[str, Any] | list[Any] | None = None

    @property
    def is_notification(self) -> bool:
        """A notification carries no ``id`` and expects no response."""
        return self.id is None


class JsonRpcError(BaseModel):
    """A JSON-RPC 2.0 error object."""

    code: int
    message: str
    data: Any | None = None


class JsonRpcResponse(BaseModel):
    """A single JSON-RPC 2.0 response.

    Exactly one of ``result`` or ``error`` must be present.
    """

    jsonrpc: Literal["2.0"] = "2.0"
    id: int | str | None
    result: Any | None = None
    error: JsonRpcError | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> JsonRpcResponse:
        if (self.result is None) == (self.error is None):
            raise ValueError("Exactly one of `result` or `error` must be set.")
        return self


class MCPError(Exception):
    """Raised by handlers to surface a structured JSON-RPC error to the client."""

    def __init__(self, code: int, message: str, data: Any | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data

    def to_error(self) -> JsonRpcError:
        return JsonRpcError(code=self.code, message=self.message, data=self.data)


class ServerInfo(BaseModel):
    """MCP server identity returned in the ``initialize`` handshake."""

    name: str
    version: str


class ServerCapabilities(BaseModel):
    """Capability advertisement returned during ``initialize``.

    Only a subset of MCP capabilities is implemented here; each field is
    a boolean rather than the richer per-capability object the spec allows.
    """

    tools: bool = True
    resources: bool = True
    prompts: bool = False
    logging: bool = True


class InitializeResult(BaseModel):
    protocol_version: str = Field(default="0.4.0", alias="protocolVersion")
    server_info: ServerInfo = Field(alias="serverInfo")
    capabilities: ServerCapabilities

    model_config = {"populate_by_name": True}

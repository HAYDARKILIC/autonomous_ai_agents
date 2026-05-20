"""Tool registry: schema-validated functions exposed over MCP.

Each tool carries a JSON-Schema description of its inputs so the LLM client
can introspect the available actions before invoking them. At call time,
arguments are validated against the schema before the Python callable runs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from src.mcp_core.protocol import INVALID_PARAMS, TOOL_NOT_FOUND, MCPError


@dataclass(slots=True)
class Tool:
    """An MCP-callable tool: name + schema + handler."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Any]
    _validator: Draft202012Validator = field(init=False)

    def __post_init__(self) -> None:
        # Pre-compile the validator once so per-call overhead is minimal.
        Draft202012Validator.check_schema(self.input_schema)
        self._validator = Draft202012Validator(self.input_schema)

    def describe(self) -> dict[str, Any]:
        """Return the descriptor sent in response to ``tools/list``."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }

    def invoke(self, arguments: dict[str, Any]) -> Any:
        """Validate the arguments and invoke the wrapped handler."""
        try:
            self._validator.validate(arguments)
        except ValidationError as exc:
            raise MCPError(
                code=INVALID_PARAMS,
                message=f"Invalid arguments for tool {self.name!r}: {exc.message}",
                data={"path": list(exc.absolute_path)},
            ) from exc
        return self.handler(**arguments)


class ToolRegistry:
    """A collection of tools indexed by name."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} is already registered.")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise MCPError(
                code=TOOL_NOT_FOUND,
                message=f"Tool {name!r} is not registered.",
            ) from exc

    def list_all(self) -> list[dict[str, Any]]:
        return [t.describe() for t in self._tools.values()]

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

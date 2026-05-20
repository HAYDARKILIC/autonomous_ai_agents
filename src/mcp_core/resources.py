"""MCP resource providers: read-only views onto local data.

An MCP *resource* is a URI-addressable artifact that the client may read
into context. Two providers are implemented:

- :class:`FilesystemResources` — paths beneath a sandbox root, enforced by
  :func:`os.path.realpath` so that symlinks cannot escape the sandbox.
- :class:`SQLiteResources` — read-only views of tables in a SQLite DB,
  exposed as ``sqlite://<table>`` URIs.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

from src.mcp_core.protocol import PERMISSION_DENIED, MCPError


class FilesystemResources:
    """Expose a read-only view of files beneath a sandbox root."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).resolve()
        if not self._root.is_dir():
            raise ValueError(f"Filesystem root {self._root} is not a directory.")

    def list_resources(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for path in self._root.rglob("*"):
            if path.is_file():
                rel = path.relative_to(self._root)
                out.append(
                    {
                        "uri": f"file://{rel.as_posix()}",
                        "name": rel.name,
                        "mimeType": _mime_for(path),
                    }
                )
        return out

    def read(self, uri: str) -> str:
        if not uri.startswith("file://"):
            raise MCPError(PERMISSION_DENIED, f"Unsupported scheme in URI: {uri!r}")
        rel = uri[len("file://"):]
        target = (self._root / rel).resolve()
        # Sandbox check: rejects ../ escapes and symlinks that point outside root.
        if os.path.commonpath([str(target), str(self._root)]) != str(self._root):
            raise MCPError(PERMISSION_DENIED, f"Path escapes sandbox: {uri!r}")
        if not target.is_file():
            raise MCPError(PERMISSION_DENIED, f"File not found: {uri!r}")
        return target.read_text(encoding="utf-8", errors="replace")


class SQLiteResources:
    """Expose tables of a SQLite database as read-only ``sqlite://<table>`` URIs."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        # `mode=ro` opens the DB read-only via the URI form.
        self._conn = sqlite3.connect(
            f"file:{self._db_path}?mode=ro", uri=True, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row

    def list_resources(self) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        return [
            {
                "uri": f"sqlite://{row['name']}",
                "name": row["name"],
                "mimeType": "application/json",
            }
            for row in cur.fetchall()
        ]

    def read(self, uri: str, limit: int = 100) -> list[dict[str, Any]]:
        if not uri.startswith("sqlite://"):
            raise MCPError(PERMISSION_DENIED, f"Unsupported scheme in URI: {uri!r}")
        table = uri[len("sqlite://"):]
        if not table.isidentifier():
            raise MCPError(PERMISSION_DENIED, f"Invalid table name: {table!r}")
        cur = self._conn.execute(f"SELECT * FROM {table} LIMIT ?", (limit,))
        return [dict(row) for row in cur.fetchall()]


_MIME_BY_EXT = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".py": "text/x-python",
    ".json": "application/json",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
}


def _mime_for(path: Path) -> str:
    return _MIME_BY_EXT.get(path.suffix.lower(), "application/octet-stream")

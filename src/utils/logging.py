"""Structured logging configured once at import time."""

from __future__ import annotations

import logging
import os
import sys

import structlog


def _configure() -> None:
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    fmt = os.environ.get("LOG_FORMAT", "console")

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=getattr(logging, level, logging.INFO),
    )

    processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    if fmt == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level, logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )


_configure()


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger bound to the calling module."""
    return structlog.get_logger(name)

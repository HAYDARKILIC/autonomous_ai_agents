"""Tests for the YAML+env-var configuration loader."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.utils.config import load_yaml_config


def test_env_var_interpolation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_VAR", "hello")
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("greeting: ${MY_VAR}\nfallback: ${NOT_SET:-default}\n")
    cfg = load_yaml_config(cfg_path)
    assert cfg["greeting"] == "hello"
    assert cfg["fallback"] == "default"


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_yaml_config(tmp_path / "nope.yaml")


def test_nested_interpolation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOST", "example.com")
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("server:\n  host: ${HOST}\n  ports: [80, 443]\n")
    cfg = load_yaml_config(cfg_path)
    assert cfg["server"]["host"] == "example.com"
    assert cfg["server"]["ports"] == [80, 443]

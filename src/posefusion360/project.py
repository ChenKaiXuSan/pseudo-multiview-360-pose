"""Project path helpers for scripts and package entry points."""

from __future__ import annotations

from pathlib import Path


def get_repo_root() -> Path:
    """Return the 360PoseFusion repository root."""
    return Path(__file__).resolve().parents[2]


def get_project_root() -> Path:
    """Return the project root used by legacy scripts and local assets."""
    return get_repo_root()


def get_legacy_script_path(name: str) -> Path:
    """Return an absolute path to a legacy root-level script."""
    return get_repo_root() / name


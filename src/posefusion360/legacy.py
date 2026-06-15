"""Compatibility helpers for root-level legacy pipeline scripts."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

from posefusion360.project import get_repo_root


def ensure_repo_on_path() -> None:
    """Make root-level legacy scripts importable without installation side effects."""
    repo_root = str(get_repo_root())
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def import_legacy_module(module_name: str) -> ModuleType:
    """Import a legacy root-level module by name."""
    ensure_repo_on_path()
    return importlib.import_module(module_name)


def run_legacy_main(module_name: str) -> int:
    """Run a legacy module's main() function and normalize the return code."""
    module = import_legacy_module(module_name)
    main = getattr(module, "main", None)
    if main is None:
        raise AttributeError(f"legacy module has no main(): {module_name}")
    result = main()
    return int(result or 0)


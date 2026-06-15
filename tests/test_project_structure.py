#!/usr/bin/env python3
"""Smoke tests for the projectized package layout."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def test_package_exposes_project_paths() -> None:
    import posefusion360
    from posefusion360.project import get_project_root, get_repo_root

    assert posefusion360.__version__
    assert get_repo_root() == REPO_ROOT
    assert get_project_root() == REPO_ROOT


def test_pipeline_wrappers_are_importable() -> None:
    from posefusion360.pipelines import direct_360_compare, multiview_fusion, yolo_tracking

    assert callable(multiview_fusion.main)
    assert callable(direct_360_compare.main)
    assert callable(yolo_tracking.main)


def test_two_stage_project_modules_are_importable() -> None:
    from posefusion360 import geometry, io, multiview, sam3d, tracking, visualization
    from posefusion360.pipelines import full_pipeline, tracking as tracking_pipeline

    assert tracking.__doc__
    assert multiview.__doc__
    assert sam3d.__doc__
    assert visualization.__doc__
    assert geometry.__doc__
    assert io.__doc__
    assert callable(tracking_pipeline.main)
    assert callable(full_pipeline.main)


def test_project_configs_exist() -> None:
    config_dir = REPO_ROOT / "configs"

    assert (config_dir / "tracking.yaml").exists()
    assert (config_dir / "multiview_fusion.yaml").exists()
    assert (config_dir / "paths.example.yaml").exists()


if __name__ == "__main__":
    test_package_exposes_project_paths()
    test_pipeline_wrappers_are_importable()
    test_two_stage_project_modules_are_importable()
    test_project_configs_exist()

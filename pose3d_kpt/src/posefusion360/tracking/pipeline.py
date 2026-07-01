"""Tracking pipeline orchestration."""

from __future__ import annotations

from posefusion360.legacy import run_legacy_main


def main() -> int:
    """Run the legacy YOLO/CoTracker tracking workflow."""
    return run_legacy_main("cotracker_person_tracking_yolo")


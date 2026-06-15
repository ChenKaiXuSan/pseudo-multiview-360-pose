"""Package entry point for YOLO/CoTracker person tracking."""

from __future__ import annotations

from posefusion360.legacy import run_legacy_main


def main() -> int:
    """Run the legacy YOLO/CoTracker tracking CLI."""
    return run_legacy_main("cotracker_person_tracking_yolo")


if __name__ == "__main__":
    raise SystemExit(main())


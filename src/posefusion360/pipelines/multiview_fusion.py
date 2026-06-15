"""Package entry point for SAM3D Body multiview fusion."""

from __future__ import annotations

from posefusion360.legacy import run_legacy_main


def main() -> int:
    """Run the legacy multiview fusion CLI."""
    return run_legacy_main("sam3d_body_multiview_fusion")


if __name__ == "__main__":
    raise SystemExit(main())


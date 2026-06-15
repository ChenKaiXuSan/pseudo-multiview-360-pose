"""Package entry point for direct 360 SAM3D Body comparison."""

from __future__ import annotations

from posefusion360.legacy import run_legacy_main


def main() -> int:
    """Run the legacy direct 360 comparison CLI."""
    return run_legacy_main("sam3d_body_360_direct_compare")


if __name__ == "__main__":
    raise SystemExit(main())


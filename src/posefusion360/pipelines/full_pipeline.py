"""Package entry point for the full tracking-to-multiview workflow."""

from __future__ import annotations

from posefusion360.pipelines import multiview_fusion, tracking


def main() -> int:
    """Run tracking first, then multiview fusion."""
    code = tracking.main()
    if code != 0:
        return code
    return multiview_fusion.main()


if __name__ == "__main__":
    raise SystemExit(main())


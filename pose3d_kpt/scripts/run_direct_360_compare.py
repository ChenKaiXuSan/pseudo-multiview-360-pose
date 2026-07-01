#!/usr/bin/env python3
"""Run the direct 360 SAM3D Body comparison pipeline from the project package."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from posefusion360.pipelines.direct_360_compare import main


if __name__ == "__main__":
    raise SystemExit(main())


#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pointcloud_reconstruction.merge_colmap import merge_manifest_colmap_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roughly merge per-view COLMAP point clouds using view manifest rotations.")
    parser.add_argument("--manifest", required=True, type=Path, help="view_manifest.json from extract_dynamic_views.py.")
    parser.add_argument("--colmap-root", required=True, type=Path, help="Directory containing <view-name>/points3D.txt files.")
    parser.add_argument("--output-ply", required=True, type=Path, help="Merged ASCII PLY path.")
    parser.add_argument("--max-error", type=float, default=None, help="Optional COLMAP point reprojection error threshold.")
    parser.add_argument("--scale", type=float, default=1.0, help="Uniform scale applied before view transform.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = merge_manifest_colmap_points(
        manifest_path=args.manifest,
        colmap_root=args.colmap_root,
        output_ply=args.output_ply,
        max_error=args.max_error,
        scale=args.scale,
    )
    print("Merged PLY saved to: " + summary["output_ply"])
    print("Total points: {}".format(summary["total_points"]))
    print("Summary JSON saved to: " + str(args.output_ply.with_suffix(".json")))


if __name__ == "__main__":
    main()

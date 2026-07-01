#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pointcloud_reconstruction.evaluate_fusion import evaluate_fusion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate bbox-guided VIPE fusion outputs without 3D GT.")
    parser.add_argument("--manifest", required=True, type=Path, help="view_manifest.json")
    parser.add_argument("--colmap-root", required=True, type=Path, help="COLMAP root containing <view>/points3D.txt")
    parser.add_argument("--vipe-results-dir", type=Path, default=None, help="Optional VIPE results root containing <view>/pose/<view>.npz")
    parser.add_argument("--output-json", required=True, type=Path, help="Evaluation JSON output path")
    parser.add_argument("--sample-points", type=int, default=5000, help="Point samples per view for nearest-neighbor overlap")
    parser.add_argument("--max-error", type=float, default=5.0, help="Optional COLMAP point error threshold; use negative to disable")
    parser.add_argument("--scale", type=float, default=1.0, help="Uniform scale applied before current view rotation")
    parser.add_argument("--seed", type=int, default=13, help="Deterministic sampling seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_error = None if args.max_error is not None and args.max_error < 0 else args.max_error
    result = evaluate_fusion(
        manifest_path=args.manifest,
        colmap_root=args.colmap_root,
        vipe_results_dir=args.vipe_results_dir,
        output_json=args.output_json,
        sample_points=args.sample_points,
        max_error=max_error,
        scale=args.scale,
        seed=args.seed,
    )
    print("Evaluation JSON saved to: " + str(args.output_json))
    print("Total points after filter: {}".format(result["total_points_after_filter"]))
    for name, metrics in result["per_view"].items():
        print("  {name}: {status}, points={points}".format(
            name=name,
            status=metrics.get("status"),
            points=metrics.get("point_count", 0),
        ))
    if result["pairwise_overlap"]:
        first = result["pairwise_overlap"][0]
        print("First pair {} chamfer_l1={}".format("-".join(first["pair"]), first["symmetric_chamfer_l1"]))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pointcloud_reconstruction.frame_ply_export import export_refined_frame_plys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export refined merged VIPE point clouds as one PLY per frame.")
    parser.add_argument("--manifest", required=True, type=Path, help="view_manifest.json")
    parser.add_argument("--colmap-root", required=True, type=Path, help="COLMAP root containing <view>/points3D.txt and images.txt")
    parser.add_argument("--alignment-json", required=True, type=Path, help="refined_alignment.json from refine_vipe_views.py")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for frame_XXXXXX.ply outputs")
    parser.add_argument("--max-error", type=float, default=5.0, help="Optional COLMAP point error threshold; use negative to disable")
    parser.add_argument("--rough-scale", type=float, default=1.0, help="Uniform scale applied before manifest view rotation")
    parser.add_argument("--max-frames", type=int, default=None, help="Only export frames with index < N")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_error = None if args.max_error is not None and args.max_error < 0 else args.max_error
    summary = export_refined_frame_plys(
        manifest_path=args.manifest,
        colmap_root=args.colmap_root,
        alignment_json=args.alignment_json,
        output_dir=args.output_dir,
        max_error=max_error,
        rough_scale=args.rough_scale,
        max_frames=args.max_frames,
    )
    print("Frame PLY dir: " + summary["output_dir"])
    print("Summary JSON: " + str(args.output_dir / "frame_plys_summary.json"))
    print("Frames written: {}".format(summary["num_frames"]))
    print("Total frame observations: {}".format(summary["total_frame_observations"]))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pointcloud_reconstruction.direct_360 import run_direct_360_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a direct equirectangular-360 VIPE baseline.")
    parser.add_argument("--video", required=True, type=Path, help="Original 360/equirectangular video path.")
    parser.add_argument("--output-root", required=True, type=Path, help="Output directory for direct-360 baseline.")
    parser.add_argument("--vipe-command", default="vipe", help="VIPE CLI command.")
    parser.add_argument("--pipeline", default="default", help="VIPE pipeline name.")
    parser.add_argument("--visualize", action="store_true", help="Pass --visualize to vipe infer.")
    parser.add_argument("--export-colmap", action="store_true", help="Export VIPE result to COLMAP text format after inference.")
    parser.add_argument("--vipe-repo", default=Path("/mnt/dataset/skiing/vipe/vipe"), type=Path, help="VIPE repository root.")
    parser.add_argument("--python-command", default="python", help="Python command used for VIPE's vipe_to_colmap.py.")
    parser.add_argument("--sequence-name", default="direct_360", help="Sequence name for COLMAP export.")
    parser.add_argument("--depth-step", type=int, default=16, help="Depth sampling step passed to vipe_to_colmap.py.")
    parser.add_argument("--use-slam-map", action="store_true", help="Pass --use_slam_map to vipe_to_colmap.py.")
    parser.add_argument("--max-frames", type=int, default=None, help="Clip the source video to the first N frames before running VIPE.")
    parser.add_argument("--overwrite-clip", action="store_true", help="Regenerate the cached max-frames clip if it already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands and write summary without running VIPE.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_direct_360_baseline(
        video_path=args.video,
        output_root=args.output_root,
        vipe_command=args.vipe_command,
        pipeline=args.pipeline,
        visualize=args.visualize,
        export_colmap=args.export_colmap,
        vipe_repo=args.vipe_repo,
        python_command=args.python_command,
        sequence_name=args.sequence_name,
        depth_step=args.depth_step,
        use_slam_map=args.use_slam_map,
        max_frames=args.max_frames,
        overwrite_clip=args.overwrite_clip,
        dry_run=args.dry_run,
    )
    print("Summary JSON saved to: " + str(args.output_root / "direct_360_summary.json"))
    print("VIPE results dir: " + summary["vipe_results_dir"])
    if args.export_colmap:
        print("COLMAP root: " + summary["colmap_root"])


if __name__ == "__main__":
    main()

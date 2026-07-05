#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pointcloud_reconstruction.export_colmap import export_colmap_views, export_vipe_depth_colmap_with_observations, flatten_nested_colmap_outputs, load_view_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export per-view VIPE outputs to flat COLMAP directories.")
    parser.add_argument("--manifest", required=True, type=Path, help="view_manifest.json from extract_dynamic_views.py")
    parser.add_argument("--vipe-results-dir", required=True, type=Path, help="Directory containing per-view VIPE result folders")
    parser.add_argument("--colmap-root", required=True, type=Path, help="Output root; files land in <colmap-root>/<view>")
    parser.add_argument("--vipe-repo", default=Path("/mnt/dataset/skiing/vipe/vipe"), type=Path, help="VIPE repository root")
    parser.add_argument("--python-command", default="python", help="Python command used to run VIPE's vipe_to_colmap.py")
    parser.add_argument("--depth-step", type=int, default=16, help="Depth sampling step passed to vipe_to_colmap.py")
    parser.add_argument("--spatial-subsample", type=int, default=4, help="Pixel stride for --with-observations depth point export.")
    parser.add_argument("--with-observations", action="store_true", help="Export COLMAP images.txt with real x y POINT3D_ID observations from VIPE depth artifacts.")
    parser.add_argument("--use-slam-map", action="store_true", help="Pass --use_slam_map to vipe_to_colmap.py")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    parser.add_argument("--flatten-only", action="store_true", help="Only flatten existing <view>/<view> outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.flatten_only:
        actions = flatten_nested_colmap_outputs(args.colmap_root, load_view_names(args.manifest))
        print(f"Flattened {len(actions)} COLMAP output item(s)")
        return
    if args.with_observations:
        for view_name in load_view_names(args.manifest):
            summary = export_vipe_depth_colmap_with_observations(
                vipe_result_dir=args.vipe_results_dir / view_name,
                sequence=view_name,
                output_dir=args.colmap_root / view_name,
                depth_step=args.depth_step,
                spatial_subsample=args.spatial_subsample,
            )
            print(f"Observation-aware COLMAP export {view_name}: {summary['images']} images, {summary['points']} points")
        return
    export_colmap_views(
        manifest_path=args.manifest,
        vipe_results_dir=args.vipe_results_dir,
        colmap_root=args.colmap_root,
        vipe_repo=args.vipe_repo,
        python_command=args.python_command,
        depth_step=args.depth_step,
        use_slam_map=args.use_slam_map,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pose_pointcloud_fusion.person_centered_filter import filter_person_centered_ply


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter a VIPE point cloud in a person-centered ROI using SAM3D world poses.")
    parser.add_argument("--input-ply", required=True, type=Path, help="Input ASCII xyzrgb PLY, e.g. refined_views_world_*.ply.")
    parser.add_argument("--pose-json", required=True, action="append", type=Path, help="Pose JSON. Repeat for multiple frames; supports fused_keypoints3d.json or frame_tracks_world.json.")
    parser.add_argument("--output-ply", required=True, type=Path, help="Filtered output ASCII xyzrgb PLY.")
    parser.add_argument("--track-id", type=int, default=None, help="Track id when --pose-json points to frame_tracks_world.json.")
    parser.add_argument("--min-conf", type=float, default=0.3, help="Minimum 3D joint confidence.")
    parser.add_argument("--mode", choices=["scene", "human"], default="scene", help="scene removes near-body dynamic points; human keeps near-body points.")
    parser.add_argument("--trajectory-radius", type=float, default=8.0, help="Horizontal radius around the person trajectory to keep.")
    parser.add_argument("--height-below", type=float, default=2.0, help="Keep points this far below the person-center y range.")
    parser.add_argument("--height-above", type=float, default=3.0, help="Keep points this far above the person-center y range.")
    parser.add_argument("--body-radius", type=float, default=0.35, help="Skeleton capsule radius for scene exclusion or human inclusion.")
    parser.add_argument("--outlier-filter", choices=["none", "statistical", "voxel"], default="voxel", help="Final local outlier filter. voxel works without scipy and scales to large PLYs.")
    parser.add_argument("--outlier-k", type=int, default=20, help="Neighbor count for statistical outlier removal.")
    parser.add_argument("--outlier-std-ratio", type=float, default=2.0, help="SOR threshold: mean + ratio * std of kNN distances.")
    parser.add_argument("--voxel-size", type=float, default=0.1, help="Voxel size for --outlier-filter voxel.")
    parser.add_argument("--voxel-min-neighbors", type=int, default=4, help="Minimum points in the 3x3x3 voxel neighborhood to keep a point.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = filter_person_centered_ply(
        input_ply=args.input_ply,
        pose_jsons=args.pose_json,
        output_ply=args.output_ply,
        track_id=args.track_id,
        min_conf=args.min_conf,
        mode=args.mode,
        trajectory_radius=args.trajectory_radius,
        height_below=args.height_below,
        height_above=args.height_above,
        body_radius=args.body_radius,
        outlier_filter=args.outlier_filter,
        outlier_k=args.outlier_k,
        outlier_std_ratio=args.outlier_std_ratio,
        voxel_size=args.voxel_size,
        voxel_min_neighbors=args.voxel_min_neighbors,
    )
    print("Filtered PLY saved to: " + summary["output_ply"])
    print("Summary JSON saved to: " + str(args.output_ply.with_suffix(".json")))
    print("Input points: {}".format(summary["input_points"]))
    print("Output points: {}".format(summary["output_points"]))
    print("Removed by ROI: {}".format(summary["removed_by_roi"]))
    print("Removed by body: {}".format(summary["removed_by_body"]))
    print("Removed by outlier: {}".format(summary["removed_by_outlier"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

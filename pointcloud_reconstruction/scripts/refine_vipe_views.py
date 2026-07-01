#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pointcloud_reconstruction.refine_alignment import refine_manifest_colmap_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine rough VIPE view fusion with ICP-style Sim3 alignment.")
    parser.add_argument("--manifest", required=True, type=Path, help="view_manifest.json from extract_dynamic_views.py.")
    parser.add_argument("--colmap-root", required=True, type=Path, help="Directory containing <view-name>/points3D.txt files.")
    parser.add_argument("--output-ply", required=True, type=Path, help="Refined merged ASCII PLY path.")
    parser.add_argument("--alignment-json", required=True, type=Path, help="JSON summary of estimated per-view transforms.")
    parser.add_argument("--reference-view", default="selfie", help="View used as the alignment anchor.")
    parser.add_argument("--max-error", type=float, default=5.0, help="Optional COLMAP point error threshold; use negative to disable.")
    parser.add_argument("--rough-scale", type=float, default=1.0, help="Uniform scale applied before manifest view rotation.")
    parser.add_argument("--sample-points", type=int, default=8000, help="Per-view point samples used for alignment.")
    parser.add_argument("--max-iterations", type=int, default=12, help="ICP refinement iterations per non-reference view.")
    parser.add_argument("--trim-fraction", type=float, default=0.7, help="Closest-match fraction used for robust alignment.")
    parser.add_argument("--distance-threshold", type=float, default=-1.0, help="Optional inlier distance threshold; negative disables.")
    parser.add_argument("--max-scale-ratio", type=float, default=4.0, help="Warn if estimated scale leaves [1/ratio, ratio].")
    parser.add_argument("--max-median-error", type=float, default=10.0, help="Warn if final median NN error exceeds this value.")
    parser.add_argument("--alignment-mode", choices=["direct", "adjacent"], default="direct", help="Use direct reference-view alignment or adjacent yaw graph propagation.")
    parser.add_argument("--scale-min", type=float, default=-1.0, help="Clamp estimated per-edge scale below this value; use negative to disable lower clamp.")
    parser.add_argument("--scale-max", type=float, default=-1.0, help="Clamp estimated per-edge scale above this value; use negative to disable upper clamp.")
    parser.add_argument("--frame-output-dir", type=Path, default=None, help="Directory for default frame-level PLY outputs; defaults to <output-ply-dir>/refined_frame_plys.")
    parser.add_argument("--frame-max-frames", type=int, default=None, help="Only export frame PLY files with frame index < N.")
    parser.add_argument("--no-frame-plys", action="store_true", help="Skip default frame-level PLY export.")
    parser.add_argument("--seed", type=int, default=13, help="Deterministic sampling seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_error = None if args.max_error is not None and args.max_error < 0 else args.max_error
    distance_threshold = None if args.distance_threshold is not None and args.distance_threshold < 0 else args.distance_threshold
    scale_min = None if args.scale_min is not None and args.scale_min < 0 else args.scale_min
    scale_max = None if args.scale_max is not None and args.scale_max < 0 else args.scale_max
    summary = refine_manifest_colmap_points(
        manifest_path=args.manifest,
        colmap_root=args.colmap_root,
        output_ply=args.output_ply,
        alignment_json=args.alignment_json,
        reference_view=args.reference_view,
        max_error=max_error,
        rough_scale=args.rough_scale,
        sample_points=args.sample_points,
        max_iterations=args.max_iterations,
        trim_fraction=args.trim_fraction,
        distance_threshold=distance_threshold,
        max_scale_ratio=args.max_scale_ratio,
        max_median_error=args.max_median_error,
        alignment_mode=args.alignment_mode,
        scale_min=scale_min,
        scale_max=scale_max,
        seed=args.seed,
        export_frame_plys=not args.no_frame_plys,
        frame_output_dir=args.frame_output_dir,
        frame_max_frames=args.frame_max_frames,
    )
    print("Refined PLY saved to: " + summary["output_ply"])
    print("Alignment JSON saved to: " + str(args.alignment_json))
    if summary.get("frame_plys"):
        print("Frame PLY dir: " + summary["frame_plys"]["output_dir"])
        print("Frame PLY summary: " + summary["frame_plys"]["summary_json"])
        print("Frame PLY files: {}".format(summary["frame_plys"]["num_frames"]))
    print("Total points: {}".format(summary["total_points"]))
    for view_name, transform in summary["transforms"].items():
        warnings = ",".join(transform.get("warnings", [])) or "none"
        print("{view}: {status}, scale={scale}, median_error={median}, warnings={warnings}".format(
            view=view_name,
            status=transform.get("status"),
            scale=transform.get("scale"),
            median=transform.get("final_median_error"),
            warnings=warnings,
        ))


if __name__ == "__main__":
    main()

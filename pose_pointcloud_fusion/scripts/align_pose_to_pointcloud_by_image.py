#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pose_pointcloud_fusion.image_anchor_alignment import write_image_anchor_alignment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Anchor a SAM3D 3D skeleton to VIPE/COLMAP point-cloud coordinates through same-image 2D observations."
        )
    )
    parser.add_argument("--sam3d-json", required=True, type=Path, help="One SAM3D view payload containing keypoints2d and keypoints3d_camera.")
    parser.add_argument("--images-txt", required=True, type=Path, help="COLMAP images.txt for the same VIPE perspective view.")
    parser.add_argument("--points3d-txt", required=True, type=Path, help="COLMAP points3D.txt for the same VIPE perspective view.")
    parser.add_argument("--output-json", required=True, type=Path, help="Output JSON with aligned_keypoints3d_world and match diagnostics.")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional view_manifest.json used to transform COLMAP local points to rough 360/world coordinates.")
    parser.add_argument("--view-name", default=None, help="View name inside --manifest and --alignment-json, for example view_00 or selfie.")
    parser.add_argument("--alignment-json", type=Path, default=None, help="Optional refined_alignment.json used to transform rough world points to refined point-cloud coordinates.")
    parser.add_argument("--rough-scale", type=float, default=1.0, help="Scale applied before manifest camera_to_world, matching pointcloud_reconstruction rough_scale.")
    parser.add_argument("--image-id", type=int, default=None, help="COLMAP IMAGE_ID to use. Prefer this when known.")
    parser.add_argument("--frame-index", type=int, default=None, help="Frame index parsed from COLMAP image names when --image-id is not provided.")
    parser.add_argument("--radius-px", type=float, default=8.0, help="Maximum 2D distance between SAM3D joint and COLMAP observation.")
    parser.add_argument("--min-conf", type=float, default=0.3, help="Minimum confidence for SAM3D 2D and 3D keypoints.")
    parser.add_argument("--max-point-error", type=float, default=None, help="Optional maximum COLMAP point reprojection error.")
    parser.add_argument("--no-scale", action="store_true", help="Estimate rigid rotation/translation only, without scale.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = write_image_anchor_alignment(
        sam3d_json=args.sam3d_json,
        images_txt=args.images_txt,
        points3d_txt=args.points3d_txt,
        output_json=args.output_json,
        image_id=args.image_id,
        frame_index=args.frame_index,
        manifest_path=args.manifest,
        view_name=args.view_name,
        alignment_json=args.alignment_json,
        rough_scale=args.rough_scale,
        radius_px=args.radius_px,
        min_conf=args.min_conf,
        max_point_error=args.max_point_error,
        allow_scaling=not args.no_scale,
    )
    print(json.dumps({
        "output_json": str(args.output_json),
        "num_matches": summary["num_matches"],
        "scale": summary["transform"]["scale"],
        "rmse": summary["transform"]["rmse"],
        "image_id": summary["image_id"],
        "image_name": summary["image_name"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

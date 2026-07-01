#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pointcloud_reconstruction.extract_views import extract_bbox_guided_views, parse_yaw_offsets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract selfie-bbox-guided perspective views from a 360 video.")
    parser.add_argument("--video", required=True, type=Path, help="Input equirectangular 360 video.")
    parser.add_argument("--bbox-json", required=True, type=Path, help="Selfie bbox JSON from 360PoseFusion.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for view videos and manifest.")
    parser.add_argument("--target-id", type=int, default=1, help="Selfie track id in the bbox JSON.")
    parser.add_argument("--view-size", type=int, default=1024, help="Square output size per perspective view.")
    parser.add_argument("--fov-deg", type=float, default=100.0, help="Perspective view field of view.")
    parser.add_argument("--yaw-offsets", default="0,60,120,180,-120,-60", help="Comma-separated yaw offsets around selfie anchor.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on extracted frames per view.")
    parser.add_argument("--every-n", type=int, default=1, help="Extract one frame every N source frames.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = extract_bbox_guided_views(
        video_path=args.video,
        bbox_json_path=args.bbox_json,
        output_dir=args.output_dir,
        target_id=args.target_id,
        view_size=args.view_size,
        fov_deg=args.fov_deg,
        yaw_offsets=parse_yaw_offsets(args.yaw_offsets),
        max_frames=args.max_frames,
        every_n=args.every_n,
    )
    print("View manifest saved to: " + str(args.output_dir / "view_manifest.json"))
    print("Anchor yaw: {:.3f} deg".format(manifest["anchor_yaw_deg"]))
    print("Frames per view: {}".format(manifest["frames_written_per_view"]))


if __name__ == "__main__":
    main()

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

from pose_pointcloud_fusion.pose_pointcloud_overlay import (
    save_matplotlib_overlay_screenshot,
    save_open3d_overlay_screenshot,
    save_pillow_overlay_screenshot,
    write_pose_pointcloud_overlay_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay fused SAM3D world keypoints onto VIPE point-cloud PLYs.")
    parser.add_argument(
        "--scene-ply",
        type=Path,
        default=Path("/mnt/dataset/skiing/360PoseFusion/pointcloud_reconstruction/outputs/kimura2_360/merged_views_world.ply"),
        help="Background ASCII xyzrgb PLY. Used for every pose frame unless --scene-frame-dir has a matching frame PLY.",
    )
    parser.add_argument(
        "--scene-frame-dir",
        type=Path,
        default=None,
        help="Optional directory with frame_XXXXXX.ply scene files. SAM3D frame N maps to scene frame N-1.",
    )
    parser.add_argument(
        "--pose-root",
        type=Path,
        default=Path("/mnt/dataset/skiing/sam3d_body_multiview/kimura2_360"),
        help="SAM3D multiview video output root containing frame_XXXXXX/track_XXXX/fused/fused_keypoints3d.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/dataset/skiing/360PoseFusion/pose_pointcloud_fusion/outputs/pose_pointcloud_overlay"),
        help="Directory for overlay PLYs, screenshots, and summary JSON.",
    )
    parser.add_argument("--start-frame", type=int, default=1, help="First SAM3D frame number to export.")
    parser.add_argument("--max-frames", type=int, default=1, help="Maximum number of pose frames to export.")
    parser.add_argument("--track-id", type=int, default=1, help="Track id to overlay.")
    parser.add_argument("--min-conf", type=float, default=0.3, help="Minimum 3D keypoint confidence.")
    parser.add_argument("--joint-radius", type=float, default=0.08, help="Marker radius, in world units.")
    parser.add_argument("--bone-step", type=float, default=0.04, help="Spacing of sampled bone points, in world units.")
    parser.add_argument("--screenshot", action="store_true", help="Save a PNG preview for the first exported frame.")
    parser.add_argument(
        "--prefer-open3d",
        action="store_true",
        help="Try Open3D offscreen rendering before falling back to Matplotlib.",
    )
    parser.add_argument("--max-scene-preview-points", type=int, default=20000, help="Scene points used in Matplotlib preview.")
    return parser.parse_args()


def resolve_pose_json(pose_root: Path, frame_number: int, track_id: int) -> Path:
    return pose_root / f"frame_{frame_number:06d}" / f"track_{track_id:04d}" / "fused" / "fused_keypoints3d.json"


def resolve_scene_ply(scene_ply: Path, scene_frame_dir: Path | None, frame_number: int) -> Path:
    if scene_frame_dir is not None:
        candidate = scene_frame_dir / f"frame_{frame_number - 1:06d}.ply"
        if candidate.exists():
            return candidate
    return scene_ply


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_summaries = []
    screenshot_summary = None

    for frame_number in range(args.start_frame, args.start_frame + args.max_frames):
        pose_json = resolve_pose_json(args.pose_root, frame_number, args.track_id)
        if not pose_json.exists():
            frame_summaries.append({
                "frame_number": frame_number,
                "track_id": args.track_id,
                "status": "missing_pose_json",
                "pose_json": str(pose_json),
            })
            continue
        scene_ply = resolve_scene_ply(args.scene_ply, args.scene_frame_dir, frame_number)
        output_ply = args.output_dir / f"frame_{frame_number:06d}_track_{args.track_id:04d}_overlay.ply"
        summary = write_pose_pointcloud_overlay_frame(
            scene_ply=scene_ply,
            pose_json=pose_json,
            output_ply=output_ply,
            min_conf=args.min_conf,
            joint_radius=args.joint_radius,
            bone_step=args.bone_step,
        )
        summary["status"] = "written"
        frame_summaries.append(summary)

        if args.screenshot and screenshot_summary is None:
            output_png = args.output_dir / f"frame_{frame_number:06d}_track_{args.track_id:04d}_overlay.png"
            renderer = "matplotlib"
            try:
                if args.prefer_open3d:
                    save_open3d_overlay_screenshot(overlay_ply=output_ply, output_png=output_png)
                    renderer = "open3d"
                else:
                    raise RuntimeError("open3d not requested")
            except Exception as exc:
                first_error = exc
                try:
                    save_matplotlib_overlay_screenshot(
                        scene_ply=scene_ply,
                        pose_json=pose_json,
                        output_png=output_png,
                        min_conf=args.min_conf,
                        max_scene_points=args.max_scene_preview_points,
                    )
                    renderer = f"matplotlib_fallback_after_{type(first_error).__name__}"
                except Exception as second_exc:
                    save_pillow_overlay_screenshot(
                        scene_ply=scene_ply,
                        pose_json=pose_json,
                        output_png=output_png,
                        min_conf=args.min_conf,
                        max_scene_points=args.max_scene_preview_points,
                    )
                    renderer = f"pillow_fallback_after_{type(first_error).__name__}_and_{type(second_exc).__name__}"
            screenshot_summary = {"path": str(output_png), "renderer": renderer}

    run_summary = {
        "scene_ply": str(args.scene_ply),
        "scene_frame_dir": str(args.scene_frame_dir) if args.scene_frame_dir is not None else None,
        "pose_root": str(args.pose_root),
        "output_dir": str(args.output_dir),
        "start_frame": args.start_frame,
        "max_frames": args.max_frames,
        "track_id": args.track_id,
        "min_conf": args.min_conf,
        "frames": frame_summaries,
        "screenshot": screenshot_summary,
    }
    summary_path = args.output_dir / "pose_pointcloud_overlay_summary.json"
    summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    written = [frame for frame in frame_summaries if frame.get("status") == "written"]
    print("Overlay frames written: {}".format(len(written)))
    for frame in written:
        print("  PLY: " + frame["output_ply"])
    if screenshot_summary is not None:
        print("  PNG: " + screenshot_summary["path"])
        print("  renderer: " + screenshot_summary["renderer"])
    print("Summary JSON: " + str(summary_path))
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Run SAM3D Body directly on original 360 equirectangular frames and let SAM3D
Body perform person detection internally. This is a comparison baseline for
sam3d_body_multiview_fusion.py.

Important: SAM3D Body is trained/implemented for perspective camera images.
This script intentionally feeds the equirectangular frame as-is, so the output
is useful as a baseline, not as geometrically correct 360 inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from sam3d_body_multiview_fusion import (
    CONFIG as MULTIVIEW_CONFIG,
    Sam3DBodyDirectRunner,
    extract_keypoints2d,
    extract_sam3d_camera_keypoints,
    finite_keypoint_mask,
    keypoints3d_to_plot_coords,
    load_mhr70_visual_style,
    load_sam3d_payload,
    normalize_command_path,
    numpy_to_jsonable,
    open_video,
    read_video_frame,
    save_keypoints2d_overlay,
    save_keypoints3d_plot,
    set_axes_equal,
    track_color_rgb01,
)


CONFIG = {
    **MULTIVIEW_CONFIG,
    "output_dir": "/mnt/dataset/skiing/raw_new/sam3d_body_360_direct_compare",
    # Direct-360 comparison lets SAM3D Body detect people in the full frame.
    "sam3d_detector_name": "vitdet",
    "sam3d_detector_path": "",
    "sam3d_detector_bbox_thr": 0.5,
    "sam3d_detector_nms_thr": 0.3,
    # Do not fabricate a 360-degree pinhole camera by default. If you want a
    # controlled fake-pinhole baseline, pass --known-intrinsics with hfov < 180.
    "sam3d_use_known_intrinsics": False,
    "visualize_keypoints": True,
    "visualize_joint_indices": True,
}


def output_bbox_xyxy(output: dict[str, Any]) -> list[int] | None:
    bbox = output.get("bbox") if isinstance(output, dict) else None
    if bbox is None:
        return None
    arr = np.asarray(bbox, dtype=np.float64).reshape(-1)
    if arr.size < 4:
        return None
    return [int(round(float(v))) for v in arr[:4]]


def make_single_person_payload(image_path: Path, output: dict[str, Any], detection_index: int) -> dict[str, Any]:
    bbox_xyxy = output_bbox_xyxy(output)
    payload = {
        "image_path": normalize_command_path(image_path),
        "bbox_format": "xyxy",
        "bbox_xyxy": bbox_xyxy,
        "detection_mode": "sam3d_internal_detector",
        "detection_index": int(detection_index),
        "outputs": [numpy_to_jsonable(output)],
    }
    if "pred_keypoints_3d" in output:
        payload["keypoints3d"] = np.asarray(output["pred_keypoints_3d"]).tolist()
        if "pred_cam_t" in output:
            kpts_camera = (
                np.asarray(output["pred_keypoints_3d"], dtype=np.float64)
                + np.asarray(output["pred_cam_t"], dtype=np.float64).reshape(1, 3)
            )
            payload["keypoints3d_camera"] = kpts_camera.tolist()
    if "pred_keypoints_2d" in output:
        payload["keypoints2d"] = np.asarray(output["pred_keypoints_2d"]).tolist()
    if "pred_joint_coords" in output:
        payload["joint_coords"] = np.asarray(output["pred_joint_coords"]).tolist()
    return payload


def rgb01_to_bgr255(color: np.ndarray | list[float]) -> tuple[int, int, int]:
    arr = np.asarray(color, dtype=np.float64)
    if arr.max(initial=0.0) <= 1.0:
        arr = arr * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return int(arr[2]), int(arr[1]), int(arr[0])


def draw_frame_direct_2d_overlay(
    frame_bgr,
    track_results: list[dict[str, Any]],
    output_path: Path,
    edges: list[tuple[int, int]],
    min_conf: float,
    show_indices: bool,
) -> str | None:
    if frame_bgr is None or not track_results:
        return None
    image = frame_bgr.copy()
    h, w = image.shape[:2]
    for result in track_results:
        track_id = int(result.get("track_id", result.get("detection_index", 0)))
        color_rgb = track_color_rgb01(track_id)
        color = rgb01_to_bgr255(color_rgb)
        bbox = result.get("source_bbox_xyxy")
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                image,
                f"person_{track_id:04d}",
                (max(0, x1), max(14, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        kpts2d = result.get("keypoints2d")
        if not kpts2d:
            continue
        kpts = np.asarray(kpts2d, dtype=np.float64)
        if kpts.ndim != 2 or kpts.shape[1] < 2:
            continue
        mask = np.isfinite(kpts[:, :2]).all(axis=1)
        if kpts.shape[1] > 2:
            mask &= kpts[:, 2] >= min_conf
        for a, b in edges:
            if a >= len(kpts) or b >= len(kpts) or not mask[a] or not mask[b]:
                continue
            p1 = tuple(np.round(kpts[a, :2]).astype(int))
            p2 = tuple(np.round(kpts[b, :2]).astype(int))
            if not (0 <= p1[0] < w and 0 <= p1[1] < h and 0 <= p2[0] < w and 0 <= p2[1] < h):
                continue
            cv2.line(image, p1, p2, color, 2, lineType=cv2.LINE_AA)
        for joint_idx, point in enumerate(kpts[:, :2]):
            if not mask[joint_idx]:
                continue
            x, y = np.round(point).astype(int)
            if not (0 <= x < w and 0 <= y < h):
                continue
            cv2.circle(image, (x, y), 4, color, -1, lineType=cv2.LINE_AA)
            if show_indices:
                cv2.putText(image, str(joint_idx), (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    return normalize_command_path(output_path)


def save_frame_direct_3d_camera_plot(
    track_results: list[dict[str, Any]],
    output_path: Path,
    edges: list[tuple[int, int]],
    show_indices: bool,
    min_conf: float,
    title: str,
) -> str | None:
    tracks = []
    for result in track_results:
        rows = result.get("keypoints3d_camera")
        if not rows:
            continue
        kpts = np.asarray(rows, dtype=np.float64)
        if kpts.ndim != 2 or kpts.shape[1] < 3:
            continue
        if kpts.shape[1] == 3:
            kpts = np.concatenate([kpts[:, :3], np.ones((len(kpts), 1), dtype=np.float64)], axis=1)
        mask = finite_keypoint_mask(kpts, min_conf)
        if not np.any(mask):
            continue
        tracks.append((int(result.get("track_id", result.get("detection_index", len(tracks) + 1))), kpts[:, :4], mask))
    if not tracks:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    all_pts = []
    axis_labels = ("camera X right", "camera Z depth", "camera -Y up")
    view_angles = (14, -70)
    for track_id, kpts, mask in tracks:
        plot_all, axis_labels, view_angles = keypoints3d_to_plot_coords(kpts, "camera")
        pts = plot_all[mask]
        ids = np.flatnonzero(mask)
        color = track_color_rgb01(track_id)
        label = f"person {track_id:04d}"
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=[color], s=24, depthshade=True, label=label)
        for a, b in edges:
            if a < len(kpts) and b < len(kpts) and mask[a] and mask[b]:
                seg = plot_all[[a, b], :3]
                ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=color, linewidth=1.4, alpha=0.9)
        if show_indices:
            for joint_id, point in zip(ids, pts):
                ax.text(point[0], point[1], point[2], str(int(joint_id)), fontsize=5, color=color)
        all_pts.append(pts)
    ax.set_title(title)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])
    ax.view_init(elev=view_angles[0], azim=view_angles[1])
    if all_pts:
        set_axes_equal(ax, np.concatenate(all_pts, axis=0))
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return normalize_command_path(output_path)


def save_frame_direct_visualizations(
    frame_bgr,
    frame_number: int,
    frame_dir: Path,
    track_results: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    style = load_mhr70_visual_style(config.get("sam3d_repo", ""))
    edges = style["edges"]
    min_conf = float(config.get("min_kpt_conf", 0.0))
    show_indices = bool(config.get("visualize_joint_indices", True))
    vis = {
        "frame_number": int(frame_number),
        "num_visualized_tracks": len(track_results),
        "detections_kpts2d_path": draw_frame_direct_2d_overlay(
            frame_bgr,
            track_results,
            frame_dir / "frame_360_detections_kpts2d.jpg",
            edges,
            min_conf,
            show_indices,
        ),
        "kpts3d_camera_path": save_frame_direct_3d_camera_plot(
            track_results,
            frame_dir / "direct_360_kpts3d_camera.png",
            edges,
            show_indices,
            min_conf,
            f"frame {frame_number:06d} direct 360 camera 3D",
        ),
    }
    metadata_path = frame_dir / "direct_360_frame_visualization.json"
    metadata_path.write_text(json.dumps(vis, indent=2), encoding="utf-8")
    vis["metadata_path"] = normalize_command_path(metadata_path)
    return vis


def write_direct_track_result(
    frame_bgr,
    frame_number: int,
    detection_index: int,
    output: dict[str, Any],
    frame_image_path: Path,
    output_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    track_id = int(detection_index)
    track_dir = output_dir / f"frame_{frame_number:06d}" / f"track_{track_id:04d}"
    track_dir.mkdir(parents=True, exist_ok=True)
    image_path = track_dir / "frame_360.jpg"
    sam_output_path = track_dir / "sam3d_360.json"
    cv2.imwrite(str(image_path), frame_bgr)

    person_payload = make_single_person_payload(image_path, output, detection_index)
    sam_output_path.write_text(json.dumps(person_payload, indent=2), encoding="utf-8")
    bbox_xyxy = person_payload.get("bbox_xyxy")
    kpts2d = extract_keypoints2d(person_payload)
    kpts_cam = extract_sam3d_camera_keypoints(
        person_payload,
        use_camera_translation=config.get("sam3d_use_camera_translation", True),
    )

    visualization: dict[str, Any] = {}
    if config.get("visualize_keypoints", True):
        style = load_mhr70_visual_style(config.get("sam3d_repo", ""))
        overlay_path = save_keypoints2d_overlay(
            image_path,
            kpts2d,
            track_dir / "frame_360_kpts2d.jpg",
            bbox_xyxy,
            style["edges"],
            style["edge_colors"],
            style["point_colors"],
            bool(config.get("visualize_joint_indices", True)),
            float(config.get("min_kpt_conf", 0.0)),
        )
        visualization["kpts2d_overlay_path"] = overlay_path
        visualization["kpts3d_camera_path"] = save_keypoints3d_plot(
            kpts_cam,
            track_dir / "kpts3d_360_camera.png",
            f"frame {frame_number:06d} track {track_id:04d} direct 360 camera 3D",
            style["edges"],
            style["edge_colors"],
            style["point_colors"],
            bool(config.get("visualize_joint_indices", True)),
            float(config.get("min_kpt_conf", 0.0)),
            plot_space="camera",
        )
    result = {
        "frame_number": int(frame_number),
        "track_id": int(track_id),
        "detection_index": int(detection_index),
        "source_bbox_xyxy": bbox_xyxy,
        "input_projection": "equirectangular_360_as_image",
        "coordinate_system": "direct SAM3D camera coordinates",
        "note": "SAM3D Body expects perspective images; this direct 360 result is a comparison baseline.",
        "image_path": normalize_command_path(image_path),
        "frame_image_path": normalize_command_path(frame_image_path),
        "sam3d_output_path": normalize_command_path(sam_output_path),
        "keypoints2d": numpy_to_jsonable(kpts2d) if kpts2d is not None else [],
        "keypoints3d_camera": numpy_to_jsonable(kpts_cam) if kpts_cam is not None else [],
        "num_model_outputs": 1,
        "visualization": visualization,
    }
    result_path = track_dir / "direct_360_result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["result_path"] = normalize_command_path(result_path)
    return result


def process_frame_direct_360(
    frame_bgr,
    frame_number: int,
    output_dir: Path,
    config: dict[str, Any],
    sam3d_runner: Sam3DBodyDirectRunner | None,
) -> list[dict[str, Any]]:
    frame_dir = output_dir / f"frame_{frame_number:06d}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    image_path = frame_dir / "frame_360.jpg"
    sam_output_path = frame_dir / "sam3d_360_all.json"
    cv2.imwrite(str(image_path), frame_bgr)

    print(f"frame {frame_number:06d}: direct 360 SAM3D internal detection")
    if sam3d_runner is None:
        print("  SAM3D Body disabled; saved frame only.")
        return []

    sam3d_runner.run(image_path, None, sam_output_path)
    payload = load_sam3d_payload(sam_output_path) or {}
    outputs = payload.get("outputs", []) if isinstance(payload, dict) else []
    print(f"  detected people: {len(outputs)}")

    results = []
    for detection_index, output in enumerate(outputs, start=1):
        if not isinstance(output, dict):
            continue
        bbox_xyxy = output_bbox_xyxy(output)
        print(f"  detection={detection_index:04d} bbox={bbox_xyxy}")
        results.append(
            write_direct_track_result(
                frame_bgr,
                frame_number,
                detection_index,
                output,
                image_path,
                output_dir,
                config,
            )
        )
    if config.get("visualize_keypoints", True):
        frame_vis = save_frame_direct_visualizations(frame_bgr, frame_number, frame_dir, results, config)
        if frame_vis.get("detections_kpts2d_path"):
            print("  frame direct 2D: " + frame_vis["detections_kpts2d_path"])
        if frame_vis.get("kpts3d_camera_path"):
            print("  frame direct 3D: " + frame_vis["kpts3d_camera_path"])
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct 360 frame -> SAM3D Body internal-detection comparison baseline")
    parser.add_argument("--video", default=CONFIG["video_path"], help="360 equirectangular video path")
    parser.add_argument("--output-dir", default=CONFIG["output_dir"], help="output directory")
    parser.add_argument("--frame-number", type=int, default=None, help="only process this 1-based frame number")
    parser.add_argument("--max-frames", type=int, default=1, help="maximum video frames to process from frame 1; set 0 for all")
    parser.add_argument("--no-run-sam3d", action="store_true", help="save frames only; do not run SAM3D Body")
    parser.add_argument("--sam3d-repo", default=CONFIG["sam3d_repo"], help="path to facebookresearch/sam-3d-body repo")
    parser.add_argument("--sam3d-checkpoint", default=CONFIG["sam3d_checkpoint_path"], help="local SAM3D Body model.ckpt path")
    parser.add_argument("--sam3d-mhr", default=CONFIG["sam3d_mhr_path"], help="local MHR model asset path")
    parser.add_argument("--sam3d-hf-repo", default=CONFIG["sam3d_hf_repo"], help="HF repo used when no local checkpoint is supplied")
    parser.add_argument("--sam3d-device", default=CONFIG["sam3d_device"], help="auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--sam3d-inference-type", default=CONFIG["sam3d_inference_type"], choices=["full", "body", "hand"])
    parser.add_argument("--sam3d-detector", default=CONFIG["sam3d_detector_name"], help="SAM3D Body detector name, e.g. vitdet or sam3; empty disables detector")
    parser.add_argument("--sam3d-detector-path", default=CONFIG["sam3d_detector_path"], help="optional local detector checkpoint directory")
    parser.add_argument("--sam3d-detector-bbox-thr", type=float, default=CONFIG["sam3d_detector_bbox_thr"], help="detector confidence threshold")
    parser.add_argument("--sam3d-detector-nms-thr", type=float, default=CONFIG["sam3d_detector_nms_thr"], help="detector NMS threshold")
    parser.add_argument("--known-intrinsics", action="store_true", help="pass fake pinhole intrinsics for the whole 360 frame")
    parser.add_argument("--hfov", type=float, default=120.0, help="fake horizontal FOV when --known-intrinsics is used")
    parser.add_argument("--vfov", type=float, default=90.0, help="fake vertical FOV when --known-intrinsics is used")
    parser.add_argument("--no-sam-cam-translation", action="store_true", help="use root-relative pred_keypoints_3d instead of pred_keypoints_3d + pred_cam_t")
    parser.add_argument("--min-kpt-conf", type=float, default=CONFIG["min_kpt_conf"])
    parser.add_argument("--no-kpt-vis", action="store_true", help="skip 2D/3D keypoint visualization PNG outputs")
    parser.add_argument("--no-joint-indices", action="store_true", help="hide joint index labels in plots")
    return parser.parse_args(argv)


def selected_frame_numbers(cap, frame_number: int | None, max_frames: int) -> list[int]:
    if frame_number is not None:
        return [int(frame_number)]
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if max_frames == 0:
        return list(range(1, frame_count + 1))
    return list(range(1, max(0, int(max_frames)) + 1))


def main() -> int:
    args = parse_args()
    config = dict(CONFIG)
    config.update({
        "sam3d_repo": args.sam3d_repo,
        "sam3d_checkpoint_path": args.sam3d_checkpoint,
        "sam3d_mhr_path": args.sam3d_mhr,
        "sam3d_hf_repo": args.sam3d_hf_repo,
        "sam3d_device": args.sam3d_device,
        "sam3d_inference_type": args.sam3d_inference_type,
        "sam3d_detector_name": args.sam3d_detector,
        "sam3d_detector_path": args.sam3d_detector_path,
        "sam3d_detector_bbox_thr": float(args.sam3d_detector_bbox_thr),
        "sam3d_detector_nms_thr": float(args.sam3d_detector_nms_thr),
        "sam3d_use_known_intrinsics": bool(args.known_intrinsics),
        "hfov_deg": float(args.hfov),
        "vfov_deg": float(args.vfov),
        "sam3d_use_camera_translation": not args.no_sam_cam_translation,
        "min_kpt_conf": float(args.min_kpt_conf),
        "visualize_keypoints": not args.no_kpt_vis,
        "visualize_joint_indices": not args.no_joint_indices,
    })

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sam3d_runner = None
    if not args.no_run_sam3d:
        sam3d_runner = Sam3DBodyDirectRunner(config)

    cap = open_video(Path(args.video))
    results = []
    try:
        frame_numbers = selected_frame_numbers(cap, args.frame_number, args.max_frames)
        if not frame_numbers:
            print("No video frames matched the requested frame filters.")
            return 1
        for frame_number in frame_numbers:
            frame_bgr = read_video_frame(cap, frame_number)
            h, w = frame_bgr.shape[:2]
            config["view_width"] = int(w)
            config["view_height"] = int(h)
            results.extend(process_frame_direct_360(frame_bgr, frame_number, output_dir, config, sam3d_runner))
    finally:
        cap.release()

    summary = {
        "video_path": str(Path(args.video).resolve()),
        "output_dir": str(output_dir.resolve()),
        "num_results": len(results),
        "sam3d_direct_api": sam3d_runner is not None,
        "sam3d_internal_detector": bool(config.get("sam3d_detector_name")),
        "sam3d_detector_name": config.get("sam3d_detector_name"),
        "known_intrinsics": bool(args.known_intrinsics),
        "results": results,
    }
    summary_path = output_dir / "direct_360_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved direct 360 comparison summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

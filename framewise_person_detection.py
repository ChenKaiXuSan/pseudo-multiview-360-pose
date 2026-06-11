#!/usr/bin/env python
"""
Frame-by-frame baseline: cubemap YOLO person detection on every frame, then YOLO
pose inside each detected bbox crop. This script is for comparing against the
CoTracker clip-based tracker.
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from test_360_detection import cubemap_sliding_detection  # noqa: E402
from cotracker_person_tracking import (  # noqa: E402
    COLORS,
    assign_track_ids,
    filter_overlapping_track_boxes,
    grid_points_in_box,
    pose_detections_from_boxes,
)


CONFIG = {
    # Paths
    "video_path": "/mnt/dataset/skiing/raw_new/kimura2_360.mp4",
    "output_path": "/mnt/dataset/skiing/raw_new/kimura2_360_framewise_detected.mp4",
    "frames_output_dir": "/mnt/dataset/skiing/raw_new/kimura2_360_framewise_frames",

    # Models
    "yolo_model_path": "yolov8n.pt",
    "pose_model_path": "yolov8n-pose.pt",

    # Per-frame cubemap bbox detection
    "window_width_ratio": 0.4,
    "step_size": 50,
    "face_size": 512,
    "conf": 0.7,
    "edge_samples": 9,
    "nms_angle_threshold_deg": 5.0,
    "nms_iou_threshold": 0.35,
    "nms_containment_threshold": 0.65,

    # Pose/keypoint extraction inside each detected box crop
    "use_pose_keypoints": True,
    "pose_conf": 0.35,
    "pose_kpt_conf": 0.35,
    "pose_crop_padding_ratio": 0.20,
    "min_pose_keypoints": 5,
    "fallback_to_grid_points": True,
    "points_per_box_axis": 3,
    "point_margin_ratio": 0.18,

    # Per-frame duplicate suppression
    "track_nms_iou_threshold": 0.30,
    "track_nms_containment_threshold": 0.65,

    # Cross-frame ID association
    "id_match_iou_threshold": 0.15,
    "id_match_center_threshold": 140.0,
    "id_match_area_ratio_threshold": 0.35,

    # Output
    "save_frames": True,
    "draw_points": True,
    "max_frames": None,
}


def make_framewise_boxes(detections, config):
    boxes = []
    for det in detections:
        if det.get("class") != "person":
            continue

        source = det.get("source", "grid")
        keypoints = det.get("keypoints") or []
        if config["use_pose_keypoints"] and len(keypoints) >= config["min_pose_keypoints"]:
            points = np.array([[x, y] for x, y, _ in keypoints], dtype=np.float32)
            source = "pose"
        elif config["fallback_to_grid_points"]:
            points = np.array(
                grid_points_in_box(
                    float(det["x1"]),
                    float(det["y1"]),
                    float(det["x2"]),
                    float(det["y2"]),
                    config["points_per_box_axis"],
                    config["point_margin_ratio"],
                ),
                dtype=np.float32,
            )
            source = "grid"
        else:
            points = np.empty((0, 2), dtype=np.float32)

        boxes.append({
            "id": -1,
            "box": (int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])),
            "source": source,
            "visible_points": int(len(points)),
            "score": float(det.get("conf", 0.0)),
            "points": points,
        })

    return filter_overlapping_track_boxes(boxes, config)


def assign_ids_to_frame_boxes(boxes, active_tracks, config, next_track_id):
    groups = []
    for box in boxes:
        x1, y1, x2, y2 = box["box"]
        groups.append({
            "id": -1,
            "seed_box": np.array([x1, y1, x2, y2], dtype=np.float32),
        })

    next_track_id, reused = assign_track_ids(
        groups,
        active_tracks,
        config,
        next_track_id,
        inv_scale_x=1.0,
        inv_scale_y=1.0,
    )

    for box, group in zip(boxes, groups):
        box["id"] = group["id"]

    return next_track_id, reused


def draw_frame(frame, boxes, draw_points=True):
    out = frame.copy()
    for box in boxes:
        track_id = box["id"]
        color = COLORS[(track_id - 1) % len(COLORS)]
        x1, y1, x2, y2 = box["box"]
        label = f"person #{track_id} {box.get('source', 'det')}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        if draw_points:
            for x, y in box.get("points", []):
                cv2.circle(out, (int(round(x)), int(round(y))), 3, color, -1)

    return out


def process_video(config):
    print("Loading YOLO model: " + config["yolo_model_path"])
    yolo_model = YOLO(config["yolo_model_path"])
    pose_model = None
    if config["use_pose_keypoints"]:
        print("Loading pose model: " + config["pose_model_path"])
        pose_model = YOLO(config["pose_model_path"])

    cap = cv2.VideoCapture(config["video_path"])
    if not cap.isOpened():
        print("Error: Cannot open video " + config["video_path"])
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video info: {width}x{height}, {fps} fps, {total_frames} frames")

    out = cv2.VideoWriter(
        config["output_path"],
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if config["save_frames"] and config["frames_output_dir"]:
        os.makedirs(config["frames_output_dir"], exist_ok=True)

    frame_count = 0
    next_track_id = 1
    active_tracks = {}

    while True:
        if config["max_frames"] is not None and frame_count >= config["max_frames"]:
            break

        ok, frame = cap.read()
        if not ok:
            break

        frame_count += 1
        print(f"Processing frame {frame_count}/{total_frames}")

        box_detections, _ = cubemap_sliding_detection(
            yolo_model,
            frame,
            window_width_ratio=config["window_width_ratio"],
            step_size=config["step_size"],
            conf=config["conf"],
            face_size=config["face_size"],
            edge_samples=config["edge_samples"],
            nms_angle_threshold_deg=config["nms_angle_threshold_deg"],
            nms_iou_threshold=config["nms_iou_threshold"],
            nms_containment_threshold=config["nms_containment_threshold"],
        )
        detections = pose_detections_from_boxes(frame, box_detections, pose_model, config)
        boxes = make_framewise_boxes(detections, config)
        next_track_id, reused = assign_ids_to_frame_boxes(boxes, active_tracks, config, next_track_id)
        active_tracks = {box["id"]: {"box": box["box"]} for box in boxes}

        annotated = draw_frame(frame, boxes, draw_points=config["draw_points"])
        out.write(annotated)

        if config["save_frames"] and config["frames_output_dir"]:
            frame_path = os.path.join(config["frames_output_dir"], f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_path, annotated)

        print(f"  detections={len(boxes)}, reused ids={reused}")

    cap.release()
    out.release()
    print("Output video saved to: " + config["output_path"])
    return True


if __name__ == "__main__":
    if not os.path.exists(CONFIG["video_path"]):
        print("Error: Video file not found: " + CONFIG["video_path"])
    else:
        process_video(CONFIG)

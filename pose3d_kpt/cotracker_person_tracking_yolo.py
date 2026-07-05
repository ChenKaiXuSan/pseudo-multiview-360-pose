#!/usr/bin/env python
"""
Person tracking on 360 video with cubemap YOLO detection + CoTracker point tracking.

CoTracker tracks points, not boxes. This script first uses the cubemap YOLO
person detector to get person boxes, runs a YOLO pose model inside each crop to
seed human keypoints, tracks those points with CoTracker, then reconstructs one
box per person from the visible tracked points.
"""

import os
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from test_360_detection import cubemap_sliding_detection  # noqa: E402


def resolve_model_path(model_path):
    """Resolve a model path relative to this script when needed."""
    path = Path(model_path)
    if path.is_absolute():
        return path
    return SCRIPT_DIR / path


def apply_video_name_output_paths(config):
    """Fill output paths from the input video name without overwriting explicit config values."""
    updated = dict(config)
    video_stem = Path(updated["video_path"]).stem
    output_root = updated.get("output_root_dir")
    if not output_root:
        output_root = str(Path(updated["video_path"]).resolve().parent)
    output_suffix = updated.get("output_suffix", "cotracker_yolo")
    output_prefix = f"{video_stem}_{output_suffix}"
    output_dir = Path(output_root) / video_stem

    updated.setdefault("output_path", str(output_dir / f"{output_prefix}_tracked.mp4"))
    updated.setdefault("frames_output_dir", str(output_dir / f"{output_prefix}_frames"))
    updated.setdefault("bbox_output_path", str(output_dir / f"{output_prefix}_bboxes.json"))
    if updated.get("save_view_debug") and not updated.get("view_debug_dir"):
        updated["view_debug_dir"] = str(output_dir / f"{output_prefix}_cubemap_views")
    return updated


CONFIG = {
    # Paths
    "video_path": "/mnt/dataset/skiing/360test/kimura2_360.mp4",
    "output_root_dir": "/mnt/dataset/skiing/360PoseFusion/output/pose3d_kpt/tracking",
    "output_suffix": "cotracker_yolo",

    # Models
    "yolo_model_path": "yolo26x.pt",
    "pose_model_path": "yolo26x-pose.pt",
    "cotracker_hub_repo": "facebookresearch/co-tracker",
    "cotracker_model_name": "cotracker3_offline",

    # Pose/keypoint seeding inside each detected box crop
    "use_pose_keypoints": True,
    "pose_conf": 0.8,
    "pose_kpt_conf": 0.8,
    "pose_crop_padding_ratio": 0.30,
    "min_pose_keypoints": 5,
    "fallback_to_grid_points": True,

    # Box detection on the first frame of each clip
    "window_width_ratio": 0.4,
    "step_size": 100,
    "face_size": 2048,
    "conf": 0.8,
    "edge_samples": 9,
    "nms_angle_threshold_deg": 5.0,
    "nms_iou_threshold": 0.35,
    "nms_containment_threshold": 0.65,
    "enable_extra_views": True,
    "horizontal_extra_yaws": [45, 135, 225, 315],
    "upper_extra_pitch": 55,
    "lower_extra_pitch": -55,
    "vertical_extra_yaws": [0, 90, 180, 270],
    "extra_view_fov_deg": 100,
    "save_view_debug": True,
    "view_debug_dir": None,

    # CoTracker clip settings
    "clip_len": 40,
    "clip_overlap": 20,
    "tracker_resize_width": 1024,
    "points_per_box_axis": 3,
    "point_margin_ratio": 0.18,
    "visibility_threshold": 0.5,
    "min_visible_points": 3,
    "box_padding_ratio": 0.08,

    # Per-frame track box de-duplication
    "track_nms_iou_threshold": 0.25,
    "track_nms_containment_threshold": 0.65,

    # Cross-clip ID association
    "id_match_iou_threshold": 0.15,
    "id_match_center_threshold": 140.0,
    "id_match_area_ratio_threshold": 0.35,

    # Output
    "save_frames": True,
    "draw_points": True,
    "max_frames": None,
}


COLORS = [
    (0, 180, 255),
    (80, 220, 80),
    (255, 120, 80),
    (220, 80, 220),
    (80, 180, 255),
    (255, 220, 80),
    (120, 255, 220),
]


def get_device():
    """Choose the best available torch device for inference."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_cotracker(repo, model_name, device):
    """Load the requested CoTracker model from torch hub and move it to the selected device."""
    print(f"Loading CoTracker model: {repo}/{model_name}")
    try:
        model = torch.hub.load(repo, model_name, trust_repo=True)
    except TypeError:
        model = torch.hub.load(repo, model_name)
    model = model.to(device)
    model.eval()
    return model


def resize_frame_for_tracker(frame, target_width):
    """Resize one frame to the tracker width while preserving aspect ratio."""
    h, w = frame.shape[:2]
    if not target_width or target_width <= 0 or w == target_width:
        return frame.copy(), 1.0, 1.0

    scale = target_width / float(w)
    target_height = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return resized, scale, scale


def frames_to_video_tensor(frames_bgr, tracker_resize_width, device):
    """Convert BGR frames into the BCHW video tensor expected by CoTracker."""
    resized_frames = []
    scale_x = scale_y = 1.0

    for frame in frames_bgr:
        resized, scale_x, scale_y = resize_frame_for_tracker(frame, tracker_resize_width)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        resized_frames.append(rgb)

    video_np = np.stack(resized_frames, axis=0)
    video = torch.from_numpy(video_np).permute(0, 3, 1, 2)[None].float().to(device)
    return video, scale_x, scale_y


def grid_points_in_box(x1, y1, x2, y2, points_per_axis, margin_ratio):
    """Generate evenly spaced fallback query points inside a detection box."""
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    mx = box_w * margin_ratio
    my = box_h * margin_ratio
    xs = np.linspace(x1 + mx, x2 - mx, points_per_axis)
    ys = np.linspace(y1 + my, y2 - my, points_per_axis)
    return [[float(x), float(y)] for y in ys for x in xs]


def padded_crop_bounds(det, frame_shape, padding_ratio):
    """Compute a clipped crop around a detection for pose-keypoint inference."""
    h, w = frame_shape[:2]
    x1 = float(det["x1"])
    y1 = float(det["y1"])
    x2 = float(det["x2"])
    y2 = float(det["y2"])
    pad_x = max(2.0, (x2 - x1) * padding_ratio)
    pad_y = max(2.0, (y2 - y1) * padding_ratio)
    crop_x1 = int(np.clip(np.floor(x1 - pad_x), 0, w - 1))
    crop_y1 = int(np.clip(np.floor(y1 - pad_y), 0, h - 1))
    crop_x2 = int(np.clip(np.ceil(x2 + pad_x), crop_x1 + 1, w))
    crop_y2 = int(np.clip(np.ceil(y2 + pad_y), crop_y1 + 1, h))
    return crop_x1, crop_y1, crop_x2, crop_y2


def keypoints_from_pose_crop(crop, pose_model, config):
    """Run the pose model on a crop and return confident keypoints in crop coordinates."""
    if crop.size == 0:
        return []

    results = pose_model(crop, verbose=False, conf=config["pose_conf"])
    if not results or results[0].boxes is None or results[0].keypoints is None:
        return []
    if len(results[0].boxes) == 0:
        return []

    best_idx = int(torch.argmax(results[0].boxes.conf).item())
    keypoints_xy = results[0].keypoints.xy[best_idx].detach().cpu().numpy()
    keypoints_conf = None
    if results[0].keypoints.conf is not None:
        keypoints_conf = results[0].keypoints.conf[best_idx].detach().cpu().numpy()

    kpts = []
    h, w = crop.shape[:2]
    for kpt_idx, (x, y) in enumerate(keypoints_xy):
        score = 1.0 if keypoints_conf is None else float(keypoints_conf[kpt_idx])
        if score < config["pose_kpt_conf"]:
            continue
        if x <= 0 or y <= 0 or x >= w or y >= h:
            continue
        kpts.append([float(x), float(y), score])
    return kpts


def pose_detections_from_boxes(frame, box_detections, pose_model, config):
    """Attach pose keypoints or grid fallback metadata to each person detection box."""
    detections = []
    h, w = frame.shape[:2]

    for det in box_detections:
        if det.get("class") != "person":
            continue

        crop_x1, crop_y1, crop_x2, crop_y2 = padded_crop_bounds(
            det,
            frame.shape,
            config["pose_crop_padding_ratio"],
        )
        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        kpts = []
        if config["use_pose_keypoints"] and pose_model is not None:
            crop_kpts = keypoints_from_pose_crop(crop, pose_model, config)
            kpts = [[x + crop_x1, y + crop_y1, score] for x, y, score in crop_kpts]

        detections.append({
            "x1": int(np.clip(det["x1"], 0, w - 1)),
            "y1": int(np.clip(det["y1"], 0, h - 1)),
            "x2": int(np.clip(det["x2"], 0, w - 1)),
            "y2": int(np.clip(det["y2"], 0, h - 1)),
            "conf": float(det.get("conf", 0.0)),
            "class": "person",
            "keypoints": kpts,
            "source": "pose" if len(kpts) >= config["min_pose_keypoints"] else "grid",
        })

    return detections


def make_query_points(detections, scale_x, scale_y, points_per_axis, margin_ratio, config, query_time=0):
    """Build CoTracker query points and group metadata from seeded detections."""
    queries = []
    groups = []

    for det in detections:
        if det.get("class") != "person":
            continue

        x1 = float(det["x1"]) * scale_x
        y1 = float(det["y1"]) * scale_y
        x2 = float(det["x2"]) * scale_x
        y2 = float(det["y2"]) * scale_y

        seed_points = []
        source = det.get("source", "grid")
        keypoints = det.get("keypoints") or []
        if config["use_pose_keypoints"] and len(keypoints) >= config["min_pose_keypoints"]:
            for x, y, _ in keypoints:
                seed_points.append([float(x) * scale_x, float(y) * scale_y])
            source = "pose"
        elif config["fallback_to_grid_points"]:
            seed_points = grid_points_in_box(x1, y1, x2, y2, points_per_axis, margin_ratio)
            source = "grid"

        if not seed_points:
            continue

        start = len(queries)
        for x, y in seed_points:
            queries.append([float(query_time), float(x), float(y)])

        end = len(queries)
        groups.append({
            "id": len(groups) + 1,
            "start": start,
            "end": end,
            "seed_box": np.array([x1, y1, x2, y2], dtype=np.float32),
            "seed_points": np.array(seed_points, dtype=np.float32),
            "conf": float(det.get("conf", 0.0)),
            "source": source,
        })

    if not queries:
        return None, []

    return np.array(queries, dtype=np.float32), groups


def reconstruct_box_from_points(points, visibility, group, frame_shape, visibility_threshold,
                                min_visible_points, padding_ratio):
    """Reconstruct a person bbox from visible tracked points using the seed-box scale."""
    visible = visibility >= visibility_threshold
    if int(visible.sum()) < min_visible_points:
        return None

    tracked = points[visible]
    seed_box = group["seed_box"]
    seed_points = group["seed_points"]

    seed_w = max(1.0, float(seed_box[2] - seed_box[0]))
    seed_h = max(1.0, float(seed_box[3] - seed_box[1]))
    seed_span_w = max(1.0, float(seed_points[:, 0].max() - seed_points[:, 0].min()))
    seed_span_h = max(1.0, float(seed_points[:, 1].max() - seed_points[:, 1].min()))

    tracked_x1 = float(np.percentile(tracked[:, 0], 5))
    tracked_y1 = float(np.percentile(tracked[:, 1], 5))
    tracked_x2 = float(np.percentile(tracked[:, 0], 95))
    tracked_y2 = float(np.percentile(tracked[:, 1], 95))
    tracked_w = max(1.0, tracked_x2 - tracked_x1)
    tracked_h = max(1.0, tracked_y2 - tracked_y1)

    box_w = max(seed_w * 0.45, tracked_w * seed_w / seed_span_w)
    box_h = max(seed_h * 0.45, tracked_h * seed_h / seed_span_h)
    cx = float(np.median(tracked[:, 0]))
    cy = float(np.median(tracked[:, 1]))

    pad_w = box_w * padding_ratio
    pad_h = box_h * padding_ratio
    x1 = cx - box_w * 0.5 - pad_w
    y1 = cy - box_h * 0.5 - pad_h
    x2 = cx + box_w * 0.5 + pad_w
    y2 = cy + box_h * 0.5 + pad_h

    h, w = frame_shape[:2]
    x1 = int(np.clip(round(x1), 0, w - 1))
    y1 = int(np.clip(round(y1), 0, h - 1))
    x2 = int(np.clip(round(x2), 0, w - 1))
    y2 = int(np.clip(round(y2), 0, h - 1))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def draw_tracking(frame, boxes, points_by_id=None):
    """Draw track boxes and optional tracked query points on a frame copy."""
    out = frame.copy()
    for box in boxes:
        track_id = box["id"]
        color = COLORS[(track_id - 1) % len(COLORS)]
        x1, y1, x2, y2 = box["box"]
        label = f"person #{track_id} {box.get('source', 'cotracker')}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    if points_by_id:
        for track_id, pts in points_by_id.items():
            color = COLORS[(track_id - 1) % len(COLORS)]
            for x, y in pts:
                cv2.circle(out, (int(round(x)), int(round(y))), 3, color, -1)

    return out


def track_points_center_xy(points_xy, frame_width):
    """Estimate a seam-aware pixel center from tracked 360-frame points."""
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 2 or len(pts) == 0:
        return None
    finite = np.isfinite(pts[:, :2]).all(axis=1)
    pts = pts[finite, :2]
    if len(pts) == 0:
        return None

    width = max(1.0, float(frame_width))
    angles = (pts[:, 0] / width) * 2.0 * np.pi
    mean_angle = float(np.arctan2(np.sin(angles).mean(), np.cos(angles).mean()))
    center_x = ((mean_angle / (2.0 * np.pi)) % 1.0) * width
    center_y = float(np.median(pts[:, 1]))
    return [float(center_x), center_y]


def points_to_jsonable(points_xy):
    """Convert tracked points to compact JSON-safe [x, y] rows."""
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 2:
        return []
    rows = []
    for x, y in pts[:, :2]:
        if np.isfinite(x) and np.isfinite(y):
            rows.append([float(x), float(y)])
    return rows



def serialize_track_boxes(boxes):
    """Convert internal track boxes to JSON-safe bbox records."""
    serialized = []
    for box in boxes:
        x1, y1, x2, y2 = box["box"]
        row = {
            "track_id": int(box["id"]),
            "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
            "conf": float(box.get("score", 0.0)),
            "source": box.get("source", "cotracker"),
            "visible_points": int(box.get("visible_points", 0)),
        }
        track_points = points_to_jsonable(box.get("track_points_xy", []))
        if track_points:
            row["track_points_xy"] = track_points
            row["track_points_source"] = box.get("track_points_source", box.get("source", "cotracker"))
        center_xy = box.get("center_xy")
        if center_xy is not None and len(center_xy) >= 2:
            row["center_xy"] = [float(center_xy[0]), float(center_xy[1])]
            row["center_source"] = box.get("center_source", "track_points" if track_points else "bbox")
        serialized.append(row)
    return serialized


def box_containment(box1, box2):
    """Return the overlap area divided by the smaller input box area."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x1 >= x2 or y1 >= y2:
        return 0.0

    inter = (x2 - x1) * (y2 - y1)
    area1 = max(1.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(1.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    return inter / min(area1, area2)


def vertical_overlap_ratio(box1, box2):
    """Measure the vertical overlap ratio relative to the shorter box."""
    y1 = max(box1[1], box2[1])
    y2 = min(box1[3], box2[3])
    if y1 >= y2:
        return 0.0
    h1 = max(1.0, float(box1[3] - box1[1]))
    h2 = max(1.0, float(box2[3] - box2[1]))
    return float((y2 - y1) / min(h1, h2))


def horizontal_overlap_or_small_gap_ratio(box1, box2):
    """Measure horizontal overlap, or return a negative normalized gap."""
    w1 = max(1.0, float(box1[2] - box1[0]))
    w2 = max(1.0, float(box2[2] - box2[0]))
    overlap = min(box1[2], box2[2]) - max(box1[0], box2[0])
    if overlap >= 0:
        return float(overlap / min(w1, w2))
    gap = -overlap
    return float(-gap / min(w1, w2))


def is_split_projection_duplicate(box1, box2):
    """Detect duplicate boxes caused by split 360-degree projection views."""
    vertical_ratio = vertical_overlap_ratio(box1, box2)
    horizontal_ratio = horizontal_overlap_or_small_gap_ratio(box1, box2)
    h1 = max(1.0, float(box1[3] - box1[1]))
    h2 = max(1.0, float(box2[3] - box2[1]))
    height_ratio = min(h1, h2) / max(h1, h2)
    union_w = max(float(box1[2]), float(box2[2])) - min(float(box1[0]), float(box2[0]))
    union_h = max(float(box1[3]), float(box2[3])) - min(float(box1[1]), float(box2[1]))
    union_aspect = union_w / max(1.0, union_h)

    if vertical_ratio < 0.75 or height_ratio < 0.60:
        return False
    if horizontal_ratio >= 0.15:
        return True
    return horizontal_ratio >= -0.25 and union_aspect <= 0.75


def is_partial_projection_fragment(box1, box2):
    """Detect small projection fragments that belong to a larger person box."""
    vertical_ratio = vertical_overlap_ratio(box1, box2)
    if vertical_ratio < 0.60:
        return False

    w1 = max(1.0, float(box1[2] - box1[0]))
    w2 = max(1.0, float(box2[2] - box2[0]))
    h1 = max(1.0, float(box1[3] - box1[1]))
    h2 = max(1.0, float(box2[3] - box2[1]))
    small_w = min(w1, w2)
    large_w = max(w1, w2)
    small_h = min(h1, h2)
    large_h = max(h1, h2)
    horizontal_ratio = horizontal_overlap_or_small_gap_ratio(box1, box2)

    if small_w / large_w <= 0.35 and small_h / large_h >= 0.45:
        return horizontal_ratio >= -1.60

    short_box, tall_box = (box1, box2) if h1 <= h2 else (box2, box1)
    short_h = min(h1, h2)
    tall_h = max(h1, h2)
    height_ratio = short_h / tall_h
    if 0.18 <= height_ratio <= 0.55:
        short_cx, short_cy = box_center(short_box)
        tall_cx, tall_cy = box_center(tall_box)
        tall_w = max(1.0, float(tall_box[2] - tall_box[0]))
        tall_top_band = float(tall_box[1]) + tall_h * 0.45
        center_close = abs(short_cx - tall_cx) <= tall_w * 0.85
        upper_overlap = short_cy <= tall_top_band
        width_reasonable = max(w1, w2) / max(1.0, min(w1, w2)) <= 2.2
        if center_close and upper_overlap and width_reasonable:
            return True

    return False


def merge_track_boxes(primary, secondary):
    """Merge two boxes for the same projected person into one enclosing record."""
    merged = dict(primary)
    box1 = primary["box"]
    box2 = secondary["box"]
    merged["box"] = (
        min(int(box1[0]), int(box2[0])),
        min(int(box1[1]), int(box2[1])),
        max(int(box1[2]), int(box2[2])),
        max(int(box1[3]), int(box2[3])),
    )
    merged["score"] = max(float(primary.get("score", 0.0)), float(secondary.get("score", 0.0)))
    merged["visible_points"] = max(int(primary.get("visible_points", 0)), int(secondary.get("visible_points", 0)))

    primary_has_points = bool(primary.get("track_points_xy"))
    secondary_has_points = bool(secondary.get("track_points_xy"))
    secondary_is_better = int(secondary.get("visible_points", 0)) > int(primary.get("visible_points", 0))
    if secondary_has_points and (secondary_is_better or not primary_has_points):
        for key in ("track_points_xy", "track_points_source", "center_xy", "center_source"):
            if key in secondary:
                merged[key] = secondary[key]
    return merged


def filter_overlapping_track_boxes(boxes, config):
    """Suppress or merge overlapping track boxes before drawing and serialization."""
    if not boxes:
        return boxes

    def rank(box):
        """Rank boxes by source quality, visible points, and detector confidence."""
        source_bonus = 1.0 if box.get("source") == "pose" else 0.0
        return (source_bonus, box.get("visible_points", 0), box.get("score", 0.0))

    kept = []
    for box in sorted(boxes, key=rank, reverse=True):
        should_keep = True
        for kept_box in kept:
            iou = box_iou(box["box"], kept_box["box"])
            containment = box_containment(box["box"], kept_box["box"])
            split_duplicate = is_split_projection_duplicate(box["box"], kept_box["box"])
            partial_fragment = is_partial_projection_fragment(box["box"], kept_box["box"])
            if split_duplicate:
                kept_idx = kept.index(kept_box)
                kept[kept_idx] = merge_track_boxes(kept_box, box)
                should_keep = False
                break
            if (
                iou >= config["track_nms_iou_threshold"]
                or containment >= config["track_nms_containment_threshold"]
                or partial_fragment
            ):
                should_keep = False
                break
        if should_keep:
            kept.append(box)

    kept.sort(key=lambda x: x["id"])
    return kept


def read_overlapped_clip(cap, clip_len, clip_overlap, max_frames, next_input_frame, overlap_frames):
    """Read the next clip while carrying overlap frames for continuity."""
    overlap_frames = list(overlap_frames)
    if len(overlap_frames) >= clip_len:
        overlap_frames = overlap_frames[-max(0, clip_len - 1):]

    frames = list(overlap_frames)
    start_frame = next_input_frame - len(overlap_frames)
    target_new_frames = clip_len - len(overlap_frames)
    new_count = 0

    while new_count < target_new_frames:
        if max_frames is not None and next_input_frame >= max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        next_input_frame += 1
        new_count += 1

    if new_count == 0:
        return [], start_frame, 0, next_input_frame, []

    output_start_idx = len(overlap_frames)
    keep_overlap = min(max(0, clip_overlap), max(0, len(frames) - 1))
    next_overlap_frames = frames[-keep_overlap:] if keep_overlap > 0 else []
    return frames, start_frame, output_start_idx, next_input_frame, next_overlap_frames


def box_iou(box1, box2):
    """Compute intersection-over-union for two xyxy boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x1 >= x2 or y1 >= y2:
        return 0.0

    inter = (x2 - x1) * (y2 - y1)
    area1 = max(1.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(1.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    return inter / (area1 + area2 - inter)


def box_center_distance(box1, box2):
    """Return Euclidean distance between two box centers."""
    cx1 = (box1[0] + box1[2]) * 0.5
    cy1 = (box1[1] + box1[3]) * 0.5
    cx2 = (box2[0] + box2[2]) * 0.5
    cy2 = (box2[1] + box2[3]) * 0.5
    return float(np.hypot(cx1 - cx2, cy1 - cy2))


def box_area_ratio(box1, box2):
    """Return the smaller-to-larger area ratio for two boxes."""
    area1 = max(1.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(1.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    return min(area1, area2) / max(area1, area2)


def assign_track_ids(groups, active_tracks, config, next_track_id, inv_scale_x, inv_scale_y):
    """Reuse active track IDs for new seed groups when geometry matches across clips."""
    candidates = []
    for group_idx, group in enumerate(groups):
        seed_box = group["seed_box"]
        full_box = (
            float(seed_box[0] * inv_scale_x),
            float(seed_box[1] * inv_scale_y),
            float(seed_box[2] * inv_scale_x),
            float(seed_box[3] * inv_scale_y),
        )

        for track_id, track in active_tracks.items():
            prev_box = track["box"]
            iou = box_iou(full_box, prev_box)
            center_dist = box_center_distance(full_box, prev_box)
            area_ratio = box_area_ratio(full_box, prev_box)
            center_match = (
                center_dist <= config["id_match_center_threshold"]
                and area_ratio >= config["id_match_area_ratio_threshold"]
            )
            if iou >= config["id_match_iou_threshold"] or center_match:
                score = iou * 3.0 + area_ratio - center_dist / max(1.0, config["id_match_center_threshold"])
                candidates.append((score, group_idx, track_id))

    candidates.sort(reverse=True)
    used_groups = set()
    used_track_ids = set()
    assigned = 0

    for _, group_idx, track_id in candidates:
        if group_idx in used_groups or track_id in used_track_ids:
            continue
        groups[group_idx]["id"] = track_id
        used_groups.add(group_idx)
        used_track_ids.add(track_id)
        assigned += 1

    for idx, group in enumerate(groups):
        if idx in used_groups:
            continue
        group["id"] = next_track_id
        next_track_id += 1

    return next_track_id, assigned


def detection_to_box(det):
    """Convert one detector result dictionary into an integer xyxy tuple."""
    return (
        int(round(float(det["x1"]))),
        int(round(float(det["y1"]))),
        int(round(float(det["x2"]))),
        int(round(float(det["y2"]))),
    )


def box_center(box):
    """Return the center point of an xyxy box."""
    return ((float(box[0]) + float(box[2])) * 0.5, (float(box[1]) + float(box[3])) * 0.5)


def box_contains_point(box, point):
    """Return whether a point lies inside an xyxy box."""
    x, y = point
    return float(box[0]) <= x <= float(box[2]) and float(box[1]) <= y <= float(box[3])


def assign_frame_detection_boxes(detections, track_refs, config, next_track_id):
    """Match per-frame YOLO detections to tracker references and assign stable IDs."""
    person_detections = [det for det in detections if det.get("class") == "person"]
    candidates = []
    center_threshold = max(1.0, float(config["id_match_center_threshold"]))

    for det_idx, det in enumerate(person_detections):
        det_box = detection_to_box(det)
        det_center = box_center(det_box)
        for ref_idx, ref in enumerate(track_refs):
            ref_box = ref["box"]
            ref_center = box_center(ref_box)
            iou = box_iou(det_box, ref_box)
            center_dist = box_center_distance(det_box, ref_box)
            area_ratio = box_area_ratio(det_box, ref_box)
            center_match = center_dist <= center_threshold and area_ratio >= config["id_match_area_ratio_threshold"]
            contains_match = box_contains_point(det_box, ref_center) or box_contains_point(ref_box, det_center)
            if iou >= config["id_match_iou_threshold"] or center_match or contains_match:
                score = iou * 3.0 + area_ratio - center_dist / center_threshold
                candidates.append((score, det_idx, ref_idx))

    candidates.sort(reverse=True)
    assigned_dets = {}
    used_refs = set()
    for _score, det_idx, ref_idx in candidates:
        if det_idx in assigned_dets or ref_idx in used_refs:
            continue
        assigned_dets[det_idx] = track_refs[ref_idx]
        used_refs.add(ref_idx)

    boxes = []
    for det_idx, det in enumerate(person_detections):
        det_box = detection_to_box(det)
        ref = assigned_dets.get(det_idx)
        if ref is None:
            track_id = next_track_id
            next_track_id += 1
            source = "yolo"
            visible_points = 0
        else:
            track_id = int(ref["id"])
            source = ref.get("source", "cotracker")
            visible_points = int(ref.get("visible_points", 0))

        box_record = {
            "id": track_id,
            "box": det_box,
            "source": source,
            "visible_points": visible_points,
            "score": float(det.get("conf", 0.0)),
        }
        if ref is not None:
            for key in ("track_points_xy", "track_points_source", "center_xy", "center_source"):
                if key in ref:
                    box_record[key] = ref[key]
        boxes.append(box_record)

    boxes.sort(key=lambda item: item["id"])
    return boxes, next_track_id, len(assigned_dets)


def detection_view_debug_path(config, frame_number):
    """Return the per-frame detection-view debug image path when enabled."""
    if not config.get("save_view_debug"):
        return None
    view_debug_dir = config.get("view_debug_dir")
    if not view_debug_dir:
        return None
    return os.path.join(view_debug_dir, f"frame_{frame_number:06d}.jpg")


def detection_kwargs(config, frame_number=None):
    """Build keyword arguments for cubemap sliding detection from config."""
    return {
        "window_width_ratio": config["window_width_ratio"],
        "step_size": config["step_size"],
        "conf": config["conf"],
        "face_size": config["face_size"],
        "edge_samples": config["edge_samples"],
        "nms_angle_threshold_deg": config["nms_angle_threshold_deg"],
        "nms_iou_threshold": config["nms_iou_threshold"],
        "nms_containment_threshold": config["nms_containment_threshold"],
        "enable_extra_views": config.get("enable_extra_views", False),
        "horizontal_extra_yaws": config.get("horizontal_extra_yaws"),
        "upper_extra_pitch": config.get("upper_extra_pitch", 55),
        "lower_extra_pitch": config.get("lower_extra_pitch", -55),
        "vertical_extra_yaws": config.get("vertical_extra_yaws"),
        "extra_view_fov_deg": config.get("extra_view_fov_deg", 100),
        "view_debug_path": detection_view_debug_path(config, frame_number) if frame_number is not None else None,
    }


def process_clip(
    frames,
    yolo_model,
    pose_model,
    cotracker,
    config,
    device,
    next_track_id,
    active_tracks,
    seed_frame_idx=0,
    start_frame_number=1,
):
    """Track all seeded people through one overlapped clip and return annotated frames plus bbox records."""
    seed_frame_idx = min(max(0, seed_frame_idx), len(frames) - 1)
    seed_frame = frames[seed_frame_idx]
    seed_frame_number = start_frame_number + seed_frame_idx
    # Seed each clip from YOLO detections, then enrich each box with pose
    # keypoints when available before creating CoTracker queries.
    box_detections, _ = cubemap_sliding_detection(
        yolo_model,
        seed_frame,
        **detection_kwargs(config, seed_frame_number),
    )
    detections = pose_detections_from_boxes(seed_frame, box_detections, pose_model, config)

    video, scale_x, scale_y = frames_to_video_tensor(frames, config["tracker_resize_width"], device)
    query_points, groups = make_query_points(
        detections,
        scale_x=scale_x,
        scale_y=scale_y,
        points_per_axis=config["points_per_box_axis"],
        margin_ratio=config["point_margin_ratio"],
        config=config,
        query_time=seed_frame_idx,
    )

    if query_points is None:
        per_frame_nums = [0 for _ in frames]
        per_frame_boxes = [{"boxes": []} for _ in frames]
        return frames, next_track_id, 0, {}, 0, per_frame_nums, per_frame_boxes

    inv_scale_x = 1.0 / scale_x
    inv_scale_y = 1.0 / scale_y
    next_track_id, reused_tracks = assign_track_ids(
        groups,
        active_tracks,
        config,
        next_track_id,
        inv_scale_x,
        inv_scale_y,
    )

    queries = torch.from_numpy(query_points)[None].float().to(device)

    with torch.no_grad():
        pred_tracks, pred_visibility = cotracker(video, queries=queries)

    tracks = pred_tracks[0].detach().cpu().numpy()
    visibility = pred_visibility[0].detach().cpu().numpy()
    if visibility.ndim == 3:
        visibility = visibility[..., 0]

    annotated_frames = []
    latest_tracks = {}
    per_frame_nums = []
    per_frame_boxes = []
    previous_detection_refs = {}

    for frame_idx, frame in enumerate(frames):
        points_by_id = {}
        tracker_shape = video.shape[-2], video.shape[-1]
        tracker_refs = {}

        for group in groups:
            start = group["start"]
            end = group["end"]
            group_points = tracks[frame_idx, start:end]
            group_visibility = visibility[frame_idx, start:end]
            box = reconstruct_box_from_points(
                group_points,
                group_visibility,
                group,
                frame_shape=(tracker_shape[0], tracker_shape[1], 3),
                visibility_threshold=config["visibility_threshold"],
                min_visible_points=config["min_visible_points"],
                padding_ratio=config["box_padding_ratio"],
            )
            if box is None:
                continue

            x1, y1, x2, y2 = box
            full_box = (
                int(round(x1 * inv_scale_x)),
                int(round(y1 * inv_scale_y)),
                int(round(x2 * inv_scale_x)),
                int(round(y2 * inv_scale_y)),
            )
            visible = group_visibility >= config["visibility_threshold"]
            visible_count = int(visible.sum())
            visible_pts = group_points[visible].copy()
            if visible_count > 0:
                visible_pts[:, 0] *= inv_scale_x
                visible_pts[:, 1] *= inv_scale_y
            track_id = int(group["id"])
            tracker_ref = {
                "id": track_id,
                "box": full_box,
                "source": group.get("source", "cotracker"),
                "visible_points": visible_count,
                "score": float(group.get("conf", 0.0)),
            }
            track_points = points_to_jsonable(visible_pts)
            if track_points:
                tracker_ref["track_points_xy"] = track_points
                tracker_ref["track_points_source"] = group.get("source", "cotracker")
                tracker_ref["center_xy"] = track_points_center_xy(track_points, frame.shape[1])
                tracker_ref["center_source"] = "track_points"
            tracker_refs[track_id] = tracker_ref

            if config["draw_points"] and visible_count > 0:
                points_by_id[track_id] = visible_pts

        # Prefer current tracker geometry, but keep the previous detection as a
        # fallback so short visibility drops do not immediately break IDs.
        reference_by_id = dict(previous_detection_refs)
        reference_by_id.update(tracker_refs)
        frame_number = start_frame_number + frame_idx
        current_detections, _ = cubemap_sliding_detection(
            yolo_model,
            frame,
            **detection_kwargs(config, frame_number),
        )
        boxes, next_track_id, _matched = assign_frame_detection_boxes(
            current_detections,
            list(reference_by_id.values()),
            config,
            next_track_id,
        )

        boxes = filter_overlapping_track_boxes(boxes, config)
        kept_ids = {box["id"] for box in boxes}
        points_by_id = {track_id: pts for track_id, pts in points_by_id.items() if track_id in kept_ids}
        latest_tracks = {box["id"]: {"box": box["box"]} for box in boxes}
        previous_detection_refs = {
            box["id"]: {
                "id": box["id"],
                "box": box["box"],
                "source": box.get("source", "yolo"),
                "visible_points": box.get("visible_points", 0),
                "score": box.get("score", 0.0),
            }
            for box in boxes
        }
        annotated_frames.append(draw_tracking(frame, boxes, points_by_id if config["draw_points"] else None))
        per_frame_nums.append(len(boxes))
        per_frame_boxes.append({"boxes": serialize_track_boxes(boxes)})

    return annotated_frames, next_track_id, len(groups), latest_tracks, reused_tracks, per_frame_nums, per_frame_boxes


def process_video(config):
    """Run the full video tracking pipeline and write annotated video plus bbox JSON."""
    config = apply_video_name_output_paths(config)
    if config["clip_overlap"] >= config["clip_len"]:
        raise ValueError("clip_overlap must be smaller than clip_len")

    device = get_device()
    print(f"Device: {device}")

    yolo_model_path = resolve_model_path(config["yolo_model_path"])
    print("Loading YOLO model: " + str(yolo_model_path))
    yolo_model = YOLO(str(yolo_model_path))
    pose_model = None
    if config["use_pose_keypoints"]:
        pose_model_path = resolve_model_path(config["pose_model_path"])
        print("Loading pose model: " + str(pose_model_path))
        pose_model = YOLO(str(pose_model_path))
    cotracker = load_cotracker(config["cotracker_hub_repo"], config["cotracker_model_name"], device)

    cap = cv2.VideoCapture(config["video_path"])
    if not cap.isOpened():
        print("Error: Cannot open video " + config["video_path"])
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video info: {width}x{height}, {fps} fps, {total_frames} frames")

    output_dir = os.path.dirname(config["output_path"])
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    out = cv2.VideoWriter(
        config["output_path"],
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if config["save_frames"] and config["frames_output_dir"]:
        os.makedirs(config["frames_output_dir"], exist_ok=True)

    next_input_frame = 0
    overlap_frames = []
    next_track_id = 1
    active_tracks = {}
    clip_index = 0

    # --- 汇总统计累计 ---
    stats_total_clips = 0
    stats_total_output_frames = 0
    stats_frames_with_tracks = 0
    stats_total_person_detections = 0
    stats_max_persons_in_a_frame = 0
    stats_all_track_ids = set()
    bbox_records = []

    while True:
        frames, start_frame, output_start_idx, next_input_frame, overlap_frames = read_overlapped_clip(
            cap,
            config["clip_len"],
            config["clip_overlap"],
            config["max_frames"],
            next_input_frame,
            overlap_frames,
        )
        if not frames:
            break

        clip_index += 1
        first_output_frame = start_frame + output_start_idx + 1
        print(
            f"Processing clip {clip_index}, frames={len(frames)}, "
            f"overlap={output_start_idx}, output_start_frame={first_output_frame}"
        )
        (
            annotated_frames,
            next_track_id,
            num_tracks,
            active_tracks,
            reused_tracks,
            per_frame_nums,
            per_frame_boxes,
        ) = process_clip(
            frames,
            yolo_model,
            pose_model,
            cotracker,
            config,
            device,
            next_track_id,
            active_tracks,
            seed_frame_idx=output_start_idx,
            start_frame_number=start_frame + 1,
        )

        print(f"  tracks seeded: {num_tracks}, reused ids: {reused_tracks}")

        stats_total_clips += 1

        for local_idx in range(output_start_idx, len(annotated_frames)):
            frame = annotated_frames[local_idx]
            frame_number = start_frame + local_idx + 1
            frame_boxes = per_frame_boxes[local_idx]["boxes"]
            num_this_frame = per_frame_nums[local_idx]

            if num_this_frame > 0:
                stats_frames_with_tracks += 1
            stats_total_person_detections += num_this_frame
            stats_max_persons_in_a_frame = max(stats_max_persons_in_a_frame, num_this_frame)
            stats_total_output_frames += 1
            stats_all_track_ids.update(box["track_id"] for box in frame_boxes)

            bbox_records.append({
                "frame_number": int(frame_number),
                "time_sec": float((frame_number - 1) / fps) if fps > 0 else None,
                "boxes": frame_boxes,
            })

            out.write(frame)
            if config["save_frames"] and config["frames_output_dir"]:
                frame_path = os.path.join(config["frames_output_dir"], f"frame_{frame_number:06d}.jpg")
                cv2.imwrite(frame_path, frame)

        if config["max_frames"] is not None and next_input_frame >= config["max_frames"]:
            break

    cap.release()
    out.release()
    print("Output video saved to: " + config["output_path"])

    bbox_output_path = config.get("bbox_output_path")
    if bbox_output_path:
        bbox_dir = os.path.dirname(bbox_output_path)
        if bbox_dir:
            os.makedirs(bbox_dir, exist_ok=True)
        bbox_payload = {
            "video_path": config["video_path"],
            "output_path": config["output_path"],
            "fps": float(fps),
            "width": int(width),
            "height": int(height),
            "bbox_format": "xyxy",
            "frames": bbox_records,
        }
        with open(bbox_output_path, "w", encoding="utf-8") as f:
            json.dump(bbox_payload, f, indent=2)
        print("BBox JSON saved to: " + bbox_output_path)

    # --- 打印汇总统计 ---
    detection_rate = stats_frames_with_tracks / max(1, stats_total_output_frames) * 100 if stats_total_output_frames > 0 else 0.0
    avg_persons_per_frame = stats_total_person_detections / max(1, stats_total_output_frames) if stats_total_output_frames > 0 else 0.0

    print("\n" + "=" * 60)
    print(f"{'360 VIDEO PERSON TRACKING — SUMMARY':^58}")
    print("=" * 60)
    print(f"{'Video resolution:':<35} {width} x {height}")
    print(f"{'Total video frames (file):':<35} {total_frames}")
    print("-" * 60)
    print(f"{'Clips processed:':<35} {stats_total_clips}")
    print(f"{'Frames output:':<35} {stats_total_output_frames}")
    print("-" * 60)
    print(f"{'Frames with tracks:':<35} {stats_frames_with_tracks}")
    print(f"{'Detection rate:':<35} {detection_rate:.1f}%")
    print("-" * 60)
    print(f"{'Avg persons per frame:':<35} {avg_persons_per_frame:.2f}")
    print(f"{'Max persons in a frame:':<35} {stats_max_persons_in_a_frame}")
    print(f"{'Unique track ids:':<35} {len(stats_all_track_ids)}")
    print("=" * 60)

    return True


if __name__ == "__main__":
    if not os.path.exists(CONFIG["video_path"]):
        print("Error: Video file not found: " + CONFIG["video_path"])
    else:
        process_video(CONFIG)

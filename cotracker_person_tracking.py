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


CONFIG = {
    # Paths
    "video_path": "/mnt/dataset/skiing/raw_new/kimura2_360.mp4",
    "output_path": "/mnt/dataset/skiing/raw_new/kimura2_360_cotracker_tracked.mp4",
    "frames_output_dir": "/mnt/dataset/skiing/raw_new/kimura2_360_cotracker_frames",
    "bbox_output_path": "/mnt/dataset/skiing/raw_new/kimura2_360_cotracker_bboxes.json",

    # Models
    "yolo_model_path": "yolov8n.pt",
    "pose_model_path": "yolov8n-pose.pt",
    "cotracker_hub_repo": "facebookresearch/co-tracker",
    "cotracker_model_name": "cotracker3_offline",

    # Pose/keypoint seeding inside each detected box crop
    "use_pose_keypoints": True,
    "pose_conf": 0.35,
    "pose_kpt_conf": 0.35,
    "pose_crop_padding_ratio": 0.20,
    "min_pose_keypoints": 5,
    "fallback_to_grid_points": True,

    # Box detection on the first frame of each clip
    "window_width_ratio": 0.4,
    "step_size": 50,
    "face_size": 512,
    "conf": 0.7,
    "edge_samples": 9,
    "nms_angle_threshold_deg": 5.0,
    "nms_iou_threshold": 0.35,
    "nms_containment_threshold": 0.65,

    # CoTracker clip settings
    "clip_len": 60,
    "clip_overlap": 20,
    "tracker_resize_width": 1024,
    "points_per_box_axis": 3,
    "point_margin_ratio": 0.18,
    "visibility_threshold": 0.5,
    "min_visible_points": 3,
    "box_padding_ratio": 0.08,

    # Per-frame track box de-duplication
    "track_nms_iou_threshold": 0.30,
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
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_cotracker(repo, model_name, device):
    print(f"Loading CoTracker model: {repo}/{model_name}")
    try:
        model = torch.hub.load(repo, model_name, trust_repo=True)
    except TypeError:
        model = torch.hub.load(repo, model_name)
    model = model.to(device)
    model.eval()
    return model


def resize_frame_for_tracker(frame, target_width):
    h, w = frame.shape[:2]
    if not target_width or target_width <= 0 or w == target_width:
        return frame.copy(), 1.0, 1.0

    scale = target_width / float(w)
    target_height = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return resized, scale, scale


def frames_to_video_tensor(frames_bgr, tracker_resize_width, device):
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
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    mx = box_w * margin_ratio
    my = box_h * margin_ratio
    xs = np.linspace(x1 + mx, x2 - mx, points_per_axis)
    ys = np.linspace(y1 + my, y2 - my, points_per_axis)
    return [[float(x), float(y)] for y in ys for x in xs]


def padded_crop_bounds(det, frame_shape, padding_ratio):
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



def serialize_track_boxes(boxes):
    serialized = []
    for box in boxes:
        x1, y1, x2, y2 = box["box"]
        serialized.append({
            "track_id": int(box["id"]),
            "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
            "conf": float(box.get("score", 0.0)),
            "source": box.get("source", "cotracker"),
            "visible_points": int(box.get("visible_points", 0)),
        })
    return serialized


def box_containment(box1, box2):
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


def filter_overlapping_track_boxes(boxes, config):
    if not boxes:
        return boxes

    def rank(box):
        source_bonus = 1.0 if box.get("source") == "pose" else 0.0
        return (source_bonus, box.get("visible_points", 0), box.get("score", 0.0))

    kept = []
    for box in sorted(boxes, key=rank, reverse=True):
        should_keep = True
        for kept_box in kept:
            iou = box_iou(box["box"], kept_box["box"])
            containment = box_containment(box["box"], kept_box["box"])
            if (
                iou >= config["track_nms_iou_threshold"]
                or containment >= config["track_nms_containment_threshold"]
            ):
                should_keep = False
                break
        if should_keep:
            kept.append(box)

    kept.sort(key=lambda x: x["id"])
    return kept


def read_overlapped_clip(cap, clip_len, clip_overlap, max_frames, next_input_frame, overlap_frames):
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
    cx1 = (box1[0] + box1[2]) * 0.5
    cy1 = (box1[1] + box1[3]) * 0.5
    cx2 = (box2[0] + box2[2]) * 0.5
    cy2 = (box2[1] + box2[3]) * 0.5
    return float(np.hypot(cx1 - cx2, cy1 - cy2))


def box_area_ratio(box1, box2):
    area1 = max(1.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(1.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    return min(area1, area2) / max(area1, area2)


def assign_track_ids(groups, active_tracks, config, next_track_id, inv_scale_x, inv_scale_y):
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
):
    seed_frame_idx = min(max(0, seed_frame_idx), len(frames) - 1)
    seed_frame = frames[seed_frame_idx]
    box_detections, _ = cubemap_sliding_detection(
        yolo_model,
        seed_frame,
        window_width_ratio=config["window_width_ratio"],
        step_size=config["step_size"],
        conf=config["conf"],
        face_size=config["face_size"],
        edge_samples=config["edge_samples"],
        nms_angle_threshold_deg=config["nms_angle_threshold_deg"],
        nms_iou_threshold=config["nms_iou_threshold"],
        nms_containment_threshold=config["nms_containment_threshold"],
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

    for frame_idx, frame in enumerate(frames):
        boxes = []
        points_by_id = {}
        tracker_shape = video.shape[-2], video.shape[-1]

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
            track_box = {
                "id": group["id"],
                "box": full_box,
                "source": group.get("source", "cotracker"),
                "visible_points": visible_count,
                "score": float(group.get("conf", 0.0)),
            }
            boxes.append(track_box)
            latest_tracks[group["id"]] = {"box": full_box}

            if config["draw_points"]:
                pts = group_points[visible].copy()
                pts[:, 0] *= inv_scale_x
                pts[:, 1] *= inv_scale_y
                points_by_id[group["id"]] = pts

        boxes = filter_overlapping_track_boxes(boxes, config)
        kept_ids = {box["id"] for box in boxes}
        points_by_id = {track_id: pts for track_id, pts in points_by_id.items() if track_id in kept_ids}
        latest_tracks = {track_id: track for track_id, track in latest_tracks.items() if track_id in kept_ids}
        annotated_frames.append(draw_tracking(frame, boxes, points_by_id if config["draw_points"] else None))
        per_frame_nums.append(len(boxes))
        per_frame_boxes.append({"boxes": serialize_track_boxes(boxes)})

    return annotated_frames, next_track_id, len(groups), latest_tracks, reused_tracks, per_frame_nums, per_frame_boxes


def process_video(config):
    if config["clip_overlap"] >= config["clip_len"]:
        raise ValueError("clip_overlap must be smaller than clip_len")

    device = get_device()
    print(f"Device: {device}")

    print("Loading YOLO model: " + config["yolo_model_path"])
    yolo_model = YOLO(config["yolo_model_path"])
    pose_model = None
    if config["use_pose_keypoints"]:
        print("Loading pose model: " + config["pose_model_path"])
        pose_model = YOLO(config["pose_model_path"])
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

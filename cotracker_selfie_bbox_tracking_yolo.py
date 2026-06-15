#!/usr/bin/env python
"""
Selfie-skier bbox tracking on 360 video with cubemap YOLO + pose/CoTracker ID support.

This is a focused variant of cotracker_person_tracking_yolo.py. It selects one
selfie skier on the seed frame, tracks only that person, and writes only that
track's bbox for each frame.
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from test_360_detection import cubemap_sliding_detection
from cotracker_person_tracking_yolo import (
    COLORS,
    CONFIG as BASE_CONFIG,
    apply_video_name_output_paths,
    assign_frame_detection_boxes,
    assign_track_ids,
    box_area_ratio,
    box_center,
    box_contains_point,
    box_center_distance,
    box_iou,
    detection_kwargs,
    detection_to_box,
    draw_tracking,
    filter_overlapping_track_boxes,
    frames_to_video_tensor,
    get_device,
    load_cotracker,
    make_query_points,
    points_to_jsonable,
    pose_detections_from_boxes,
    read_overlapped_clip,
    reconstruct_box_from_points,
    resolve_model_path,
    serialize_track_boxes,
    track_points_center_xy,
)

CONFIG = dict(BASE_CONFIG)
for key in ("output_path", "frames_output_dir", "bbox_output_path"):
    CONFIG.pop(key, None)
CONFIG.update({
    "video_path": "/mnt/dataset/skiing/360test/VID_20251006_222108_00_024.mp4",
    "output_suffix": "cotracker_selfie_yolo",
    "selfie_track_id": 1,
    "selfie_selection": "bottom_near_largest",
    "selfie_near_largest_area_ratio": 0.70,
})


def detection_area(det):
    """Return the area of a detection box with a minimum side length of one pixel."""
    return max(1.0, float(det["x2"] - det["x1"])) * max(1.0, float(det["y2"] - det["y1"]))


def select_selfie_detection(detections, frame_shape, config):
    """Choose the seed detection for the selfie skier according to the configured heuristic."""
    person_detections = [det for det in detections if det.get("class") == "person"]
    if not person_detections:
        return None

    strategy = config.get("selfie_selection", "bottom_near_largest")
    if strategy == "largest_area":
        return max(person_detections, key=lambda det: (detection_area(det), float(det.get("conf", 0.0))))
    if strategy != "bottom_near_largest":
        raise ValueError("Unsupported selfie_selection: " + str(strategy))

    # The selfie target is usually one of the largest people and closest to the
    # bottom of the equirectangular frame, not necessarily the absolute largest.
    max_area = max(detection_area(det) for det in person_detections)
    area_ratio = float(config.get("selfie_near_largest_area_ratio", 0.70))
    min_area = max_area * max(0.0, min(1.0, area_ratio))
    candidates = [det for det in person_detections if detection_area(det) >= min_area]
    return max(
        candidates,
        key=lambda det: (
            float(det["y2"]),
            (float(det["y1"]) + float(det["y2"])) * 0.5,
            detection_area(det),
            float(det.get("conf", 0.0)),
        ),
    )


def match_score(det_box, ref_box, config):
    """Score a current detection against a selfie tracker reference."""
    iou = box_iou(det_box, ref_box)
    center_dist = box_center_distance(det_box, ref_box)
    area_ratio = box_area_ratio(det_box, ref_box)
    det_center = box_center(det_box)
    ref_center = box_center(ref_box)
    center_threshold = max(1.0, float(config["id_match_center_threshold"]))
    center_match = center_dist <= center_threshold and area_ratio >= config["id_match_area_ratio_threshold"]
    contains_match = box_contains_point(det_box, ref_center) or box_contains_point(ref_box, det_center)
    if iou < config["id_match_iou_threshold"] and not center_match and not contains_match:
        return None
    contains_bonus = 0.5 if contains_match else 0.0
    return iou * 3.0 + area_ratio + contains_bonus - center_dist / center_threshold


def select_selfie_frame_box(detections, track_refs, config, track_id=1):
    """Select the current-frame bbox for the fixed selfie track ID."""
    refs = [ref for ref in track_refs if int(ref.get("id", -1)) == int(track_id)]
    if not refs:
        return None

    best = None
    for det in detections:
        if det.get("class") != "person":
            continue
        det_box = detection_to_box(det)
        for ref in refs:
            score = match_score(det_box, ref["box"], config)
            if score is None:
                continue
            if best is None or score > best[0]:
                best = (score, det, ref)

    if best is not None:
        _score, det, ref = best
        box_record = {
            "id": int(track_id),
            "box": detection_to_box(det),
            "source": "yolo_matched",
            "ref_source": ref.get("source", "cotracker"),
            "visible_points": int(ref.get("visible_points", 0)),
            "score": float(det.get("conf", 0.0)),
        }
        for key in ("track_points_xy", "track_points_source", "center_xy", "center_source"):
            if key in ref:
                box_record[key] = ref[key]
        return box_record

    fallback = refs[0]
    box_record = {
        "id": int(track_id),
        "box": tuple(int(v) for v in fallback["box"]),
        "source": "tracker_fallback",
        "visible_points": int(fallback.get("visible_points", 0)),
        "score": float(fallback.get("score", 0.0)),
    }
    for key in ("track_points_xy", "track_points_source", "center_xy", "center_source"):
        if key in fallback:
            box_record[key] = fallback[key]
    return box_record


def detect_person_boxes(yolo_model, frame, config, frame_number=None):
    """Run cubemap sliding detection and return raw person detections."""
    detections, _ = cubemap_sliding_detection(
        yolo_model,
        frame,
        **detection_kwargs(config, frame_number),
    )
    return detections


def serialize_selfie_track_boxes(boxes):
    """Serialize selfie boxes while preserving matched-reference metadata."""
    serialized = serialize_track_boxes(boxes)
    for item, box in zip(serialized, boxes):
        if "ref_source" in box:
            item["ref_source"] = box["ref_source"]
    return serialized


def process_clip_selfie(
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
    """Track only the selected selfie person through one overlapped clip."""
    selfie_track_id = int(config.get("selfie_track_id", 1))
    seed_frame_idx = min(max(0, seed_frame_idx), len(frames) - 1)
    seed_frame = frames[seed_frame_idx]
    seed_frame_number = start_frame_number + seed_frame_idx

    seed_detections = detect_person_boxes(yolo_model, seed_frame, config, seed_frame_number)
    selfie_det = select_selfie_detection(seed_detections, seed_frame.shape, config)
    if selfie_det is None:
        per_frame_nums = [0 for _ in frames]
        per_frame_boxes = [{"boxes": []} for _ in frames]
        return frames, next_track_id, 0, {}, 0, per_frame_nums, per_frame_boxes

    detections = pose_detections_from_boxes(seed_frame, [selfie_det], pose_model, config)
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
    groups[0]["id"] = selfie_track_id
    next_track_id = max(next_track_id, selfie_track_id + 1)

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
    previous_detection_ref = None
    tracker_shape = video.shape[-2], video.shape[-1]

    for frame_idx, frame in enumerate(frames):
        group = groups[0]
        group_points = tracks[frame_idx, group["start"]:group["end"]]
        group_visibility = visibility[frame_idx, group["start"]:group["end"]]
        visible = group_visibility >= config["visibility_threshold"]
        visible_count = int(visible.sum())
        points_by_id = {}
        refs = []

        tracker_box = reconstruct_box_from_points(
            group_points,
            group_visibility,
            group,
            frame_shape=(tracker_shape[0], tracker_shape[1], 3),
            visibility_threshold=config["visibility_threshold"],
            min_visible_points=config["min_visible_points"],
            padding_ratio=config["box_padding_ratio"],
        )
        if tracker_box is not None:
            x1, y1, x2, y2 = tracker_box
            full_box = (
                int(round(x1 * inv_scale_x)),
                int(round(y1 * inv_scale_y)),
                int(round(x2 * inv_scale_x)),
                int(round(y2 * inv_scale_y)),
            )
            visible_pts = group_points[visible].copy()
            if visible_count > 0:
                visible_pts[:, 0] *= inv_scale_x
                visible_pts[:, 1] *= inv_scale_y
            ref = {
                "id": selfie_track_id,
                "box": full_box,
                "source": group.get("source", "cotracker"),
                "visible_points": visible_count,
                "score": float(group.get("conf", 0.0)),
            }
            track_points = points_to_jsonable(visible_pts)
            if track_points:
                ref["track_points_xy"] = track_points
                ref["track_points_source"] = group.get("source", "cotracker")
                ref["center_xy"] = track_points_center_xy(track_points, frame.shape[1])
                ref["center_source"] = "track_points"
            refs.append(ref)
            if config["draw_points"] and visible_count > 0:
                points_by_id[selfie_track_id] = visible_pts

        if previous_detection_ref is not None:
            refs.append(previous_detection_ref)

        # Re-detect on every frame and use tracker geometry only as the identity
        # reference; the exported bbox stays tied to the current YOLO frame.
        frame_number = start_frame_number + frame_idx
        current_detections = detect_person_boxes(yolo_model, frame, config, frame_number)
        selfie_box = select_selfie_frame_box(current_detections, refs, config, track_id=selfie_track_id)
        boxes = [] if selfie_box is None else [selfie_box]
        boxes = filter_overlapping_track_boxes(boxes, config)

        if boxes:
            latest_tracks = {selfie_track_id: {"box": boxes[0]["box"]}}
            previous_detection_ref = {
                "id": selfie_track_id,
                "box": boxes[0]["box"],
                "source": boxes[0].get("source", "yolo"),
                "visible_points": boxes[0].get("visible_points", 0),
                "score": boxes[0].get("score", 0.0),
            }
            for key in ("track_points_xy", "track_points_source", "center_xy", "center_source"):
                if key in boxes[0]:
                    previous_detection_ref[key] = boxes[0][key]
        else:
            latest_tracks = {}
            previous_detection_ref = None

        annotated_frames.append(draw_tracking(frame, boxes, points_by_id if config["draw_points"] else None))
        per_frame_nums.append(len(boxes))
        per_frame_boxes.append({"boxes": serialize_selfie_track_boxes(boxes)})

    return annotated_frames, next_track_id, 1, latest_tracks, reused_tracks, per_frame_nums, per_frame_boxes


def process_video(config):
    """Run selfie-only bbox tracking for the configured video and write outputs."""
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

    out = cv2.VideoWriter(config["output_path"], cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if config["save_frames"] and config["frames_output_dir"]:
        os.makedirs(config["frames_output_dir"], exist_ok=True)

    next_input_frame = 0
    overlap_frames = []
    next_track_id = int(config.get("selfie_track_id", 1))
    active_tracks = {}
    clip_index = 0
    stats_total_output_frames = 0
    stats_frames_with_tracks = 0
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
            f"Processing selfie clip {clip_index}, frames={len(frames)}, "
            f"overlap={output_start_idx}, output_start_frame={first_output_frame}"
        )
        annotated_frames, next_track_id, num_tracks, active_tracks, reused_tracks, per_frame_nums, per_frame_boxes = process_clip_selfie(
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
        print(f"  selfie tracks seeded: {num_tracks}, reused ids: {reused_tracks}")

        for local_idx in range(output_start_idx, len(annotated_frames)):
            frame = annotated_frames[local_idx]
            frame_number = start_frame + local_idx + 1
            frame_boxes = per_frame_boxes[local_idx]["boxes"]
            if frame_boxes:
                stats_frames_with_tracks += 1
            stats_total_output_frames += 1
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
        payload = {
            "video_path": config["video_path"],
            "output_path": config["output_path"],
            "fps": float(fps),
            "width": int(width),
            "height": int(height),
            "bbox_format": "xyxy",
            "target": "selfie",
            "frames": bbox_records,
        }
        with open(bbox_output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print("Selfie bbox JSON saved to: " + bbox_output_path)

    tracking_rate = stats_frames_with_tracks / max(1, stats_total_output_frames) * 100 if stats_total_output_frames else 0.0
    print("\n" + "=" * 60)
    print(f"{'360 VIDEO SELFIE BBOX TRACKING SUMMARY':^58}")
    print("=" * 60)
    print(f"{'Frames output:':<35} {stats_total_output_frames}")
    print(f"{'Frames with selfie bbox:':<35} {stats_frames_with_tracks}")
    print(f"{'Tracking rate:':<35} {tracking_rate:.1f}%")
    print("=" * 60)
    return True


if __name__ == "__main__":
    if not os.path.exists(CONFIG["video_path"]):
        print("Error: Video file not found: " + CONFIG["video_path"])
    else:
        process_video(CONFIG)

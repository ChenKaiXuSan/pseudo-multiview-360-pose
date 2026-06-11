#!/usr/bin/env python
"""
Person tracking on 360 video with cubemap SAM3D Body detection + CoTracker point tracking.

CoTracker tracks points, not boxes. This script first scans cubemap face windows
with SAM3D Body internal person detection to get person boxes and 2D body
keypoints, tracks those points with CoTracker, then reconstructs one box per
person from the visible tracked points.
"""

import os
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sam3d_body_multiview_fusion import (  # noqa: E402
    CONFIG as SAM3D_BASE_CONFIG,
    Sam3DBodyDirectRunner,
    load_sam3d_payload,
)
os.environ["NVIDIA_VISIBLE_DEVICES"] = "cuda:1"

CONFIG = {
    # Paths
    "video_path": "/mnt/dataset/skiing/raw_new/kimura2_360.mp4",
    "output_path": "/mnt/dataset/skiing/raw_new/kimura2_360_cotracker_tracked.mp4",
    "frames_output_dir": "/mnt/dataset/skiing/raw_new/kimura2_360_cotracker_frames",
    "bbox_output_path": "/mnt/dataset/skiing/raw_new/kimura2_360_cotracker_bboxes_sam3d_body.json",

    # Models
    "cotracker_hub_repo": "facebookresearch/co-tracker",
    "cotracker_model_name": "cotracker3_offline",

    # SAM3D Body scanning. This preserves cubemap sliding-window geometry while
    # providing both bbox detection and 2D keypoint seeding.
    "sam3d_repo": SAM3D_BASE_CONFIG["sam3d_repo"],
    "sam3d_checkpoint_path": SAM3D_BASE_CONFIG["sam3d_checkpoint_path"],
    "sam3d_mhr_path": SAM3D_BASE_CONFIG["sam3d_mhr_path"],
    "sam3d_hf_repo": SAM3D_BASE_CONFIG["sam3d_hf_repo"],
    "sam3d_detector_name": "vitdet",
    "sam3d_detector_path": "",
    "sam3d_detector_bbox_thr": 0.5,
    "sam3d_detector_nms_thr": 0.3,
    "sam3d_device": "cuda:1",
    "sam3d_inference_type": "body",
    "sam3d_use_known_intrinsics": True,
    "sam3d_use_camera_translation": True,
    "sam3d_scan_cache_dir": "/mnt/dataset/skiing/raw_new/kimura2_360_cotracker_sam3d_scan",
    "sam3d_kpt_conf": 0.35,
    "min_sam3d_keypoints": 5,
    "fallback_to_grid_points": True,

    # Box detection on the first frame of each clip
    "window_width_ratio": 0.4,
    "step_size": 50,
    "face_size": 512,
    "edge_samples": 9,
    "nms_angle_threshold_deg": 5.0,
    "nms_iou_threshold": 0.6,
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


def resolve_runtime_device(config):
    requested = str(config.get("sam3d_device") or "").strip()
    if not requested:
        return get_device()

    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"sam3d_device is '{requested}', but CUDA is not available on this machine."
            )
        if requested == "cuda":
            return "cuda"
        if requested.startswith("cuda:"):
            try:
                gpu_idx = int(requested.split(":", 1)[1])
            except ValueError as exc:
                raise RuntimeError(
                    f"Invalid sam3d_device '{requested}'. Expected format like 'cuda' or 'cuda:0'."
                ) from exc

            gpu_count = torch.cuda.device_count()
            if gpu_idx < 0 or gpu_idx >= gpu_count:
                raise RuntimeError(
                    f"sam3d_device '{requested}' is out of range. Available CUDA devices: 0..{gpu_count - 1}."
                )
            return f"cuda:{gpu_idx}"

    if requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("sam3d_device is 'mps', but MPS is not available on this machine.")
        return "mps"

    if requested == "cpu":
        return "cpu"

    try:
        torch.device(requested)
    except (TypeError, RuntimeError) as exc:
        raise RuntimeError(
            f"Invalid sam3d_device '{requested}'. Use one of: cpu, mps, cuda, cuda:<index>."
        ) from exc
    return requested


def validate_sam3d_scanner_detector(config):
    detector_name = str(config.get("sam3d_detector_name") or "").strip()
    if not detector_name:
        raise RuntimeError(
            "SAM3D cubemap scanning needs a real person detector for each window. "
            "sam3d_detector_name is empty, which would make SAM3D treat every "
            "window as a full-image person crop instead of detecting bbox. "
            "Set sam3d_detector_name to 'vitdet' with detectron2 installed, or "
            "to 'sam3' with the sam3 package installed."
        )

    required_modules = {
        "vitdet": "detectron2",
        "sam3": "sam3",
    }
    module_name = required_modules.get(detector_name)
    if module_name and importlib.util.find_spec(module_name) is None:
        raise RuntimeError(
            f"SAM3D detector backend '{detector_name}' requires Python module "
            f"'{module_name}', but it is not installed in this environment. "
            "Install that detector dependency or choose another available "
            "sam3d_detector_name. Without it, SAM3D Body cannot replace bbox "
            "detection in the sliding-window scan."
        )


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


def cubemap_face_xyz(u, v, face):
    if face == 0:  # front (z+)
        return u, v, np.ones_like(u)
    if face == 1:  # right (x+)
        return np.ones_like(u), v, -u
    if face == 2:  # back (z-)
        return -u, v, -np.ones_like(u)
    if face == 3:  # left (x-)
        return -np.ones_like(u), v, u
    if face == 4:  # top (y+)
        return u, np.ones_like(v), v
    if face == 5:  # bottom (y-)
        return u, -np.ones_like(v), -v
    raise ValueError("face must be 0-5")


def xyz_to_equirectangular(x, y, z, width, height):
    lon = np.arctan2(x, z)
    lat = np.arctan2(y, np.sqrt(x**2 + z**2))
    x_e = (lon / (2 * np.pi) + 0.5) * width
    y_e = (0.5 - lat / np.pi) * height
    x_e = np.mod(x_e, width)
    y_e = np.clip(y_e, 0, height - 1)
    return lon, lat, x_e, y_e


def equirectangular_to_face(equirect_img, face_size=1024, face=0):
    h, w = equirect_img.shape[:2]
    u = (np.arange(face_size) / face_size - 0.5) * 2
    v = -(np.arange(face_size) / face_size - 0.5) * 2
    u, v = np.meshgrid(u, v)
    x, y, z = cubemap_face_xyz(u, v, face)
    _, _, map_x, map_y = xyz_to_equirectangular(x, y, z, w, h)
    return cv2.remap(
        equirect_img,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )


def project_face_points_to_equirectangular(points_face, face_id, face_size, width, height):
    points = np.asarray(points_face, dtype=np.float64)
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)
    u = 2.0 * points[:, 0] / float(face_size) - 1.0
    v = 1.0 - 2.0 * points[:, 1] / float(face_size)
    x, y, z = cubemap_face_xyz(u, v, face_id)
    lon, lat, x_e, y_e = xyz_to_equirectangular(x, y, z, width, height)
    projected = np.stack([x_e, y_e], axis=1).astype(np.float32)
    return projected, np.asarray(lon, dtype=np.float32), np.asarray(lat, dtype=np.float32)


def projected_bbox_from_face_box(x1, y1, x2, y2, face_id, face_size, width, height, edge_samples):
    xs = np.linspace(float(x1), float(x2), max(2, int(edge_samples)))
    ys = np.linspace(float(y1), float(y2), max(2, int(edge_samples)))
    sample_points = []
    sample_points.extend((x, y1) for x in xs)
    sample_points.extend((x, y2) for x in xs)
    sample_points.extend((x1, y) for y in ys)
    sample_points.extend((x2, y) for y in ys)
    projected, _, _ = project_face_points_to_equirectangular(sample_points, face_id, face_size, width, height)
    if projected.size == 0:
        return None

    x_vals = projected[:, 0].astype(np.float32)
    y_vals = projected[:, 1].astype(np.float32)
    if x_vals.max() - x_vals.min() > width * 0.5:
        x_vals = x_vals.copy()
        x_vals[x_vals < width * 0.5] += width

    x_min = int(np.clip(np.floor(x_vals.min()), 0, width - 1))
    x_max = int(np.clip(np.ceil(x_vals.max()), 0, width - 1))
    y_min = int(np.clip(np.floor(y_vals.min()), 0, height - 1))
    y_max = int(np.clip(np.ceil(y_vals.max()), 0, height - 1))
    if x_max <= x_min or y_max <= y_min:
        return None
    return x_min, y_min, x_max, y_max


def sam3d_output_bbox_xyxy(output):
    if not isinstance(output, dict):
        return None
    bbox = output.get("bbox")
    if bbox is None:
        return None
    arr = np.asarray(bbox, dtype=np.float64).reshape(-1)
    if arr.size < 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in arr[:4]]
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def sam3d_output_keypoints2d(output, min_conf):
    if not isinstance(output, dict) or "pred_keypoints_2d" not in output:
        return []
    arr = np.asarray(output["pred_keypoints_2d"], dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return []
    if arr.shape[1] == 2:
        conf = np.ones((arr.shape[0], 1), dtype=np.float64)
        arr = np.concatenate([arr[:, :2], conf], axis=1)

    keypoints = []
    for x, y, score in arr[:, :3]:
        if not np.isfinite([x, y, score]).all() or float(score) < min_conf:
            continue
        keypoints.append([float(x), float(y), float(score)])
    return keypoints


def sphere_distance(phi1, lat1, phi2, lat2):
    cos_lat1 = np.cos(lat1)
    x1 = cos_lat1 * np.cos(phi1)
    y1 = cos_lat1 * np.sin(phi1)
    z1 = np.sin(lat1)
    cos_lat2 = np.cos(lat2)
    x2 = cos_lat2 * np.cos(phi2)
    y2 = cos_lat2 * np.sin(phi2)
    z2 = np.sin(lat2)
    cross_x = y1 * z2 - z1 * y2
    cross_y = z1 * x2 - x1 * z2
    cross_z = x1 * y2 - y1 * x2
    sin_angle = np.sqrt(cross_x**2 + cross_y**2 + cross_z**2)
    cos_angle = x1 * x2 + y1 * y2 + z1 * z2
    return float(np.degrees(np.arctan2(sin_angle, np.clip(cos_angle, -1.0, 1.0))))


def detection_iou(det1, det2):
    x1 = max(det1["x1"], det2["x1"])
    y1 = max(det1["y1"], det2["y1"])
    x2 = min(det1["x2"], det2["x2"])
    y2 = min(det1["y2"], det2["y2"])
    if x1 >= x2 or y1 >= y2:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = max(0, det1["x2"] - det1["x1"]) * max(0, det1["y2"] - det1["y1"])
    area2 = max(0, det2["x2"] - det2["x1"]) * max(0, det2["y2"] - det2["y1"])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def detection_containment(det1, det2):
    x1 = max(det1["x1"], det2["x1"])
    y1 = max(det1["y1"], det2["y1"])
    x2 = min(det1["x2"], det2["x2"])
    y2 = min(det1["y2"], det2["y2"])
    if x1 >= x2 or y1 >= y2:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = max(0, det1["x2"] - det1["x1"]) * max(0, det1["y2"] - det1["y1"])
    area2 = max(0, det2["x2"] - det2["x1"]) * max(0, det2["y2"] - det2["y1"])
    min_area = min(area1, area2)
    return inter / min_area if min_area > 0 else 0.0


def non_max_suppression_sphere(detections, angle_threshold_deg=5.0, iou_threshold=0.35, containment_threshold=0.65):
    if not detections:
        return []
    kept = []
    for cls in set(d["class"] for d in detections):
        cls_dets = [d for d in detections if d["class"] == cls]
        cls_dets.sort(key=lambda x: x["conf"], reverse=True)
        while cls_dets:
            current = cls_dets.pop(0)
            kept.append(current)
            remaining = []
            for det in cls_dets:
                angle_match = False
                if angle_threshold_deg > 0 and all(k in det and k in current for k in ("_phi", "_lat")):
                    angle_match = sphere_distance(current["_phi"], current["_lat"], det["_phi"], det["_lat"]) < angle_threshold_deg
                box_match = (
                    detection_iou(current, det) > iou_threshold
                    or detection_containment(current, det) > containment_threshold
                )
                if not (angle_match or box_match):
                    remaining.append(det)
            cls_dets = remaining
    return kept


def sam3d_cubemap_sliding_detection(
    sam3d_runner,
    frame,
    cache_dir,
    frame_number,
    clip_index,
    window_width_ratio=0.4,
    step_size=50,
    face_size=512,
    edge_samples=9,
    min_keypoints=5,
    keypoint_conf=0.0,
    nms_angle_threshold_deg=5.0,
    nms_iou_threshold=0.35,
    nms_containment_threshold=0.65,
):
    h, w = frame.shape[:2]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    window_w = max(2, int(face_size * window_width_ratio))
    all_detections = []

    for face_id in range(6):
        face_img = equirectangular_to_face(frame, face_size=face_size, face=face_id)
        x_start = 0
        window_idx = 0
        while x_start + window_w <= face_size:
            window = face_img[:, x_start:x_start + window_w]
            image_path = cache_dir / f"clip_{clip_index:04d}_frame_{frame_number:06d}_face_{face_id}_win_{window_idx:03d}.jpg"
            json_path = image_path.with_suffix(".sam3d.json")
            cv2.imwrite(str(image_path), window)

            old_view = {
                "view_width": sam3d_runner.config.get("view_width"),
                "view_height": sam3d_runner.config.get("view_height"),
                "hfov_deg": sam3d_runner.config.get("hfov_deg"),
                "vfov_deg": sam3d_runner.config.get("vfov_deg"),
            }
            sam3d_runner.config["view_width"] = int(window_w)
            sam3d_runner.config["view_height"] = int(face_size)
            sam3d_runner.config["hfov_deg"] = 90.0 * float(window_w) / float(face_size)
            sam3d_runner.config["vfov_deg"] = 90.0
            try:
                sam3d_runner.run(image_path, None, json_path)
            finally:
                for key, value in old_view.items():
                    if value is None:
                        sam3d_runner.config.pop(key, None)
                    else:
                        sam3d_runner.config[key] = value

            payload = load_sam3d_payload(json_path) or {}
            outputs = payload.get("outputs", []) if isinstance(payload, dict) else []
            for output_idx, output in enumerate(outputs):
                bbox = sam3d_output_bbox_xyxy(output)
                if bbox is None:
                    continue
                x1_rel, y1_rel, x2_rel, y2_rel = bbox
                x1_face = x1_rel + x_start
                x2_face = x2_rel + x_start
                projected_box = projected_bbox_from_face_box(
                    x1_face, y1_rel, x2_face, y2_rel, face_id, face_size, w, h, edge_samples
                )
                if projected_box is None:
                    continue

                keypoints = []
                for kx, ky, score in sam3d_output_keypoints2d(output, min_conf=keypoint_conf):
                    face_point = np.array([[kx + x_start, ky]], dtype=np.float32)
                    projected, _, _ = project_face_points_to_equirectangular(face_point, face_id, face_size, w, h)
                    if projected.size == 0:
                        continue
                    px, py = projected[0]
                    keypoints.append([float(px), float(py), float(score)])

                center_face = np.array([[(x1_face + x2_face) * 0.5, (y1_rel + y2_rel) * 0.5]], dtype=np.float32)
                center_projected, phi, lat = project_face_points_to_equirectangular(center_face, face_id, face_size, w, h)
                if center_projected.size == 0:
                    continue
                conf = float(output.get("score", output.get("bbox_score", output.get("conf", 1.0))))
                source = "sam3d_body" if len(keypoints) >= min_keypoints else "sam3d_body_grid"
                all_detections.append({
                    "x1": projected_box[0],
                    "y1": projected_box[1],
                    "x2": projected_box[2],
                    "y2": projected_box[3],
                    "conf": conf,
                    "class": "person",
                    "face_id": face_id,
                    "window_index": window_idx,
                    "sam3d_output_index": int(output_idx),
                    "keypoints": keypoints,
                    "source": source,
                    "_phi": float(phi[0]),
                    "_lat": float(lat[0]),
                    "_x_c": int(np.clip(round(float(center_projected[0, 0])), 0, w - 1)),
                    "_y_c": int(np.clip(round(float(center_projected[0, 1])), 0, h - 1)),
                })
            x_start += step_size
            window_idx += 1

    detections = non_max_suppression_sphere(
        all_detections,
        angle_threshold_deg=nms_angle_threshold_deg,
        iou_threshold=nms_iou_threshold,
        containment_threshold=nms_containment_threshold,
    )
    print(f"[sam3d-scan] raw={len(all_detections)}, after_nms={len(detections)}")
    return detections, frame.copy()


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
        min_sam3d_keypoints = int(config.get("min_sam3d_keypoints", 5))
        sam3d_kpt_conf = float(config.get("sam3d_kpt_conf", 0.0))
        valid_keypoints = [kpt for kpt in keypoints if len(kpt) >= 3 and float(kpt[2]) >= sam3d_kpt_conf]
        if len(valid_keypoints) >= min_sam3d_keypoints:
            for x, y, _ in valid_keypoints:
                seed_points.append([float(x) * scale_x, float(y) * scale_y])
            source = "sam3d_body"
        elif config["fallback_to_grid_points"]:
            seed_points = grid_points_in_box(x1, y1, x2, y2, points_per_axis, margin_ratio)
            source = "sam3d_body_grid"

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
        source_bonus = 1.0 if box.get("source") == "sam3d_body" else 0.0
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
    sam3d_runner,
    cotracker,
    config,
    device,
    next_track_id,
    active_tracks,
    seed_frame_idx=0,
):
    seed_frame_idx = min(max(0, seed_frame_idx), len(frames) - 1)
    seed_frame = frames[seed_frame_idx]
    frame_number = int(config.get("_current_seed_frame_number", seed_frame_idx + 1))
    detections, _ = sam3d_cubemap_sliding_detection(
        sam3d_runner,
        seed_frame,
        cache_dir=config["sam3d_scan_cache_dir"],
        frame_number=frame_number,
        clip_index=int(config.get("_current_clip_index", 0)),
        window_width_ratio=config["window_width_ratio"],
        step_size=config["step_size"],
        face_size=config["face_size"],
        edge_samples=config["edge_samples"],
        min_keypoints=config["min_sam3d_keypoints"],
        keypoint_conf=config["sam3d_kpt_conf"],
        nms_angle_threshold_deg=config["nms_angle_threshold_deg"],
        nms_iou_threshold=config["nms_iou_threshold"],
        nms_containment_threshold=config["nms_containment_threshold"],
    )

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

    device = resolve_runtime_device(config)
    config["sam3d_device"] = device
    print(f"Runtime device (SAM3D + CoTracker): {device}")

    validate_sam3d_scanner_detector(config)
    print("Loading SAM3D Body scanner")
    sam3d_runner = Sam3DBodyDirectRunner(config)
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
        config["_current_clip_index"] = clip_index
        config["_current_seed_frame_number"] = start_frame + output_start_idx + 1

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
            sam3d_runner,
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

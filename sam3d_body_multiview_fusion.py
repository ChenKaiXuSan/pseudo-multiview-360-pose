#!/usr/bin/env python3
"""
Use tracked 360-video person bboxes to create eight perspective views, run
SAM3D Body on each view, and fuse the returned 3D keypoints in a shared world
coordinate system.

The SAM3D Body call is intentionally command-template based because local
installations often expose different entry points. Example:

    python sam3d_body_multiview_fusion.py \
      --sam3d-command 'python run_sam3d_body.py --image {image} --bbox-json {bbox_json} --output {output}'

The command must write a JSON file containing one of these keys:
    keypoints3d, keypoints_3d, joints3d, joints_3d
with values shaped as [[x, y, z], ...] or [[x, y, z, conf], ...].
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent

CONFIG = {
    "video_path": "/mnt/dataset/skiing/raw_new/kimura2_360.mp4",
    "bbox_json_path": "/mnt/dataset/skiing/raw_new/kimura2_360_cotracker_bboxes.json",
    "output_dir": "/mnt/dataset/skiing/raw_new/sam3d_body_multiview",
    "view_width": 768,
    "view_height": 768,
    "hfov_deg": 90.0,
    "vfov_deg": 90.0,
    "bbox_sample_points": 25,
    "min_projected_bbox_size": 8,
    # Eight nearby view directions around the person center. These keep the
    # person visible while changing perspective projection distortion.
    "view_offsets_deg": [
        [0.0, -16.0],
        [16.0, -11.0],
        [22.0, 0.0],
        [16.0, 11.0],
        [0.0, 16.0],
        [-16.0, 11.0],
        [-22.0, 0.0],
        [-16.0, -11.0],
    ],
    "sam_y_axis": "down",  # SAM3D camera convention: x right, y down, z forward.
    "min_kpt_conf": 0.0,
    "save_views": True,

    # Direct official SAM 3D Body API integration. Set sam3d_repo when the
    # facebookresearch/sam-3d-body repo is cloned but not pip-installed.
    "sam3d_repo": "/mnt/dataset/skiing/third_party/sam-3d-body",
    "sam3d_checkpoint_path": "/mnt/dataset/skiing/checkpoints/sam-3d-body-dinov3/model.ckpt",
    "sam3d_mhr_path": "/mnt/dataset/skiing/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt",
    "sam3d_hf_repo": "facebook/sam-3d-body-dinov3",
    "sam3d_device": "auto",
    "sam3d_inference_type": "full",
    "sam3d_use_known_intrinsics": True,
    "sam3d_use_camera_translation": True,
    # 0 means auto: run all selected perspective views concurrently.
    "sam3d_view_workers": 0,
    "view_indices": None,

    # Visualization
    "visualize_keypoints": True,
    "visualize_joint_indices": True,
    "visualize_frame_tracks": True,
}


def wrap_lon(lon: float) -> float:
    return (lon + math.pi) % (2.0 * math.pi) - math.pi


def clamp_lat(lat: float, eps: float = 1e-4) -> float:
    return float(np.clip(lat, -math.pi / 2.0 + eps, math.pi / 2.0 - eps))


def bbox_center_to_lon_lat(bbox_xyxy: list[int | float], width: int, height: int) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    lon = (cx / float(width) - 0.5) * 2.0 * math.pi
    lat = (0.5 - cy / float(height)) * math.pi
    return wrap_lon(lon), clamp_lat(lat)


def lon_lat_to_xyz(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cos_lat = np.cos(lat)
    x = cos_lat * np.sin(lon)
    y = np.sin(lat)
    z = cos_lat * np.cos(lon)
    return x, y, z


def camera_basis(yaw: float, pitch: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = np.array([
        math.cos(pitch) * math.sin(yaw),
        math.sin(pitch),
        math.cos(pitch) * math.cos(yaw),
    ], dtype=np.float64)
    forward /= max(np.linalg.norm(forward), 1e-12)

    right = np.array([math.cos(yaw), 0.0, -math.sin(yaw)], dtype=np.float64)
    right /= max(np.linalg.norm(right), 1e-12)

    up = np.cross(forward, right)
    up /= max(np.linalg.norm(up), 1e-12)
    return right, up, forward


def equirect_to_perspective(
    frame_bgr: np.ndarray,
    yaw: float,
    pitch: float,
    out_w: int,
    out_h: int,
    hfov_deg: float,
    vfov_deg: float,
) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    hfov = math.radians(hfov_deg)
    vfov = math.radians(vfov_deg)
    right, up, forward = camera_basis(yaw, pitch)

    xs = (np.arange(out_w, dtype=np.float32) + 0.5) / out_w
    ys = (np.arange(out_h, dtype=np.float32) + 0.5) / out_h
    px, py = np.meshgrid(xs, ys)
    x_cam = np.tan((px - 0.5) * hfov)
    y_cam = -np.tan((py - 0.5) * vfov)

    dirs = (
        x_cam[..., None] * right[None, None, :]
        + y_cam[..., None] * up[None, None, :]
        + forward[None, None, :]
    )
    dirs /= np.maximum(np.linalg.norm(dirs, axis=2, keepdims=True), 1e-12)

    lon = np.arctan2(dirs[..., 0], dirs[..., 2])
    lat = np.arcsin(np.clip(dirs[..., 1], -1.0, 1.0))
    map_x = ((lon / (2.0 * np.pi) + 0.5) * w).astype(np.float32)
    map_y = ((0.5 - lat / np.pi) * h).astype(np.float32)
    map_x = np.mod(map_x, w).astype(np.float32)
    map_y = np.clip(map_y, 0, h - 1).astype(np.float32)
    return cv2.remap(frame_bgr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def sample_bbox_edges(bbox_xyxy: list[int | float], samples_per_edge: int) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    n = max(2, int(samples_per_edge))
    xs = np.linspace(x1, x2, n)
    ys = np.linspace(y1, y2, n)
    top = np.stack([xs, np.full_like(xs, y1)], axis=1)
    bottom = np.stack([xs, np.full_like(xs, y2)], axis=1)
    left = np.stack([np.full_like(ys, x1), ys], axis=1)
    right = np.stack([np.full_like(ys, x2), ys], axis=1)
    return np.concatenate([top, bottom, left, right], axis=0)


def project_equirect_points_to_view(
    points_xy: np.ndarray,
    frame_w: int,
    frame_h: int,
    yaw: float,
    pitch: float,
    out_w: int,
    out_h: int,
    hfov_deg: float,
    vfov_deg: float,
) -> np.ndarray:
    hfov = math.radians(hfov_deg)
    vfov = math.radians(vfov_deg)
    right, up, forward = camera_basis(yaw, pitch)

    lon = (points_xy[:, 0] / float(frame_w) - 0.5) * 2.0 * math.pi
    lat = (0.5 - points_xy[:, 1] / float(frame_h)) * math.pi
    x, y, z = lon_lat_to_xyz(lon, lat)
    world = np.stack([x, y, z], axis=1)

    x_cam = world @ right
    y_cam = world @ up
    z_cam = world @ forward
    visible = z_cam > 1e-6
    if not np.any(visible):
        return np.empty((0, 2), dtype=np.float32)

    theta_x = np.arctan2(x_cam[visible], z_cam[visible])
    theta_y = np.arctan2(y_cam[visible], np.sqrt(x_cam[visible] ** 2 + z_cam[visible] ** 2))
    in_fov = (np.abs(theta_x) <= hfov * 0.5) & (np.abs(theta_y) <= vfov * 0.5)
    if not np.any(in_fov):
        return np.empty((0, 2), dtype=np.float32)

    u = (theta_x[in_fov] / hfov + 0.5) * out_w
    v = (0.5 - theta_y[in_fov] / vfov) * out_h
    return np.stack([u, v], axis=1).astype(np.float32)


def project_bbox_to_view(
    bbox_xyxy: list[int | float],
    frame_w: int,
    frame_h: int,
    yaw: float,
    pitch: float,
    out_w: int,
    out_h: int,
    hfov_deg: float,
    vfov_deg: float,
    sample_points: int,
    min_size: int,
) -> list[int] | None:
    edge_points = sample_bbox_edges(bbox_xyxy, sample_points)
    projected = project_equirect_points_to_view(
        edge_points, frame_w, frame_h, yaw, pitch, out_w, out_h, hfov_deg, vfov_deg
    )
    if len(projected) < 4:
        return None
    x1, y1 = projected.min(axis=0)
    x2, y2 = projected.max(axis=0)
    x1 = int(np.clip(math.floor(x1), 0, out_w - 1))
    y1 = int(np.clip(math.floor(y1), 0, out_h - 1))
    x2 = int(np.clip(math.ceil(x2), 0, out_w - 1))
    y2 = int(np.clip(math.ceil(y2), 0, out_h - 1))
    if x2 - x1 < min_size or y2 - y1 < min_size:
        return None
    return [x1, y1, x2, y2]


def read_bbox_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "frames" not in data:
        raise ValueError(f"bbox json missing 'frames': {path}")
    return data


def open_video(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    return cap


def read_video_frame(cap: cv2.VideoCapture, frame_number_1based: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_number_1based - 1))
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"failed to read frame_number={frame_number_1based}")
    return frame


def normalize_command_path(path: Path) -> str:
    return str(path.resolve())


def view_output_dir(person_dir: Path, view_idx: int) -> Path:
    return person_dir / "views" / f"view_{view_idx:02d}"


def fused_output_dir(person_dir: Path) -> Path:
    return person_dir / "fused"


def copy_if_exists(src: Path | None, dst: Path) -> None:
    if src is None or not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dst.resolve():
        return
    shutil.copy2(src, dst)


def canonicalize_view_assets(person_dir: Path, view: dict[str, Any]) -> dict[str, Any]:
    view_idx = int(view["view_index"])
    view_dir = view_output_dir(person_dir, view_idx)
    view_dir.mkdir(parents=True, exist_ok=True)

    old_image = Path(view["image_path"]) if view.get("image_path") else None
    old_vis = Path(view["vis_path"]) if view.get("vis_path") else None
    old_bbox = Path(view["bbox_json_path"]) if view.get("bbox_json_path") else None
    old_sam = Path(view["sam3d_output_path"]) if view.get("sam3d_output_path") else None

    image_path = view_dir / "frame.jpg"
    vis_path = view_dir / "frame_bbox.jpg"
    bbox_json_path = view_dir / "bbox.json"
    sam_output_path = view_dir / "sam3d.json"

    copy_if_exists(old_image, image_path)
    copy_if_exists(old_vis, vis_path)
    copy_if_exists(old_bbox, bbox_json_path)
    copy_if_exists(old_sam, sam_output_path)

    view["view_dir"] = normalize_command_path(view_dir)
    view["image_path"] = normalize_command_path(image_path)
    view["vis_path"] = normalize_command_path(vis_path)
    view["bbox_json_path"] = normalize_command_path(bbox_json_path)
    view["sam3d_output_path"] = normalize_command_path(sam_output_path)
    return view


def write_person_result(person_dir: Path, result: dict[str, Any]) -> None:
    root_path = person_dir / "fused_keypoints3d.json"
    root_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    fused_dir = fused_output_dir(person_dir)
    fused_dir.mkdir(parents=True, exist_ok=True)
    (fused_dir / "fused_keypoints3d.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


def write_view_bbox_json(path: Path, bbox_xyxy: list[int], image_path: Path, meta: dict[str, Any]) -> None:
    payload = {
        "image_path": normalize_command_path(image_path),
        "bbox_format": "xyxy",
        "bbox_xyxy": [int(v) for v in bbox_xyxy],
        **meta,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_sam3d_body_command(
    command_template: str,
    image_path: Path,
    bbox_json_path: Path,
    output_json_path: Path,
    bbox_xyxy: list[int],
) -> None:
    command = command_template.format(
        image=normalize_command_path(image_path),
        bbox_json=normalize_command_path(bbox_json_path),
        output=normalize_command_path(output_json_path),
        x1=bbox_xyxy[0],
        y1=bbox_xyxy[1],
        x2=bbox_xyxy[2],
        y2=bbox_xyxy[3],
    )
    print(f"    SAM3D Body: {command}")
    subprocess.run(command, shell=True, check=True)
    if not output_json_path.exists():
        raise FileNotFoundError(f"SAM3D command did not create output: {output_json_path}")


def extract_keypoints3d(payload: Any) -> np.ndarray | None:
    if isinstance(payload, list):
        arr = payload
    elif isinstance(payload, dict):
        arr = None
        for key in ("keypoints3d", "keypoints_3d", "joints3d", "joints_3d", "kpts3d", "kpts_3d"):
            if key in payload:
                arr = payload[key]
                break
        if arr is None and "people" in payload and payload["people"]:
            return extract_keypoints3d(payload["people"][0])
    else:
        return None

    kpts = np.asarray(arr, dtype=np.float64)
    if kpts.ndim != 2 or kpts.shape[1] < 3:
        return None
    if kpts.shape[1] == 3:
        conf = np.ones((kpts.shape[0], 1), dtype=np.float64)
        kpts = np.concatenate([kpts[:, :3], conf], axis=1)
    return kpts[:, :4]


def extract_sam3d_camera_keypoints(payload: Any, use_camera_translation: bool = True) -> np.ndarray | None:
    if (
        use_camera_translation
        and isinstance(payload, dict)
        and payload.get("outputs")
        and isinstance(payload["outputs"], list)
    ):
        first = payload["outputs"][0]
        if isinstance(first, dict) and "pred_keypoints_3d" in first:
            kpts = np.asarray(first["pred_keypoints_3d"], dtype=np.float64)
            if kpts.ndim == 2 and kpts.shape[1] >= 3:
                if "pred_cam_t" in first:
                    cam_t = np.asarray(first["pred_cam_t"], dtype=np.float64).reshape(1, 3)
                    kpts = kpts[:, :3] + cam_t
                else:
                    kpts = kpts[:, :3]
                conf = np.ones((kpts.shape[0], 1), dtype=np.float64)
                return np.concatenate([kpts, conf], axis=1)
    return extract_keypoints3d(payload)


def load_sam3d_keypoints(path: Path, use_camera_translation: bool = True) -> np.ndarray | None:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return extract_sam3d_camera_keypoints(payload, use_camera_translation=use_camera_translation)


def load_sam3d_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_keypoints2d(payload: Any) -> np.ndarray | None:
    if not isinstance(payload, dict):
        return None
    arr = payload.get("keypoints2d")
    if arr is None and payload.get("outputs") and isinstance(payload["outputs"], list):
        first = payload["outputs"][0]
        if isinstance(first, dict):
            arr = first.get("pred_keypoints_2d")
    if arr is None:
        return None
    kpts = np.asarray(arr, dtype=np.float64)
    if kpts.ndim != 2 or kpts.shape[1] < 2:
        return None
    if kpts.shape[1] == 2:
        conf = np.ones((kpts.shape[0], 1), dtype=np.float64)
        kpts = np.concatenate([kpts[:, :2], conf], axis=1)
    return kpts[:, :3]


def numpy_to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): numpy_to_jsonable(v) for k, v in value.items() if k != "mask"}
    if isinstance(value, (list, tuple)):
        return [numpy_to_jsonable(v) for v in value]
    return value


def make_camera_intrinsics(width: int, height: int, hfov_deg: float, vfov_deg: float):
    import torch

    fx = width / (2.0 * math.tan(math.radians(hfov_deg) * 0.5))
    fy = height / (2.0 * math.tan(math.radians(vfov_deg) * 0.5))
    cx = width * 0.5
    cy = height * 0.5
    return torch.tensor(
        [[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
    )


class Sam3DBodyDirectRunner:
    def __init__(self, config: dict[str, Any]):
        repo = str(config.get("sam3d_repo") or "").strip()
        if repo:
            repo_path = Path(repo).expanduser().resolve()
            if not repo_path.exists():
                raise FileNotFoundError(f"SAM3D Body repo not found: {repo_path}")
            sys.path.insert(0, str(repo_path))

        import torch
        from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body, load_sam_3d_body_hf

        requested_device = str(config.get("sam3d_device") or "auto")
        if requested_device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = requested_device

        checkpoint_path = str(config.get("sam3d_checkpoint_path") or "").strip()
        mhr_path = str(config.get("sam3d_mhr_path") or "").strip()
        hf_repo = str(config.get("sam3d_hf_repo") or "facebook/sam-3d-body-dinov3")

        if checkpoint_path:
            print(f"Loading SAM3D Body checkpoint: {checkpoint_path}")
            model, model_cfg = load_sam_3d_body(checkpoint_path, device=device, mhr_path=mhr_path)
        else:
            print(f"Loading SAM3D Body from Hugging Face: {hf_repo}")
            model, model_cfg = load_sam_3d_body_hf(hf_repo, device=device)

        self.estimator = SAM3DBodyEstimator(
            sam_3d_body_model=model,
            model_cfg=model_cfg,
            human_detector=None,
            human_segmentor=None,
            fov_estimator=None,
        )
        self.config = config

    def run(self, image_path: Path, bbox_xyxy: list[int], output_json_path: Path) -> np.ndarray | None:
        import torch

        bboxes = np.asarray([bbox_xyxy], dtype=np.float32)
        cam_int = None
        if self.config.get("sam3d_use_known_intrinsics", True):
            cam_int = make_camera_intrinsics(
                self.config["view_width"],
                self.config["view_height"],
                self.config["hfov_deg"],
                self.config["vfov_deg"],
            )

        with torch.no_grad():
            outputs = self.estimator.process_one_image(
                str(image_path),
                bboxes=bboxes,
                cam_int=cam_int,
                use_mask=False,
                inference_type=self.config.get("sam3d_inference_type", "full"),
            )

        payload = {
            "image_path": normalize_command_path(image_path),
            "bbox_format": "xyxy",
            "bbox_xyxy": [int(v) for v in bbox_xyxy],
            "outputs": numpy_to_jsonable(outputs),
        }
        if outputs:
            first = outputs[0]
            if "pred_keypoints_3d" in first:
                payload["keypoints3d"] = numpy_to_jsonable(first["pred_keypoints_3d"])
                if "pred_cam_t" in first:
                    kpts_camera = (
                        np.asarray(first["pred_keypoints_3d"], dtype=np.float64)
                        + np.asarray(first["pred_cam_t"], dtype=np.float64).reshape(1, 3)
                    )
                    payload["keypoints3d_camera"] = numpy_to_jsonable(kpts_camera)
            if "pred_keypoints_2d" in first:
                payload["keypoints2d"] = numpy_to_jsonable(first["pred_keypoints_2d"])
            if "pred_joint_coords" in first:
                payload["joint_coords"] = numpy_to_jsonable(first["pred_joint_coords"])

        output_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return extract_sam3d_camera_keypoints(
            payload,
            use_camera_translation=self.config.get("sam3d_use_camera_translation", True),
        )


def camera_keypoints_to_world(kpts_cam: np.ndarray, yaw: float, pitch: float, y_axis: str) -> np.ndarray:
    right, up, forward = camera_basis(yaw, pitch)
    xyz = kpts_cam[:, :3].copy()
    if y_axis == "down":
        xyz[:, 1] *= -1.0
    world_xyz = xyz[:, [0]] * right + xyz[:, [1]] * up + xyz[:, [2]] * forward
    return np.concatenate([world_xyz, kpts_cam[:, 3:4]], axis=1)


def fuse_keypoints_weighted(keypoints_world: list[np.ndarray], min_conf: float) -> np.ndarray | None:
    if not keypoints_world:
        return None
    n_joints = min(k.shape[0] for k in keypoints_world)
    if n_joints <= 0:
        return None

    fused = np.zeros((n_joints, 4), dtype=np.float64)
    for joint_idx in range(n_joints):
        xyz_values = []
        weights = []
        for kpts in keypoints_world:
            conf = float(kpts[joint_idx, 3])
            if conf <= min_conf:
                continue
            xyz_values.append(kpts[joint_idx, :3])
            weights.append(conf)
        if not weights:
            fused[joint_idx, :] = np.nan
            continue
        xyz_arr = np.stack(xyz_values, axis=0)
        w = np.asarray(weights, dtype=np.float64)
        fused[joint_idx, :3] = np.average(xyz_arr, axis=0, weights=w)
        fused[joint_idx, 3] = float(np.mean(w))
    return fused


def keypoints_to_jsonable(kpts: np.ndarray | None) -> list[list[float]]:
    if kpts is None:
        return []
    out = []
    for row in kpts:
        if np.any(np.isnan(row[:3])):
            out.append([None, None, None, 0.0])
        else:
            out.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
    return out


def jsonable_to_keypoints(rows: list[list[float]] | None) -> np.ndarray | None:
    if not rows:
        return None
    parsed = []
    for row in rows:
        if row is None or len(row) < 3 or row[0] is None:
            parsed.append([np.nan, np.nan, np.nan, 0.0])
            continue
        conf = float(row[3]) if len(row) > 3 and row[3] is not None else 1.0
        parsed.append([float(row[0]), float(row[1]), float(row[2]), conf])
    return np.asarray(parsed, dtype=np.float64)


def load_mhr70_visual_style(sam3d_repo: str) -> dict[str, Any]:
    repo = str(sam3d_repo or "").strip()
    if repo:
        repo_path = str(Path(repo).expanduser().resolve())
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
    try:
        from sam_3d_body.metadata.mhr70 import pose_info
    except Exception:
        return {"edges": [], "edge_colors": [], "point_colors": None}

    keypoint_info = pose_info.get("keypoint_info", {})
    name_to_id = {
        item["name"]: int(item["id"])
        for item in keypoint_info.values()
        if isinstance(item, dict) and "name" in item and "id" in item
    }

    max_id = max(name_to_id.values(), default=-1)
    point_colors = np.tile(np.array([[51, 153, 255]], dtype=np.float64), (max_id + 1, 1))
    for item in keypoint_info.values():
        if not isinstance(item, dict) or "id" not in item:
            continue
        color = item.get("color")
        if color is None:
            continue
        point_colors[int(item["id"])] = np.asarray(color, dtype=np.float64)
    if len(point_colors) == 0:
        point_colors = None
    else:
        point_colors = np.clip(point_colors / 255.0, 0.0, 1.0)

    edges = []
    edge_colors = []
    for item in pose_info.get("skeleton_info", {}).values():
        link = item.get("link") if isinstance(item, dict) else None
        if not link or len(link) != 2:
            continue
        a, b = link
        if a in name_to_id and b in name_to_id:
            edges.append((name_to_id[a], name_to_id[b]))
            color = item.get("color", [96, 96, 255])
            edge_colors.append(np.clip(np.asarray(color, dtype=np.float64) / 255.0, 0.0, 1.0))
    return {"edges": edges, "edge_colors": edge_colors, "point_colors": point_colors}


def finite_keypoint_mask(kpts: np.ndarray, min_conf: float = 0.0) -> np.ndarray:
    if kpts is None or kpts.size == 0:
        return np.zeros(0, dtype=bool)
    mask = np.isfinite(kpts[:, :3]).all(axis=1)
    if kpts.shape[1] > 3:
        mask &= kpts[:, 3] >= min_conf
    return mask


def set_axes_equal(ax, pts: np.ndarray) -> None:
    if pts.size == 0:
        return
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    centers = (mins + maxs) * 0.5
    radius = max(float((maxs - mins).max()) * 0.55, 1e-3)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def keypoints3d_to_plot_coords(kpts: np.ndarray, plot_space: str) -> tuple[np.ndarray, tuple[str, str, str], tuple[float, float]]:
    coords = np.asarray(kpts[:, :3], dtype=np.float64)
    if plot_space == "camera":
        # SAM3D camera convention follows image projection: X right, Y down, Z forward.
        # Matplotlib's vertical axis is Z, so draw -Y as vertical and keep depth on Y.
        return coords[:, [0, 2, 1]] * np.array([1.0, 1.0, -1.0]), ("camera X right", "camera Z depth", "camera -Y up"), (14, -70)
    if plot_space == "world":
        # Our fused world convention stores Y as up. Draw world Y on Matplotlib's vertical axis.
        return coords[:, [0, 2, 1]], ("world X", "world Z depth", "world Y up"), (14, -70)
    return coords.copy(), ("X", "Y", "Z"), (14, -70)


def draw_keypoints3d_axis(
    ax,
    kpts: np.ndarray,
    title: str,
    edges: list[tuple[int, int]],
    edge_colors: list[np.ndarray],
    point_colors: np.ndarray | None,
    show_indices: bool,
    min_conf: float,
    plot_space: str = "raw",
) -> None:
    mask = finite_keypoint_mask(kpts, min_conf)
    if not np.any(mask):
        ax.set_title(title + "\n(no valid kpts)")
        return

    plot_all, axis_labels, view_angles = keypoints3d_to_plot_coords(kpts, plot_space)
    pts = plot_all[mask]
    ids = np.flatnonzero(mask)
    if point_colors is not None and len(point_colors) >= len(kpts):
        colors = point_colors[ids]
    else:
        colors = np.tile(np.array([[0.2, 0.6, 1.0]]), (len(pts), 1))
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors, s=20, depthshade=True)
    for edge_idx, (a, b) in enumerate(edges):
        if a < len(kpts) and b < len(kpts) and mask[a] and mask[b]:
            seg = plot_all[[a, b], :3]
            color = edge_colors[edge_idx] if edge_idx < len(edge_colors) else np.array([0.38, 0.38, 1.0])
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=color, linewidth=1.4, alpha=0.9)
    if show_indices:
        for color_idx, (joint_id, point) in enumerate(zip(ids, pts)):
            text_color = colors[color_idx] if len(colors) else "black"
            ax.text(point[0], point[1], point[2], str(int(joint_id)), fontsize=5, color=text_color)
    ax.set_title(title)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])
    ax.view_init(elev=view_angles[0], azim=view_angles[1])
    set_axes_equal(ax, pts)
    return None


def save_keypoints3d_plot(
    kpts: np.ndarray | None,
    output_path: Path,
    title: str,
    edges: list[tuple[int, int]],
    edge_colors: list[np.ndarray],
    point_colors: np.ndarray | None,
    show_indices: bool,
    min_conf: float,
    plot_space: str = "raw",
) -> str | None:
    if kpts is None:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    draw_keypoints3d_axis(ax, kpts, title, edges, edge_colors, point_colors, show_indices, min_conf, plot_space)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return normalize_command_path(output_path)


def log_progress(prefix: str, message: str, verbose: bool = True) -> None:
    if verbose:
        print(f"[{prefix}] {message}", flush=True)


def track_color_rgb01(track_id: int) -> np.ndarray:
    palette = np.array([
        [0.90, 0.18, 0.20],
        [0.16, 0.55, 0.95],
        [0.20, 0.70, 0.32],
        [0.88, 0.48, 0.12],
        [0.58, 0.32, 0.86],
        [0.08, 0.66, 0.70],
        [0.86, 0.22, 0.58],
        [0.55, 0.55, 0.12],
    ], dtype=np.float64)
    stable_index = (int(track_id) * 2654435761) % len(palette)
    return palette[stable_index]


def track_id_from_dir(track_dir: Path) -> int:
    try:
        return int(track_dir.name.split("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def collect_frame_world_tracks(frame_dir: Path, min_conf: float = 0.0, verbose: bool = False) -> list[dict[str, Any]]:
    tracks = []
    track_dirs = sorted(frame_dir.glob("track_*"), key=track_id_from_dir)
    if verbose:
        log_progress("frame-vis", f"scan tracks: {frame_dir} ({len(track_dirs)} candidates)")
    for track_pos, track_dir in enumerate(track_dirs, start=1):
        result_path = track_dir / "fused_keypoints3d.json"
        if not result_path.exists():
            result_path = track_dir / "fused" / "fused_keypoints3d.json"
        if not result_path.exists():
            if verbose:
                log_progress("frame-vis", f"[{track_pos}/{len(track_dirs)}] skip {track_dir.name}: missing fused_keypoints3d.json")
            continue
        with result_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        arr = payload.get("fused_keypoints3d_world")
        if arr is None:
            if verbose:
                log_progress("frame-vis", f"[{track_pos}/{len(track_dirs)}] skip {track_dir.name}: no fused_keypoints3d_world")
            continue
        kpts = np.asarray(arr, dtype=np.float64)
        if kpts.ndim != 2 or kpts.shape[1] < 3:
            if verbose:
                log_progress("frame-vis", f"[{track_pos}/{len(track_dirs)}] skip {track_dir.name}: invalid keypoint shape {kpts.shape}")
            continue
        if kpts.shape[1] == 3:
            conf = np.ones((kpts.shape[0], 1), dtype=np.float64)
            kpts = np.concatenate([kpts[:, :3], conf], axis=1)
        else:
            kpts = kpts[:, :4]
        if not np.any(finite_keypoint_mask(kpts, min_conf)):
            if verbose:
                log_progress("frame-vis", f"[{track_pos}/{len(track_dirs)}] skip {track_dir.name}: no valid keypoints above min_conf={min_conf}")
            continue
        track_id = int(payload.get("track_id", track_id_from_dir(track_dir)))
        tracks.append({
            "track_id": track_id,
            "keypoints": kpts,
            "result_path": normalize_command_path(result_path),
        })
        if verbose:
            valid_count = int(finite_keypoint_mask(kpts, min_conf).sum())
            log_progress("frame-vis", f"[{track_pos}/{len(track_dirs)}] loaded track {track_id:04d}: {valid_count}/{len(kpts)} valid joints")
    if verbose:
        log_progress("frame-vis", f"collected {len(tracks)} valid tracks")
    return tracks


def draw_frame_tracks_world_axis(
    ax,
    tracks: list[dict[str, Any]],
    edges: list[tuple[int, int]],
    show_indices: bool,
    min_conf: float,
) -> None:
    if not tracks:
        ax.set_title("world 3D tracks\n(no valid tracks)")
        return

    all_pts = []
    axis_labels = ("world X", "world Z depth", "world Y up")
    view_angles = (14, -70)
    for color_idx, track in enumerate(tracks):
        kpts = np.asarray(track["keypoints"], dtype=np.float64)
        mask = finite_keypoint_mask(kpts, min_conf)
        if not np.any(mask):
            continue
        plot_all, axis_labels, view_angles = keypoints3d_to_plot_coords(kpts, "world")
        pts = plot_all[mask]
        ids = np.flatnonzero(mask)
        track_id = int(track["track_id"])
        color = track_color_rgb01(track_id)
        label = f"track {track_id:04d}"
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=[color], s=24, depthshade=True, label=label)
        for a, b in edges:
            if a < len(kpts) and b < len(kpts) and mask[a] and mask[b]:
                seg = plot_all[[a, b], :3]
                ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=color, linewidth=1.5, alpha=0.88)
        label_point = pts[np.argmin(pts[:, 2])]
        ax.text(label_point[0], label_point[1], label_point[2], label, fontsize=8, color=color)
        if show_indices:
            for joint_id, point in zip(ids, pts):
                ax.text(point[0], point[1], point[2], str(int(joint_id)), fontsize=5, color=color)
        all_pts.append(pts)

    ax.set_title("all tracks in shared world 3D")
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])
    ax.view_init(elev=view_angles[0], azim=view_angles[1])
    if all_pts:
        set_axes_equal(ax, np.concatenate(all_pts, axis=0))
        ax.legend(loc="upper left", fontsize=8)


def draw_frame_tracks_topdown_axis(
    ax,
    tracks: list[dict[str, Any]],
    edges: list[tuple[int, int]],
    min_conf: float,
) -> None:
    if not tracks:
        ax.set_title("world XZ top-down\n(no valid tracks)")
        return

    all_pts = []
    for track in tracks:
        kpts = np.asarray(track["keypoints"], dtype=np.float64)
        mask = finite_keypoint_mask(kpts, min_conf)
        if not np.any(mask):
            continue
        pts = kpts[mask, :3]
        track_id = int(track["track_id"])
        color = track_color_rgb01(track_id)
        label = f"track {track_id:04d}"
        ax.scatter(pts[:, 0], pts[:, 2], c=[color], s=22, label=label)
        for a, b in edges:
            if a < len(kpts) and b < len(kpts) and mask[a] and mask[b]:
                seg = kpts[[a, b], :3]
                ax.plot(seg[:, 0], seg[:, 2], color=color, linewidth=1.4, alpha=0.86)
        label_point = pts[np.argmin(pts[:, 1])]
        ax.text(label_point[0], label_point[2], label, fontsize=8, color=color)
        all_pts.append(pts[:, [0, 2]])

    ax.set_title("world XZ top-down")
    ax.set_xlabel("world X")
    ax.set_ylabel("world Z depth")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    if all_pts:
        pts2d = np.concatenate(all_pts, axis=0)
        mins = pts2d.min(axis=0)
        maxs = pts2d.max(axis=0)
        centers = (mins + maxs) * 0.5
        radius = max(float((maxs - mins).max()) * 0.58, 1e-3)
        ax.set_xlim(centers[0] - radius, centers[0] + radius)
        ax.set_ylim(centers[1] - radius, centers[1] + radius)
        ax.legend(loc="upper left", fontsize=8)


def write_frame_tracks_metadata(
    output_path: Path,
    tracks: list[dict[str, Any]],
    frame_title: str | None,
    min_conf: float,
) -> str:
    metadata_path = output_path.with_suffix(".json")
    metadata = {
        "image_path": normalize_command_path(output_path),
        "frame_title": frame_title or "original 360 frame",
        "tracks": [int(track["track_id"]) for track in tracks],
        "source_files": [track.get("result_path", "") for track in tracks],
        "min_conf": float(min_conf),
        "plot_views": ["original_360", "world_3d", "world_xz_topdown"],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return normalize_command_path(metadata_path)


def save_frame_tracks_world_visualization(
    frame_bgr: np.ndarray,
    tracks: list[dict[str, Any]],
    output_path: Path,
    edges: list[tuple[int, int]],
    edge_colors: list[np.ndarray],
    point_colors: np.ndarray | None,
    show_indices: bool,
    min_conf: float,
    frame_title: str | None = None,
    verbose: bool = False,
) -> str | None:
    if frame_bgr is None or not tracks:
        if verbose:
            reason = "missing frame" if frame_bgr is None else "no valid tracks"
            log_progress("frame-vis", f"skip combined world view: {reason}")
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if verbose:
        log_progress("frame-vis", f"render combined world view: tracks={len(tracks)}, output={output_path}")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize=(16, 12))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25], hspace=0.16, wspace=0.10)
    image_ax = fig.add_subplot(grid[0, :])
    image_ax.imshow(frame_rgb)
    image_ax.set_title(frame_title or "original 360 frame")
    image_ax.axis("off")

    world_ax = fig.add_subplot(grid[1, 0], projection="3d")
    draw_frame_tracks_world_axis(world_ax, tracks, edges, show_indices, min_conf)
    topdown_ax = fig.add_subplot(grid[1, 1])
    draw_frame_tracks_topdown_axis(topdown_ax, tracks, edges, min_conf)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    metadata_path = write_frame_tracks_metadata(output_path, tracks, frame_title, min_conf)
    plt.close(fig)
    if verbose:
        log_progress("frame-vis", f"saved combined world view: {output_path}")
        log_progress("frame-vis", f"saved metadata: {metadata_path}")
    return normalize_command_path(output_path)


def visualize_existing_frame_output(
    output_dir: Path,
    frame_number: int,
    frame_bgr: np.ndarray,
    config: dict[str, Any],
    verbose: bool = False,
) -> dict[str, Any]:
    frame_dir = output_dir / f"frame_{frame_number:06d}"
    if verbose:
        log_progress("frame-vis", f"start frame {frame_number:06d}: {frame_dir}")
    style = load_mhr70_visual_style(config.get("sam3d_repo", ""))
    tracks = collect_frame_world_tracks(frame_dir, float(config.get("min_kpt_conf", 0.0)), verbose=verbose)
    output_path = frame_dir / "frame_tracks_world.png"
    vis_path = save_frame_tracks_world_visualization(
        frame_bgr,
        tracks,
        output_path,
        style["edges"],
        style["edge_colors"],
        style["point_colors"],
        bool(config.get("visualize_joint_indices", True)),
        float(config.get("min_kpt_conf", 0.0)),
        frame_title=f"frame {frame_number:06d} original 360",
        verbose=verbose,
    )
    return {
        "frame_number": int(frame_number),
        "num_tracks": len(tracks),
        "frame_tracks_world_vis_path": vis_path,
    }


def rgb01_to_bgr255(color: np.ndarray | list[float]) -> tuple[int, int, int]:
    arr = np.asarray(color, dtype=np.float64)
    if arr.max(initial=0.0) <= 1.0:
        arr = arr * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return int(arr[2]), int(arr[1]), int(arr[0])


def save_keypoints2d_overlay(
    image_path: Path,
    kpts2d: np.ndarray | None,
    output_path: Path,
    bbox_xyxy: list[int] | None,
    edges: list[tuple[int, int]],
    edge_colors: list[np.ndarray],
    point_colors: np.ndarray | None,
    show_indices: bool,
    min_conf: float,
) -> str | None:
    if kpts2d is None or not image_path.exists():
        return None
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    h, w = image.shape[:2]
    if bbox_xyxy is not None:
        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 255), 2)

    mask = np.isfinite(kpts2d[:, :2]).all(axis=1)
    if kpts2d.shape[1] > 2:
        mask &= kpts2d[:, 2] >= min_conf

    for edge_idx, (a, b) in enumerate(edges):
        if a >= len(kpts2d) or b >= len(kpts2d) or not mask[a] or not mask[b]:
            continue
        p1 = tuple(np.round(kpts2d[a, :2]).astype(int))
        p2 = tuple(np.round(kpts2d[b, :2]).astype(int))
        if not (0 <= p1[0] < w and 0 <= p1[1] < h and 0 <= p2[0] < w and 0 <= p2[1] < h):
            continue
        color = edge_colors[edge_idx] if edge_idx < len(edge_colors) else np.array([0.38, 0.38, 1.0])
        cv2.line(image, p1, p2, rgb01_to_bgr255(color), 2, lineType=cv2.LINE_AA)

    for joint_idx, point in enumerate(kpts2d[:, :2]):
        if not mask[joint_idx]:
            continue
        x, y = np.round(point).astype(int)
        if not (0 <= x < w and 0 <= y < h):
            continue
        if point_colors is not None and joint_idx < len(point_colors):
            color = rgb01_to_bgr255(point_colors[joint_idx])
        else:
            color = (255, 153, 51)
        cv2.circle(image, (x, y), 4, color, -1, lineType=cv2.LINE_AA)
        if show_indices:
            cv2.putText(image, str(joint_idx), (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    return normalize_command_path(output_path)


def save_view_frames_grid(
    views: list[dict[str, Any]],
    output_path: Path,
    image_key: str = "image_path",
    tile_width: int = 384,
    cols: int = 4,
) -> str | None:
    tiles = []
    for view in sorted(views, key=lambda item: int(item.get("view_index", 0))):
        image_path = Path(view.get(image_key, ""))
        if not image_path.exists():
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        scale = tile_width / max(float(w), 1.0)
        tile_height = max(1, int(round(h * scale)))
        image = cv2.resize(image, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
        label_h = 34
        labeled = np.full((tile_height + label_h, tile_width, 3), 255, dtype=np.uint8)
        labeled[label_h:, :, :] = image
        label = (
            f"view {int(view['view_index']):02d}  "
            f"yaw {float(view.get('yaw_offset_deg', 0.0)):+.0f}  "
            f"pitch {float(view.get('pitch_offset_deg', 0.0)):+.0f}"
        )
        cv2.putText(labeled, label, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
        tiles.append(labeled)

    if not tiles:
        return None

    cols = max(1, min(cols, len(tiles)))
    rows = int(math.ceil(len(tiles) / cols))
    tile_h = max(tile.shape[0] for tile in tiles)
    tile_w = max(tile.shape[1] for tile in tiles)
    grid = np.full((rows * tile_h, cols * tile_w, 3), 245, dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        row, col = divmod(idx, cols)
        y0 = row * tile_h
        x0 = col * tile_w
        grid[y0:y0 + tile.shape[0], x0:x0 + tile.shape[1], :] = tile

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), grid)
    return normalize_command_path(output_path)


def save_multiview_keypoints_grid(
    per_view: list[tuple[int, np.ndarray]],
    output_path: Path,
    edges: list[tuple[int, int]],
    edge_colors: list[np.ndarray],
    point_colors: np.ndarray | None,
    show_indices: bool,
    min_conf: float,
    plot_space: str = "raw",
) -> str | None:
    if not per_view:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(18, 9))
    for plot_idx, (view_idx, kpts) in enumerate(per_view[:8], start=1):
        ax = fig.add_subplot(2, 4, plot_idx, projection="3d")
        draw_keypoints3d_axis(
            ax,
            kpts,
            f"view {view_idx:02d} camera 3D",
            edges,
            edge_colors,
            point_colors,
            show_indices,
            min_conf,
            plot_space,
        )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return normalize_command_path(output_path)


def visualize_person_keypoints(
    person_dir: Path,
    views: list[dict[str, Any]],
    fused: np.ndarray | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    canonical_views = [canonicalize_view_assets(person_dir, view) for view in views]
    views_dir = person_dir / "views"
    views_dir.mkdir(parents=True, exist_ok=True)
    vis = {
        "views": [],
        "views_frames_grid_path": save_view_frames_grid(canonical_views, views_dir / "views_frames_grid.jpg", "image_path"),
        "views_frame_bboxes_grid_path": save_view_frames_grid(canonical_views, views_dir / "views_frame_bboxes_grid.jpg", "vis_path"),
    }
    if not config.get("visualize_keypoints", True):
        return vis
    style = load_mhr70_visual_style(config.get("sam3d_repo", ""))
    edges = style["edges"]
    edge_colors = style["edge_colors"]
    point_colors = style["point_colors"]
    show_indices = bool(config.get("visualize_joint_indices", True))
    min_conf = float(config.get("min_kpt_conf", 0.0))
    per_view_root_camera = []
    per_view_translated_camera = []

    for view in canonical_views:
        sam_output_path = Path(view["sam3d_output_path"])
        if not sam_output_path.exists():
            continue
        sam_payload = load_sam3d_payload(sam_output_path)
        kpts2d = extract_keypoints2d(sam_payload) if sam_payload is not None else None
        kpts_root_cam = load_sam3d_keypoints(sam_output_path, use_camera_translation=False)
        kpts_cam = load_sam3d_keypoints(
            sam_output_path,
            use_camera_translation=config.get("sam3d_use_camera_translation", True),
        )
        if kpts_cam is None:
            continue
        view_idx = int(view["view_index"])
        view_dir = Path(view["view_dir"])
        kpts2d_vis = save_keypoints2d_overlay(
            Path(view["image_path"]),
            kpts2d,
            view_dir / "frame_kpts2d.jpg",
            view.get("bbox_xyxy"),
            edges,
            edge_colors,
            point_colors,
            show_indices,
            min_conf,
        )
        root_camera_vis = None
        if kpts_root_cam is not None:
            root_camera_path = view_dir / "kpts3d_root_camera.png"
            root_camera_vis = save_keypoints3d_plot(
                kpts_root_cam,
                root_camera_path,
                f"frame {view['frame_number']} track {view['track_id']} view {view_idx:02d} root camera 3D",
                edges,
                edge_colors,
                point_colors,
                show_indices,
                min_conf,
                plot_space="camera",
            )
        camera_path = view_dir / "kpts3d_camera.png"
        camera_vis = save_keypoints3d_plot(
            kpts_cam,
            camera_path,
            f"frame {view['frame_number']} track {view['track_id']} view {view_idx:02d} translated camera 3D",
            edges,
            edge_colors,
            point_colors,
            show_indices,
            min_conf,
            plot_space="camera",
        )
        kpts_world = camera_keypoints_to_world(
            kpts_cam,
            math.radians(view["yaw_deg"]),
            math.radians(view["pitch_deg"]),
            config["sam_y_axis"],
        )
        world_path = view_dir / "kpts3d_world.png"
        world_vis = save_keypoints3d_plot(
            kpts_world,
            world_path,
            f"frame {view['frame_number']} track {view['track_id']} view {view_idx:02d} world 3D",
            edges,
            edge_colors,
            point_colors,
            show_indices,
            min_conf,
            plot_space="world",
        )
        view["kpts2d_vis_path"] = kpts2d_vis
        view["kpts3d_root_camera_vis_path"] = root_camera_vis
        view["kpts3d_camera_vis_path"] = camera_vis
        view["kpts3d_world_vis_path"] = world_vis
        vis["views"].append({
            "view_index": view_idx,
            "kpts2d_vis_path": kpts2d_vis,
            "root_camera_vis_path": root_camera_vis,
            "camera_vis_path": camera_vis,
            "world_vis_path": world_vis,
        })
        if kpts_root_cam is not None:
            per_view_root_camera.append((view_idx, kpts_root_cam))
        per_view_translated_camera.append((view_idx, kpts_cam))

    root_grid_path = views_dir / "views_kpts3d_root_camera_grid.png"
    vis["views_root_camera_grid_path"] = save_multiview_keypoints_grid(
        per_view_root_camera,
        root_grid_path,
        edges,
        edge_colors,
        point_colors,
        show_indices,
        min_conf,
        plot_space="camera",
    )
    camera_grid_path = views_dir / "views_kpts3d_camera_grid.png"
    vis["views_camera_grid_path"] = save_multiview_keypoints_grid(
        per_view_translated_camera,
        camera_grid_path,
        edges,
        edge_colors,
        point_colors,
        show_indices,
        min_conf,
        plot_space="camera",
    )
    fused_dir = fused_output_dir(person_dir)
    fused_dir.mkdir(parents=True, exist_ok=True)
    fused_path = fused_dir / "fused_kpts3d_world.png"
    vis["fused_world_vis_path"] = save_keypoints3d_plot(
        fused,
        fused_path,
        "fused world 3D keypoints",
        edges,
        edge_colors,
        point_colors,
        show_indices,
        min_conf,
        plot_space="world",
    )
    return vis


def visualize_existing_person_output(person_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    fused_path = person_dir / "fused_keypoints3d.json"
    if not fused_path.exists():
        raise FileNotFoundError(f"missing fused output: {fused_path}")
    with fused_path.open("r", encoding="utf-8") as f:
        result = json.load(f)
    views = result.get("views", [])
    keypoints_world = []
    for view in views:
        sam_output_path = Path(view["sam3d_output_path"])
        if not sam_output_path.exists():
            continue
        kpts_cam = load_sam3d_keypoints(
            sam_output_path,
            use_camera_translation=config.get("sam3d_use_camera_translation", True),
        )
        if kpts_cam is None:
            continue
        keypoints_world.append(camera_keypoints_to_world(
            kpts_cam,
            math.radians(view["yaw_deg"]),
            math.radians(view["pitch_deg"]),
            config["sam_y_axis"],
        ))
    fused = fuse_keypoints_weighted(keypoints_world, config["min_kpt_conf"])
    result["fused_keypoints3d_world"] = keypoints_to_jsonable(fused)
    result["num_fused_views"] = len(keypoints_world)
    vis = visualize_person_keypoints(person_dir, views, fused, config)
    result["visualization"] = vis
    write_person_result(person_dir, result)
    return vis


def build_person_views(
    frame: np.ndarray,
    source_bbox: list[int | float],
    frame_number: int,
    track_id: int,
    person_dir: Path,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    frame_h, frame_w = frame.shape[:2]
    center_yaw, center_pitch = bbox_center_to_lon_lat(source_bbox, frame_w, frame_h)
    view_records = []

    selected_view_indices = config.get("view_indices")
    if selected_view_indices is not None:
        selected_view_indices = set(int(v) for v in selected_view_indices)

    for view_idx, (yaw_offset_deg, pitch_offset_deg) in enumerate(config["view_offsets_deg"]):
        if selected_view_indices is not None and view_idx not in selected_view_indices:
            continue
        yaw = wrap_lon(center_yaw + math.radians(float(yaw_offset_deg)))
        pitch = clamp_lat(center_pitch + math.radians(float(pitch_offset_deg)))
        image = equirect_to_perspective(
            frame,
            yaw,
            pitch,
            config["view_width"],
            config["view_height"],
            config["hfov_deg"],
            config["vfov_deg"],
        )
        view_bbox = project_bbox_to_view(
            source_bbox,
            frame_w,
            frame_h,
            yaw,
            pitch,
            config["view_width"],
            config["view_height"],
            config["hfov_deg"],
            config["vfov_deg"],
            config["bbox_sample_points"],
            config["min_projected_bbox_size"],
        )
        if view_bbox is None:
            print(f"    skip view {view_idx}: source bbox is outside the perspective view")
            continue

        view_dir = view_output_dir(person_dir, view_idx)
        view_dir.mkdir(parents=True, exist_ok=True)
        image_path = view_dir / "frame.jpg"
        vis_path = view_dir / "frame_bbox.jpg"
        bbox_json_path = view_dir / "bbox.json"
        sam_output_path = view_dir / "sam3d.json"
        cv2.imwrite(str(image_path), image)
        if config["save_views"]:
            vis = image.copy()
            x1, y1, x2, y2 = view_bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 255), 2)
            cv2.imwrite(str(vis_path), vis)

        meta = {
            "frame_number": int(frame_number),
            "track_id": int(track_id),
            "view_index": int(view_idx),
            "center_yaw_deg": float(math.degrees(center_yaw)),
            "center_pitch_deg": float(math.degrees(center_pitch)),
            "yaw_deg": float(math.degrees(yaw)),
            "pitch_deg": float(math.degrees(pitch)),
            "yaw_offset_deg": float(yaw_offset_deg),
            "pitch_offset_deg": float(pitch_offset_deg),
        }
        write_view_bbox_json(bbox_json_path, view_bbox, image_path, meta)
        view_records.append({
            **meta,
            "view_dir": normalize_command_path(view_dir),
            "image_path": normalize_command_path(image_path),
            "vis_path": normalize_command_path(vis_path),
            "bbox_json_path": normalize_command_path(bbox_json_path),
            "bbox_xyxy": view_bbox,
            "sam3d_output_path": normalize_command_path(sam_output_path),
        })
    return view_records


def run_sam3d_for_view(
    view: dict[str, Any],
    config: dict[str, Any],
    sam3d_command: str | None,
    sam3d_runner: Sam3DBodyDirectRunner | None,
) -> np.ndarray | None:
    image_path = Path(view["image_path"])
    bbox_json_path = Path(view["bbox_json_path"])
    sam_output_path = Path(view["sam3d_output_path"])
    view_idx = int(view["view_index"])
    if sam3d_runner is not None:
        print(f"    SAM3D Body direct API: view {view_idx}")
        return sam3d_runner.run(image_path, view["bbox_xyxy"], sam_output_path)
    if sam3d_command:
        run_sam3d_body_command(
            sam3d_command,
            image_path,
            bbox_json_path,
            sam_output_path,
            view["bbox_xyxy"],
        )
        return load_sam3d_keypoints(
            sam_output_path,
            use_camera_translation=config.get("sam3d_use_camera_translation", True),
        )
    return None


def run_sam3d_for_views(
    views: list[dict[str, Any]],
    config: dict[str, Any],
    sam3d_command: str | None,
    sam3d_runner: Sam3DBodyDirectRunner | None,
) -> list[tuple[dict[str, Any], np.ndarray | None]]:
    if not sam3d_runner and not sam3d_command:
        return [(view, None) for view in views]
    configured_workers = int(config.get("sam3d_view_workers", 0))
    workers = len(views) if configured_workers <= 0 else configured_workers
    workers = max(1, workers)
    if workers <= 1 or len(views) <= 1:
        return [
            (view, run_sam3d_for_view(view, config, sam3d_command, sam3d_runner))
            for view in views
        ]

    workers = min(workers, len(views))
    print(f"    running SAM3D Body on {len(views)} views with {workers} workers")
    results_by_index: dict[int, tuple[dict[str, Any], np.ndarray | None]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_sam3d_for_view, view, config, sam3d_command, sam3d_runner): idx
            for idx, view in enumerate(views)
        }
        for future in as_completed(futures):
            idx = futures[future]
            results_by_index[idx] = (views[idx], future.result())
    return [results_by_index[idx] for idx in range(len(views))]


def process_person(
    frame: np.ndarray,
    frame_number: int,
    box: dict[str, Any],
    output_dir: Path,
    config: dict[str, Any],
    sam3d_command: str | None,
    sam3d_runner: Sam3DBodyDirectRunner | None = None,
) -> dict[str, Any]:
    track_id = int(box.get("track_id", box.get("id", -1)))
    source_bbox = box.get("bbox_xyxy") or box.get("box") or box.get("bbox")
    if not source_bbox or len(source_bbox) != 4:
        raise ValueError(f"invalid bbox for frame={frame_number}, track_id={track_id}: {box}")

    person_dir = output_dir / f"frame_{frame_number:06d}" / f"track_{track_id:04d}"
    person_dir.mkdir(parents=True, exist_ok=True)
    print(f"  frame={frame_number} track={track_id} bbox={source_bbox}")

    views = build_person_views(frame, source_bbox, frame_number, track_id, person_dir, config)
    keypoints_world = []
    for view, kpts_cam in run_sam3d_for_views(views, config, sam3d_command, sam3d_runner):
        if kpts_cam is None:
            print(f"    skip view {view['view_index']}: no parseable 3D keypoints")
            continue
        kpts_world = camera_keypoints_to_world(
            kpts_cam,
            math.radians(view["yaw_deg"]),
            math.radians(view["pitch_deg"]),
            config["sam_y_axis"],
        )
        keypoints_world.append(kpts_world)

    fused = fuse_keypoints_weighted(keypoints_world, config["min_kpt_conf"])
    visualization = visualize_person_keypoints(person_dir, views, fused, config)
    result = {
        "frame_number": int(frame_number),
        "track_id": int(track_id),
        "source_bbox_xyxy": [int(v) for v in source_bbox],
        "views": views,
        "fused_keypoints3d_world": keypoints_to_jsonable(fused),
        "num_views": len(views),
        "num_fused_views": len(keypoints_world),
        "visualization": visualization,
    }
    write_person_result(person_dir, result)
    return result


def select_frame_records(
    bbox_data: dict[str, Any],
    frame_number: int | None,
    track_id: int | None,
    max_frames: int | None,
) -> list[dict[str, Any]]:
    selected = []
    for record in bbox_data["frames"]:
        if frame_number is not None and int(record["frame_number"]) != frame_number:
            continue
        boxes = record.get("boxes", [])
        if track_id is not None:
            boxes = [b for b in boxes if int(b.get("track_id", b.get("id", -1))) == track_id]
        if not boxes:
            continue
        selected.append({**record, "boxes": boxes})
        if max_frames is not None and len(selected) >= max_frames:
            break
    return selected


def parse_view_indices(value: str | None) -> list[int] | None:
    if value is None or not value.strip():
        return None
    indices = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        idx = int(part)
        if idx < 0 or idx >= len(CONFIG["view_offsets_deg"]):
            raise ValueError(f"view index out of range: {idx}")
        indices.append(idx)
    return sorted(set(indices))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="360 bbox -> 8 perspective views -> SAM3D Body -> fused 3D kpts")
    parser.add_argument("--video", default=CONFIG["video_path"], help="360 equirectangular video path")
    parser.add_argument("--bbox-json", default=CONFIG["bbox_json_path"], help="bbox JSON from cotracker_person_tracking.py")
    parser.add_argument("--output-dir", default=CONFIG["output_dir"], help="output directory")
    parser.add_argument("--frame-number", type=int, default=None, help="only process this 1-based frame number")
    parser.add_argument("--track-id", type=int, default=None, help="only process this track id")
    parser.add_argument("--max-frames", type=int, default=1, help="maximum bbox frames to process; set 0 for all")
    parser.add_argument("--run-sam3d", action="store_true", help="run SAM3D Body direct API using --sam3d-hf-repo or local checkpoint")
    parser.add_argument("--sam3d-command", default=None, help="fallback command template for SAM3D Body")
    parser.add_argument("--sam3d-repo", default=CONFIG["sam3d_repo"], help="path to facebookresearch/sam-3d-body repo")
    parser.add_argument("--sam3d-checkpoint", default=CONFIG["sam3d_checkpoint_path"], help="local SAM3D Body model.ckpt path")
    parser.add_argument("--sam3d-mhr", default=CONFIG["sam3d_mhr_path"], help="local MHR model asset path")
    parser.add_argument("--sam3d-hf-repo", default=CONFIG["sam3d_hf_repo"], help="HF repo used when no local checkpoint is supplied")
    parser.add_argument("--sam3d-device", default=CONFIG["sam3d_device"], help="auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--sam3d-inference-type", default=CONFIG["sam3d_inference_type"], choices=["full", "body", "hand"])
    parser.add_argument("--sam3d-view-workers", type=int, default=CONFIG["sam3d_view_workers"], help="number of perspective views to run concurrently; 0 means all selected views")
    parser.add_argument("--view-indices", default=None, help="comma-separated view indices to run, e.g. 0,2,4,6; default runs all views")
    parser.add_argument("--no-known-intrinsics", action="store_true", help="let SAM3D Body use its default/FOV estimator intrinsics")
    parser.add_argument("--view-size", type=int, default=CONFIG["view_width"], help="square perspective view size")
    parser.add_argument("--hfov", type=float, default=CONFIG["hfov_deg"], help="horizontal FOV in degrees")
    parser.add_argument("--vfov", type=float, default=CONFIG["vfov_deg"], help="vertical FOV in degrees")
    parser.add_argument("--sam-y-axis", choices=["up", "down"], default=CONFIG["sam_y_axis"])
    parser.add_argument("--no-sam-cam-translation", action="store_true", help="use root-relative pred_keypoints_3d instead of pred_keypoints_3d + pred_cam_t")
    parser.add_argument("--min-kpt-conf", type=float, default=CONFIG["min_kpt_conf"])
    parser.add_argument("--no-save-views", action="store_true", help="do not draw bbox on saved view images")
    parser.add_argument("--no-kpt-vis", action="store_true", help="skip 3D keypoint visualization PNG outputs")
    parser.add_argument("--no-joint-indices", action="store_true", help="hide joint index labels in 3D plots")
    parser.add_argument("--visualize-existing", action="store_true", help="visualize existing SAM3D/fused outputs without running video/SAM3D")
    parser.add_argument("--visualize-frame-existing", action="store_true", help="visualize all existing tracks for one frame in shared world coordinates")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = dict(CONFIG)
    config.update({
        "view_width": int(args.view_size),
        "view_height": int(args.view_size),
        "hfov_deg": float(args.hfov),
        "vfov_deg": float(args.vfov),
        "sam_y_axis": args.sam_y_axis,
        "min_kpt_conf": float(args.min_kpt_conf),
        "save_views": not args.no_save_views,
        "sam3d_repo": args.sam3d_repo,
        "sam3d_checkpoint_path": args.sam3d_checkpoint,
        "sam3d_mhr_path": args.sam3d_mhr,
        "sam3d_hf_repo": args.sam3d_hf_repo,
        "sam3d_device": args.sam3d_device,
        "sam3d_inference_type": args.sam3d_inference_type,
        "sam3d_view_workers": max(0, int(args.sam3d_view_workers)),
        "view_indices": parse_view_indices(args.view_indices),
        "sam3d_use_known_intrinsics": not args.no_known_intrinsics,
        "sam3d_use_camera_translation": not args.no_sam_cam_translation,
        "visualize_keypoints": not args.no_kpt_vis,
        "visualize_joint_indices": not args.no_joint_indices,
        "visualize_frame_tracks": CONFIG["visualize_frame_tracks"],
    })

    bbox_json_path = Path(args.bbox_json)
    video_path = Path(args.video)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.visualize_existing:
        if args.frame_number is None or args.track_id is None:
            raise ValueError("--visualize-existing requires --frame-number and --track-id")
        person_dir = output_dir / f"frame_{args.frame_number:06d}" / f"track_{args.track_id:04d}"
        vis = visualize_existing_person_output(person_dir, config)
        print(f"Saved visualization for existing output: {person_dir}")
        if vis.get("views_frames_grid_path"):
            print("  view frames grid: " + vis["views_frames_grid_path"])
        if vis.get("views_frame_bboxes_grid_path"):
            print("  view bbox frames grid: " + vis["views_frame_bboxes_grid_path"])
        if vis.get("views_root_camera_grid_path"):
            print("  root views grid: " + vis["views_root_camera_grid_path"])
        if vis.get("views_camera_grid_path"):
            print("  views grid: " + vis["views_camera_grid_path"])
        if vis.get("fused_world_vis_path"):
            print("  fused: " + vis["fused_world_vis_path"])
        return 0

    if args.visualize_frame_existing:
        if args.frame_number is None:
            raise ValueError("--visualize-frame-existing requires --frame-number")
        cap = open_video(video_path)
        try:
            frame = read_video_frame(cap, args.frame_number)
        finally:
            cap.release()
        log_progress("main", f"read frame {args.frame_number:06d} from {video_path}")
        vis = visualize_existing_frame_output(output_dir, args.frame_number, frame, config, verbose=True)
        print(f"Saved frame visualization for existing output: frame_{args.frame_number:06d}")
        if vis.get("frame_tracks_world_vis_path"):
            print("  frame tracks world: " + vis["frame_tracks_world_vis_path"])
        print(f"  tracks: {vis['num_tracks']}")
        return 0

    bbox_data = read_bbox_json(bbox_json_path)
    frame_records = select_frame_records(
        bbox_data,
        args.frame_number,
        args.track_id,
        None if args.max_frames == 0 else args.max_frames,
    )
    if not frame_records:
        print("No bbox records matched the requested frame/track filters.")
        return 1

    sam3d_runner = None
    direct_sam3d_requested = bool(args.run_sam3d)
    if direct_sam3d_requested:
        sam3d_runner = Sam3DBodyDirectRunner(config)
    elif args.sam3d_command is None:
        print("No SAM3D Body runner supplied; saving perspective views and projected bboxes only.")

    cap = open_video(video_path)
    results = []
    frame_visualizations = []
    total_frames = len(frame_records)
    total_tracks = sum(len(record.get("boxes", [])) for record in frame_records)
    log_progress("main", f"processing {total_frames} frame(s), {total_tracks} track box(es)")
    processed_tracks = 0
    try:
        for frame_pos, frame_record in enumerate(frame_records, start=1):
            frame_number = int(frame_record["frame_number"])
            boxes = frame_record.get("boxes", [])
            log_progress("main", f"frame {frame_pos}/{total_frames}: read video frame {frame_number:06d}; tracks={len(boxes)}")
            frame = read_video_frame(cap, frame_number)
            for track_pos, box in enumerate(boxes, start=1):
                processed_tracks += 1
                track_id = int(box.get("track_id", box.get("id", -1)))
                log_progress("main", f"frame {frame_pos}/{total_frames}, track {track_pos}/{len(boxes)}, total {processed_tracks}/{total_tracks}: start track_{track_id:04d}")
                results.append(process_person(
                    frame,
                    frame_number,
                    box,
                    output_dir,
                    config,
                    args.sam3d_command,
                    sam3d_runner,
                ))
            if config.get("visualize_keypoints", True) and config.get("visualize_frame_tracks", True):
                log_progress("main", f"frame {frame_pos}/{total_frames}: build combined world visualization")
                frame_vis = visualize_existing_frame_output(output_dir, frame_number, frame, config, verbose=True)
                frame_visualizations.append(frame_vis)
                if frame_vis.get("frame_tracks_world_vis_path"):
                    print("  frame tracks world: " + frame_vis["frame_tracks_world_vis_path"])
    finally:
        cap.release()

    summary = {
        "video_path": str(video_path),
        "bbox_json_path": str(bbox_json_path),
        "coordinate_system": "world: x=right/east at yaw=0, y=up, z=front at yaw=0; lon=atan2(x,z)",
        "sam3d_command": args.sam3d_command,
        "sam3d_direct_api": sam3d_runner is not None,
        "sam3d_view_workers": config["sam3d_view_workers"],
        "view_indices": config["view_indices"],
        "sam3d_hf_repo": args.sam3d_hf_repo,
        "sam3d_checkpoint": args.sam3d_checkpoint,
        "results": results,
        "frame_visualizations": frame_visualizations,
    }
    summary_path = output_dir / "multiview_fused_keypoints3d.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

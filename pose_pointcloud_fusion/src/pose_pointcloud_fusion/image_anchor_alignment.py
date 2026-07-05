from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


FRAME_RE = re.compile(r"frame_(\d+)")


@dataclass(frozen=True)
class ColmapObservation2D:
    xy: np.ndarray
    point3d_id: int


@dataclass(frozen=True)
class ColmapImageObservations:
    image_id: int
    image_name: str
    frame_index: int | None
    observations: list[ColmapObservation2D]


@dataclass(frozen=True)
class ColmapPoint3D:
    point3d_id: int
    xyz: np.ndarray
    rgb: tuple[int, int, int]
    error: float


@dataclass(frozen=True)
class SimilarityTransform:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    rmse: float


@dataclass(frozen=True)
class JointAnchorMatch:
    joint_index: int
    point3d_id: int
    joint_xy: np.ndarray
    observation_xy: np.ndarray
    source_xyz: np.ndarray
    target_xyz: np.ndarray
    pixel_distance: float


@dataclass(frozen=True)
class ImageAnchorAlignmentResult:
    aligned_keypoints: np.ndarray
    transform: SimilarityTransform
    matches: list[JointAnchorMatch]

    @property
    def num_matches(self) -> int:
        return len(self.matches)


def _frame_index_from_name(name: str) -> int | None:
    match = FRAME_RE.search(name)
    if match is None:
        return None
    return int(match.group(1))


def parse_colmap_images_observations(path: Path) -> dict[int, ColmapImageObservations]:
    """Parse COLMAP images.txt and keep x/y/POINT3D_ID observations."""
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    images: dict[int, ColmapImageObservations] = {}
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        idx += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        try:
            image_id = int(parts[0])
        except ValueError:
            continue
        image_name = parts[-1]
        points_line = ""
        if idx < len(lines):
            points_line = lines[idx]
            idx += 1
        point_parts = points_line.split()
        observations: list[ColmapObservation2D] = []
        for pos in range(0, len(point_parts) - 2, 3):
            try:
                point3d_id = int(float(point_parts[pos + 2]))
            except ValueError:
                continue
            if point3d_id < 0:
                continue
            observations.append(
                ColmapObservation2D(
                    xy=np.asarray([float(point_parts[pos]), float(point_parts[pos + 1])], dtype=np.float64),
                    point3d_id=point3d_id,
                )
            )
        images[image_id] = ColmapImageObservations(
            image_id=image_id,
            image_name=image_name,
            frame_index=_frame_index_from_name(image_name),
            observations=observations,
        )
    return images


def parse_colmap_points3d_by_id(path: Path) -> dict[int, ColmapPoint3D]:
    """Parse COLMAP points3D.txt into a POINT3D_ID keyed map."""
    points: dict[int, ColmapPoint3D] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        try:
            point3d_id = int(parts[0])
            xyz = np.asarray([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
            rgb = (int(parts[4]), int(parts[5]), int(parts[6]))
            error = float(parts[7])
        except ValueError:
            continue
        points[point3d_id] = ColmapPoint3D(point3d_id=point3d_id, xyz=xyz, rgb=rgb, error=error)
    return points


def _transform_xyz_by_matrix(xyz: np.ndarray, matrix: Any, *, scale: float = 1.0) -> np.ndarray:
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError("camera_to_world must be a 4x4 matrix")
    vec = np.asarray([xyz[0] * scale, xyz[1] * scale, xyz[2] * scale, 1.0], dtype=np.float64)
    out = mat @ vec
    return out[:3]


def _apply_alignment_xyz(xyz: np.ndarray, alignment: dict[str, Any] | None) -> np.ndarray:
    if not alignment:
        return np.asarray(xyz, dtype=np.float64)
    scale = float(alignment.get("scale", 1.0))
    rotation = np.asarray(alignment.get("rotation", np.eye(3)), dtype=np.float64)
    translation = np.asarray(alignment.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64)
    return scale * (rotation @ np.asarray(xyz, dtype=np.float64)) + translation


def _load_manifest_view(manifest_path: Path, view_name: str) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for view in manifest.get("views", []):
        if str(view.get("name")) == str(view_name):
            return view
    raise ValueError(f"view '{view_name}' not found in manifest: {manifest_path}")


def load_view_points3d_by_id(
    points3d_txt: Path,
    *,
    manifest_path: Path | None = None,
    view_name: str | None = None,
    alignment_json: Path | None = None,
    rough_scale: float = 1.0,
) -> dict[int, ColmapPoint3D]:
    """Load COLMAP points and optionally transform them into merged/refined world coordinates."""
    points = parse_colmap_points3d_by_id(points3d_txt)
    if manifest_path is None and alignment_json is None:
        return points
    if view_name is None:
        raise ValueError("view_name is required when manifest_path or alignment_json is provided")
    camera_to_world = None
    if manifest_path is not None:
        view = _load_manifest_view(manifest_path, view_name)
        camera_to_world = view.get("camera_to_world")
        if camera_to_world is None:
            raise ValueError(f"view '{view_name}' has no camera_to_world in {manifest_path}")
    alignment = None
    if alignment_json is not None:
        alignment_payload = json.loads(alignment_json.read_text(encoding="utf-8"))
        alignment = alignment_payload.get("transforms", {}).get(str(view_name), {})
    transformed: dict[int, ColmapPoint3D] = {}
    for point_id, point in points.items():
        xyz = point.xyz
        if camera_to_world is not None:
            xyz = _transform_xyz_by_matrix(xyz, camera_to_world, scale=rough_scale)
        xyz = _apply_alignment_xyz(xyz, alignment)
        transformed[point_id] = ColmapPoint3D(
            point3d_id=point.point3d_id,
            xyz=xyz,
            rgb=point.rgb,
            error=point.error,
        )
    return transformed


def _ensure_keypoints2d(arr: Any) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64)
    if out.ndim != 2 or out.shape[1] < 2:
        raise ValueError("keypoints2d must be an Nx2 or Nx3 array")
    if out.shape[1] == 2:
        out = np.concatenate([out, np.ones((out.shape[0], 1), dtype=np.float64)], axis=1)
    return out[:, :3]


def _ensure_keypoints3d(arr: Any) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64)
    if out.ndim != 2 or out.shape[1] < 3:
        raise ValueError("keypoints3d must be an Nx3 or Nx4 array")
    if out.shape[1] == 3:
        out = np.concatenate([out, np.ones((out.shape[0], 1), dtype=np.float64)], axis=1)
    return out[:, :4]


def load_sam3d_view_keypoints(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load SAM3D 2D keypoints and camera/root 3D keypoints from one view payload."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    kpts2d = payload.get("keypoints2d")
    kpts3d = payload.get("keypoints3d_camera") or payload.get("keypoints3d")
    if (kpts2d is None or kpts3d is None) and payload.get("outputs") and isinstance(payload["outputs"], list):
        first = payload["outputs"][0]
        if kpts2d is None:
            kpts2d = first.get("pred_keypoints_2d")
        if kpts3d is None:
            kpts3d = first.get("pred_keypoints_3d")
    if kpts2d is None:
        raise ValueError(f"SAM3D payload has no keypoints2d: {path}")
    if kpts3d is None:
        raise ValueError(f"SAM3D payload has no keypoints3d/keypoints3d_camera: {path}")
    return _ensure_keypoints2d(kpts2d), _ensure_keypoints3d(kpts3d), payload


def match_keypoints2d_to_colmap_points(
    *,
    keypoints2d: np.ndarray,
    keypoints3d: np.ndarray,
    image_observations: ColmapImageObservations,
    points3d_by_id: dict[int, ColmapPoint3D],
    radius_px: float = 8.0,
    min_conf: float = 0.3,
    max_point_error: float | None = None,
) -> list[JointAnchorMatch]:
    """Match each valid SAM3D 2D joint to its nearest COLMAP 2D point observation."""
    kpts2d = _ensure_keypoints2d(keypoints2d)
    kpts3d = _ensure_keypoints3d(keypoints3d)
    n = min(len(kpts2d), len(kpts3d))
    matches: list[JointAnchorMatch] = []
    used_point_ids: set[int] = set()
    for joint_idx in range(n):
        if not np.isfinite(kpts2d[joint_idx, :2]).all() or not np.isfinite(kpts3d[joint_idx, :3]).all():
            continue
        if float(kpts2d[joint_idx, 2]) < float(min_conf) or float(kpts3d[joint_idx, 3]) < float(min_conf):
            continue
        best: tuple[float, ColmapObservation2D, ColmapPoint3D] | None = None
        for obs in image_observations.observations:
            if obs.point3d_id in used_point_ids:
                continue
            point = points3d_by_id.get(obs.point3d_id)
            if point is None:
                continue
            if max_point_error is not None and point.error > max_point_error:
                continue
            distance = float(np.linalg.norm(kpts2d[joint_idx, :2] - obs.xy))
            if distance > float(radius_px):
                continue
            if best is None or distance < best[0]:
                best = (distance, obs, point)
        if best is None:
            continue
        distance, obs, point = best
        used_point_ids.add(obs.point3d_id)
        matches.append(
            JointAnchorMatch(
                joint_index=joint_idx,
                point3d_id=obs.point3d_id,
                joint_xy=kpts2d[joint_idx, :2].copy(),
                observation_xy=obs.xy.copy(),
                source_xyz=kpts3d[joint_idx, :3].copy(),
                target_xyz=point.xyz.copy(),
                pixel_distance=distance,
            )
        )
    return matches


def estimate_similarity_transform(source_xyz: np.ndarray, target_xyz: np.ndarray, *, allow_scaling: bool = True) -> SimilarityTransform:
    """Estimate target ~= scale * rotation @ source + translation with Umeyama alignment."""
    source = np.asarray(source_xyz, dtype=np.float64)
    target = np.asarray(target_xyz, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source_xyz and target_xyz must both be Nx3 arrays")
    if source.shape[0] < 3:
        raise ValueError("at least 3 matched joints are required to estimate a 3D similarity transform")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    source_var = float(np.mean(np.sum(source_centered * source_centered, axis=1)))
    if source_var <= 1e-12:
        raise ValueError("source keypoints are degenerate; cannot estimate transform")
    covariance = (target_centered.T @ source_centered) / float(source.shape[0])
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vt) < 0.0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt
    if allow_scaling:
        scale = float(np.sum(singular_values * np.diag(correction)) / source_var)
    else:
        scale = 1.0
    translation = target_mean - scale * (rotation @ source_mean)
    aligned = (scale * (rotation @ source.T)).T + translation
    rmse = float(math.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))))
    return SimilarityTransform(scale=scale, rotation=rotation, translation=translation, rmse=rmse)


def apply_similarity_transform(keypoints3d: np.ndarray, transform: SimilarityTransform) -> np.ndarray:
    """Apply a similarity transform to Nx3/Nx4 keypoints while preserving confidence."""
    kpts = _ensure_keypoints3d(keypoints3d)
    aligned_xyz = (transform.scale * (transform.rotation @ kpts[:, :3].T)).T + transform.translation
    return np.concatenate([aligned_xyz, kpts[:, 3:4]], axis=1)


def align_sam3d_keypoints_to_colmap(
    *,
    keypoints2d: np.ndarray,
    keypoints3d: np.ndarray,
    image_observations: ColmapImageObservations,
    points3d_by_id: dict[int, ColmapPoint3D],
    radius_px: float = 8.0,
    min_conf: float = 0.3,
    max_point_error: float | None = None,
    allow_scaling: bool = True,
) -> ImageAnchorAlignmentResult:
    """Anchor a SAM3D 3D skeleton to COLMAP points through same-image 2D observations."""
    matches = match_keypoints2d_to_colmap_points(
        keypoints2d=keypoints2d,
        keypoints3d=keypoints3d,
        image_observations=image_observations,
        points3d_by_id=points3d_by_id,
        radius_px=radius_px,
        min_conf=min_conf,
        max_point_error=max_point_error,
    )
    if len(matches) < 3:
        raise ValueError(f"at least 3 image-anchor matches are required; got {len(matches)}")
    source = np.asarray([match.source_xyz for match in matches], dtype=np.float64)
    target = np.asarray([match.target_xyz for match in matches], dtype=np.float64)
    transform = estimate_similarity_transform(source, target, allow_scaling=allow_scaling)
    aligned = apply_similarity_transform(keypoints3d, transform)
    return ImageAnchorAlignmentResult(aligned_keypoints=aligned, transform=transform, matches=matches)


def _jsonable_array(arr: np.ndarray) -> list[Any]:
    return np.asarray(arr, dtype=np.float64).tolist()


def _match_to_json(match: JointAnchorMatch) -> dict[str, Any]:
    return {
        "joint_index": int(match.joint_index),
        "point3d_id": int(match.point3d_id),
        "joint_xy": _jsonable_array(match.joint_xy),
        "observation_xy": _jsonable_array(match.observation_xy),
        "source_xyz": _jsonable_array(match.source_xyz),
        "target_xyz": _jsonable_array(match.target_xyz),
        "pixel_distance": float(match.pixel_distance),
    }


def _transform_to_json(transform: SimilarityTransform) -> dict[str, Any]:
    return {
        "scale": float(transform.scale),
        "rotation": _jsonable_array(transform.rotation),
        "translation": _jsonable_array(transform.translation),
        "rmse": float(transform.rmse),
    }


def write_image_anchor_alignment(
    *,
    sam3d_json: Path,
    images_txt: Path,
    points3d_txt: Path,
    output_json: Path,
    image_id: int | None = None,
    frame_index: int | None = None,
    manifest_path: Path | None = None,
    view_name: str | None = None,
    alignment_json: Path | None = None,
    rough_scale: float = 1.0,
    radius_px: float = 8.0,
    min_conf: float = 0.3,
    max_point_error: float | None = None,
    allow_scaling: bool = True,
) -> dict[str, Any]:
    """Write transformed SAM3D keypoints anchored to COLMAP/VIPE points through image observations."""
    keypoints2d, keypoints3d, payload = load_sam3d_view_keypoints(sam3d_json)
    images = parse_colmap_images_observations(images_txt)
    points = load_view_points3d_by_id(
        points3d_txt,
        manifest_path=manifest_path,
        view_name=view_name,
        alignment_json=alignment_json,
        rough_scale=rough_scale,
    )
    if image_id is not None:
        image_observations = images[int(image_id)]
    else:
        if frame_index is None:
            frame_index = payload.get("frame_number")
        candidates = [image for image in images.values() if image.frame_index == frame_index]
        if not candidates:
            raise ValueError(f"no COLMAP image observations found for frame_index={frame_index}")
        image_observations = candidates[0]
    result = align_sam3d_keypoints_to_colmap(
        keypoints2d=keypoints2d,
        keypoints3d=keypoints3d,
        image_observations=image_observations,
        points3d_by_id=points,
        radius_px=radius_px,
        min_conf=min_conf,
        max_point_error=max_point_error,
        allow_scaling=allow_scaling,
    )
    output = {
        "sam3d_json": str(sam3d_json),
        "images_txt": str(images_txt),
        "points3d_txt": str(points3d_txt),
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "view_name": view_name,
        "alignment_json": str(alignment_json) if alignment_json is not None else None,
        "rough_scale": float(rough_scale),
        "image_id": int(image_observations.image_id),
        "image_name": image_observations.image_name,
        "frame_index": image_observations.frame_index,
        "frame_number": payload.get("frame_number"),
        "track_id": payload.get("track_id"),
        "radius_px": float(radius_px),
        "min_conf": float(min_conf),
        "max_point_error": max_point_error,
        "num_matches": int(result.num_matches),
        "transform": _transform_to_json(result.transform),
        "matches": [_match_to_json(match) for match in result.matches],
        "aligned_keypoints3d_world": _jsonable_array(result.aligned_keypoints),
        "notes": [
            "SAM3D 3D keypoints remain the skeleton source.",
            "COLMAP/VIPE 2D observations provide point-cloud anchors used to estimate one similarity transform.",
            "Matched point-cloud points are not used to replace individual joints.",
        ],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output

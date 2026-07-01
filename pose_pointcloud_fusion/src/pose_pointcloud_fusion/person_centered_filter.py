from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .pose_pointcloud_overlay import DEFAULT_BODY_EDGES, FusedKeypoints, load_fused_keypoints


@dataclass(frozen=True)
class FilteredPointCloud:
    xyz: np.ndarray
    rgb: np.ndarray
    summary: dict[str, Any]


def read_ascii_xyzrgb_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertex_count: int | None = None
    header_lines = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            header_lines += 1
            stripped = line.strip()
            if stripped.startswith("element vertex "):
                vertex_count = int(stripped.split()[-1])
            if stripped == "end_header":
                break
    if vertex_count is None:
        raise ValueError(f"PLY header has no vertex count: {path}")

    xyz = np.zeros((vertex_count, 3), dtype=np.float64)
    rgb = np.zeros((vertex_count, 3), dtype=np.uint8)
    used = 0
    with path.open("r", encoding="utf-8") as handle:
        for _ in range(header_lines):
            next(handle)
        for line in handle:
            if used >= vertex_count:
                break
            parts = line.split()
            if len(parts) < 3:
                continue
            xyz[used] = [float(parts[0]), float(parts[1]), float(parts[2])]
            if len(parts) >= 6:
                rgb[used] = [int(float(parts[3])), int(float(parts[4])), int(float(parts[5]))]
            else:
                rgb[used] = [255, 255, 255]
            used += 1
    return xyz[:used], rgb[:used]


def write_ascii_xyzrgb_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    pts = np.asarray(xyz, dtype=np.float64)
    colors = np.asarray(rgb, dtype=np.uint8)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("xyz must have shape (N, 3)")
    if colors.shape != pts.shape:
        raise ValueError("rgb must have shape (N, 3)")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(pts)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    for point, color in zip(pts, colors):
        lines.append(
            f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {int(color[0])} {int(color[1])} {int(color[2])}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_frame_tracks_keypoints(path: Path, track_id: int | None, min_conf: float) -> FusedKeypoints:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tracks = payload.get("tracks", [])
    source_files = payload.get("source_files", [])
    if track_id is None and len(source_files) == 1:
        return load_fused_keypoints(Path(source_files[0]), min_conf=min_conf)
    for idx, current_track in enumerate(tracks):
        if int(current_track) == int(track_id):
            if idx >= len(source_files):
                raise ValueError(f"track {track_id} has no source file in {path}")
            source_path = Path(source_files[idx])
            if not source_path.is_absolute():
                source_path = path.parent / source_path
            return load_fused_keypoints(source_path, min_conf=min_conf)
    raise ValueError(f"track {track_id} not found in {path}")


def load_person_keypoints(path: Path, track_id: int | None = None, min_conf: float = 0.3) -> FusedKeypoints:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "fused_keypoints3d_world" in payload:
        return load_fused_keypoints(path, min_conf=min_conf)
    if "tracks" in payload and "source_files" in payload:
        return _load_frame_tracks_keypoints(path, track_id, min_conf)
    raise ValueError(f"unsupported pose JSON format: {path}")


load_person_keypoints.from_rows = load_fused_keypoints.from_rows  # type: ignore[attr-defined]


def valid_keypoint_xyz(keypoints: FusedKeypoints) -> np.ndarray:
    if keypoints.values.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(keypoints.values[keypoints.mask_valid, :3], dtype=np.float64)


def pose_center(keypoints: FusedKeypoints) -> np.ndarray | None:
    pts = valid_keypoint_xyz(keypoints)
    if len(pts) == 0:
        return None
    return np.median(pts, axis=0)


def build_pose_segments(
    poses: list[FusedKeypoints],
    edges: list[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    edges = DEFAULT_BODY_EDGES if edges is None else edges
    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    for pose in poses:
        values = pose.values
        mask = pose.mask_valid
        for a, b in edges:
            if a < len(values) and b < len(values) and mask[a] and mask[b]:
                starts.append(values[a, :3].astype(np.float64))
                ends.append(values[b, :3].astype(np.float64))
    if not starts:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)
    return np.asarray(starts, dtype=np.float64), np.asarray(ends, dtype=np.float64)


def point_to_segments_distance(
    points: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    chunk_size: int = 32768,
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    seg_a = np.asarray(starts, dtype=np.float64)
    seg_b = np.asarray(ends, dtype=np.float64)
    if len(pts) == 0:
        return np.zeros(0, dtype=np.float64)
    if len(seg_a) == 0:
        return np.full(len(pts), np.inf, dtype=np.float64)
    seg = seg_b - seg_a
    denom = np.sum(seg * seg, axis=1)
    denom = np.where(denom <= 1e-12, 1.0, denom)
    out = np.empty(len(pts), dtype=np.float64)
    for start in range(0, len(pts), chunk_size):
        chunk = pts[start:start + chunk_size]
        rel = chunk[:, None, :] - seg_a[None, :, :]
        t = np.sum(rel * seg[None, :, :], axis=2) / denom[None, :]
        t = np.clip(t, 0.0, 1.0)
        closest = seg_a[None, :, :] + t[:, :, None] * seg[None, :, :]
        diff = chunk[:, None, :] - closest
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        out[start:start + len(chunk)] = np.min(dist, axis=1)
    return out


def statistical_outlier_mask(points: np.ndarray, k: int = 20, std_ratio: float = 2.0) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n == 0:
        return np.zeros(0, dtype=bool)
    if n <= max(2, int(k)):
        return np.ones(n, dtype=bool)
    neighbors = min(int(k) + 1, n)
    try:
        from scipy.spatial import cKDTree  # type: ignore

        distances, _ = cKDTree(pts).query(pts, k=neighbors)
        if distances.ndim == 1:
            mean_dist = distances
        else:
            mean_dist = distances[:, 1:].mean(axis=1)
    except Exception:
        if n > 50000:
            # Avoid an accidental O(N^2) fallback on full production clouds.
            return np.ones(n, dtype=bool)
        diff = pts[:, None, :] - pts[None, :, :]
        sq = np.sum(diff * diff, axis=2)
        order = np.partition(sq, kth=neighbors - 1, axis=1)[:, 1:neighbors]
        mean_dist = np.sqrt(order).mean(axis=1)
    threshold = float(mean_dist.mean() + float(std_ratio) * mean_dist.std())
    return mean_dist <= threshold



def voxel_density_outlier_mask(points: np.ndarray, voxel_size: float = 0.1, min_neighbors: int = 4) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n == 0:
        return np.zeros(0, dtype=bool)
    size = max(float(voxel_size), 1e-9)
    min_count = max(1, int(min_neighbors))
    origin = np.min(pts, axis=0)
    voxels = np.floor((pts - origin) / size).astype(np.int64)
    unique, inverse, counts = np.unique(voxels, axis=0, return_inverse=True, return_counts=True)
    count_by_voxel = {tuple(int(v) for v in voxel): int(count) for voxel, count in zip(unique, counts)}
    neighbor_counts = np.zeros(len(unique), dtype=np.int64)
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    for idx, voxel in enumerate(unique):
        base = tuple(int(v) for v in voxel)
        total = 0
        for dx, dy, dz in offsets:
            total += count_by_voxel.get((base[0] + dx, base[1] + dy, base[2] + dz), 0)
        neighbor_counts[idx] = total
    return neighbor_counts[inverse] >= min_count


def _trajectory_roi_mask(
    xyz: np.ndarray,
    centers: np.ndarray,
    *,
    trajectory_radius: float,
    height_below: float,
    height_above: float,
) -> np.ndarray:
    if len(centers) == 0:
        return np.ones(len(xyz), dtype=bool)
    horizontal = xyz[:, None, [0, 2]] - centers[None, :, [0, 2]]
    horizontal_dist = np.sqrt(np.sum(horizontal * horizontal, axis=2))
    near_path = np.min(horizontal_dist, axis=1) <= float(trajectory_radius)
    y_min = float(np.min(centers[:, 1]) - float(height_below))
    y_max = float(np.max(centers[:, 1]) + float(height_above))
    return near_path & (xyz[:, 1] >= y_min) & (xyz[:, 1] <= y_max)


def filter_person_centered_points(
    xyz: np.ndarray,
    rgb: np.ndarray,
    poses: list[FusedKeypoints],
    *,
    mode: str = "scene",
    trajectory_radius: float = 8.0,
    height_below: float = 2.0,
    height_above: float = 3.0,
    body_radius: float = 0.35,
    outlier_filter: str = "statistical",
    outlier_k: int = 20,
    outlier_std_ratio: float = 2.0,
    voxel_size: float = 0.1,
    voxel_min_neighbors: int = 4,
    edges: list[tuple[int, int]] | None = None,
) -> FilteredPointCloud:
    pts = np.asarray(xyz, dtype=np.float64)
    colors = np.asarray(rgb, dtype=np.uint8)
    if pts.shape != colors.shape or pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("xyz and rgb must both have shape (N, 3)")
    centers = [center for center in (pose_center(pose) for pose in poses) if center is not None]
    center_arr = np.asarray(centers, dtype=np.float64) if centers else np.zeros((0, 3), dtype=np.float64)

    roi_mask = _trajectory_roi_mask(
        pts,
        center_arr,
        trajectory_radius=trajectory_radius,
        height_below=height_below,
        height_above=height_above,
    )
    starts, ends = build_pose_segments(poses, edges=edges)
    if len(starts):
        body_distance = point_to_segments_distance(pts, starts, ends)
        if mode == "scene":
            body_mask = body_distance > float(body_radius)
        elif mode == "human":
            body_mask = body_distance <= float(body_radius)
        else:
            raise ValueError("mode must be 'scene' or 'human'")
    else:
        body_distance = np.full(len(pts), np.inf, dtype=np.float64)
        body_mask = np.ones(len(pts), dtype=bool) if mode == "scene" else np.zeros(len(pts), dtype=bool)

    pre_outlier_mask = roi_mask & body_mask
    filtered_xyz = pts[pre_outlier_mask]
    filtered_rgb = colors[pre_outlier_mask]

    if outlier_filter == "none":
        outlier_mask = np.ones(len(filtered_xyz), dtype=bool)
    elif outlier_filter == "statistical":
        outlier_mask = statistical_outlier_mask(filtered_xyz, k=outlier_k, std_ratio=outlier_std_ratio)
    elif outlier_filter == "voxel":
        outlier_mask = voxel_density_outlier_mask(filtered_xyz, voxel_size=voxel_size, min_neighbors=voxel_min_neighbors)
    else:
        raise ValueError("outlier_filter must be 'none', 'statistical', or 'voxel'")

    output_xyz = filtered_xyz[outlier_mask]
    output_rgb = filtered_rgb[outlier_mask]
    summary = {
        "input_points": int(len(pts)),
        "output_points": int(len(output_xyz)),
        "mode": mode,
        "trajectory_radius": float(trajectory_radius),
        "height_below": float(height_below),
        "height_above": float(height_above),
        "body_radius": float(body_radius),
        "pose_count": int(len(poses)),
        "valid_pose_centers": int(len(center_arr)),
        "body_segments": int(len(starts)),
        "removed_by_roi": int(np.count_nonzero(~roi_mask)),
        "removed_by_body": int(np.count_nonzero(roi_mask & ~body_mask)),
        "removed_by_outlier": int(np.count_nonzero(pre_outlier_mask) - len(output_xyz)),
        "outlier_filter": outlier_filter,
        "outlier_k": int(outlier_k),
        "outlier_std_ratio": float(outlier_std_ratio),
        "voxel_size": float(voxel_size),
        "voxel_min_neighbors": int(voxel_min_neighbors),
    }
    if len(center_arr):
        summary["person_center_mean"] = [float(v) for v in center_arr.mean(axis=0)]
    return FilteredPointCloud(output_xyz, output_rgb, summary)


def filter_person_centered_ply(
    *,
    input_ply: Path,
    pose_jsons: list[Path],
    output_ply: Path,
    track_id: int | None = None,
    min_conf: float = 0.3,
    mode: str = "scene",
    trajectory_radius: float = 8.0,
    height_below: float = 2.0,
    height_above: float = 3.0,
    body_radius: float = 0.35,
    outlier_filter: str = "statistical",
    outlier_k: int = 20,
    outlier_std_ratio: float = 2.0,
    voxel_size: float = 0.1,
    voxel_min_neighbors: int = 4,
) -> dict[str, Any]:
    xyz, rgb = read_ascii_xyzrgb_ply(input_ply)
    poses = [load_person_keypoints(path, track_id=track_id, min_conf=min_conf) for path in pose_jsons]
    result = filter_person_centered_points(
        xyz,
        rgb,
        poses,
        mode=mode,
        trajectory_radius=trajectory_radius,
        height_below=height_below,
        height_above=height_above,
        body_radius=body_radius,
        outlier_filter=outlier_filter,
        outlier_k=outlier_k,
        outlier_std_ratio=outlier_std_ratio,
        voxel_size=voxel_size,
        voxel_min_neighbors=voxel_min_neighbors,
    )
    write_ascii_xyzrgb_ply(output_ply, result.xyz, result.rgb)
    summary = {
        "input_ply": str(input_ply),
        "output_ply": str(output_ply),
        "pose_jsons": [str(path) for path in pose_jsons],
        "track_id": track_id,
        "min_conf": float(min_conf),
        **result.summary,
    }
    output_ply.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

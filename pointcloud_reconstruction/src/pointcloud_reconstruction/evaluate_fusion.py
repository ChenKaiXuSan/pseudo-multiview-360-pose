from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from .merge_colmap import find_colmap_points3d, read_colmap_points3d, transform_point


def finite_array(values) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    return arr[np.isfinite(arr)]


def describe_values(values) -> dict[str, float | int | None]:
    arr = finite_array(values)
    if arr.size == 0:
        return {"count": 0, "mean": None, "median": None, "std": None, "min": None, "max": None, "p90": None, "p95": None}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
    }


def points_xyz(points: list[dict[str, Any]]) -> np.ndarray:
    if not points:
        return np.zeros((0, 3), dtype=np.float32)
    xyz = np.asarray([point["xyz"] for point in points], dtype=np.float32)
    finite = np.isfinite(xyz).all(axis=1)
    return xyz[finite]


def point_cloud_metrics(points: list[dict[str, Any]]) -> dict[str, Any]:
    xyz = points_xyz(points)
    errors = [float(point.get("error", np.nan)) for point in points]
    if xyz.size == 0:
        bbox = {"min": None, "max": None, "extent": None}
    else:
        mins = xyz.min(axis=0)
        maxs = xyz.max(axis=0)
        bbox = {
            "min": [float(v) for v in mins],
            "max": [float(v) for v in maxs],
            "extent": [float(v) for v in (maxs - mins)],
        }
    return {
        "point_count": int(len(points)),
        "finite_xyz_count": int(len(xyz)),
        "reprojection_error": describe_values(errors),
        "bbox": bbox,
    }


def deterministic_sample(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(points), size=max_points, replace=False)
    return points[np.sort(indices)]


def nearest_neighbor_distances(src: np.ndarray, dst: np.ndarray, chunk_size: int = 1024) -> np.ndarray:
    if len(src) == 0 or len(dst) == 0:
        return np.zeros(0, dtype=np.float32)
    distances = []
    dst64 = dst.astype(np.float32, copy=False)
    for start in range(0, len(src), chunk_size):
        chunk = src[start:start + chunk_size].astype(np.float32, copy=False)
        diff = chunk[:, None, :] - dst64[None, :, :]
        sq = np.einsum("ijk,ijk->ij", diff, diff)
        distances.append(np.sqrt(np.min(sq, axis=1)))
    return np.concatenate(distances).astype(np.float32)


def pairwise_overlap_metrics(view_points: dict[str, np.ndarray], sample_points: int = 5000, seed: int = 13) -> list[dict[str, Any]]:
    metrics = []
    sampled = {
        name: deterministic_sample(points, sample_points, seed + idx)
        for idx, (name, points) in enumerate(sorted(view_points.items()))
    }
    for a, b in combinations(sorted(sampled.keys()), 2):
        a_pts = sampled[a]
        b_pts = sampled[b]
        a_to_b = nearest_neighbor_distances(a_pts, b_pts)
        b_to_a = nearest_neighbor_distances(b_pts, a_pts)
        both = np.concatenate([a_to_b, b_to_a]) if len(a_to_b) or len(b_to_a) else np.zeros(0, dtype=np.float32)
        metrics.append({
            "pair": [a, b],
            "sample_counts": {a: int(len(a_pts)), b: int(len(b_pts))},
            "a_to_b": describe_values(a_to_b),
            "b_to_a": describe_values(b_to_a),
            "symmetric_chamfer_l1": float(both.mean()) if both.size else None,
            "symmetric_median_nn": float(np.median(both)) if both.size else None,
            "symmetric_p90_nn": float(np.percentile(both, 90)) if both.size else None,
        })
    return metrics


def trajectory_metrics(poses: np.ndarray) -> dict[str, Any]:
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        return {"frame_count": 0, "path_length": 0.0, "translation_step": describe_values([]), "rotation_step_deg": describe_values([]), "translation_accel": describe_values([])}
    translations = poses[:, :3, 3]
    if len(translations) < 2:
        return {"frame_count": int(len(poses)), "path_length": 0.0, "translation_step": describe_values([]), "rotation_step_deg": describe_values([]), "translation_accel": describe_values([])}
    deltas = np.diff(translations, axis=0)
    step = np.linalg.norm(deltas, axis=1)
    accel = np.linalg.norm(np.diff(deltas, axis=0), axis=1) if len(deltas) > 1 else np.zeros(0, dtype=np.float64)
    rot_steps = []
    for prev, curr in zip(poses[:-1, :3, :3], poses[1:, :3, :3]):
        rel = prev.T @ curr
        trace = np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0)
        rot_steps.append(np.degrees(np.arccos(trace)))
    return {
        "frame_count": int(len(poses)),
        "path_length": float(step.sum()),
        "translation_step": describe_values(step),
        "rotation_step_deg": describe_values(rot_steps),
        "translation_accel": describe_values(accel),
    }


def load_pose_metrics(vipe_results_dir: Path, view_name: str) -> dict[str, Any]:
    pose_path = vipe_results_dir / view_name / "pose" / f"{view_name}.npz"
    if not pose_path.exists():
        return {"status": "missing", "path": str(pose_path)}
    data = np.load(pose_path)
    poses = data["data"] if "data" in data.files else np.zeros((0, 4, 4), dtype=np.float32)
    metrics = trajectory_metrics(poses)
    metrics["status"] = "used"
    metrics["path"] = str(pose_path)
    if "inds" in data.files:
        metrics["frame_indices"] = {"first": int(data["inds"][0]), "last": int(data["inds"][-1]), "count": int(len(data["inds"]))} if len(data["inds"]) else {"first": None, "last": None, "count": 0}
    return metrics


def evaluate_fusion(
    *,
    manifest_path: Path,
    colmap_root: Path,
    vipe_results_dir: Path | None,
    output_json: Path,
    sample_points: int = 5000,
    max_error: float | None = None,
    scale: float = 1.0,
    seed: int = 13,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    per_view: dict[str, Any] = {}
    world_points: dict[str, np.ndarray] = {}
    total_points = 0
    for view in manifest.get("views", []):
        view_name = str(view["name"])
        points_path = find_colmap_points3d(colmap_root, view_name)
        if points_path is None:
            per_view[view_name] = {"status": "missing", "points3D": str(colmap_root / view_name / "points3D.txt")}
            world_points[view_name] = np.zeros((0, 3), dtype=np.float32)
            continue
        raw_points = read_colmap_points3d(points_path)
        if max_error is not None:
            raw_points = [point for point in raw_points if float(point.get("error", np.inf)) <= max_error]
        view_metrics = point_cloud_metrics(raw_points)
        view_metrics["status"] = "used"
        view_metrics["points3D"] = str(points_path)
        if vipe_results_dir is not None:
            view_metrics["trajectory"] = load_pose_metrics(vipe_results_dir, view_name)
        per_view[view_name] = view_metrics
        total_points += int(view_metrics["point_count"])
        transformed = [transform_point(point["xyz"], view["camera_to_world"], scale=scale) for point in raw_points]
        world_points[view_name] = np.asarray(transformed, dtype=np.float32)

    overlap = pairwise_overlap_metrics(world_points, sample_points=sample_points, seed=seed)
    result = {
        "manifest_path": str(manifest_path),
        "colmap_root": str(colmap_root),
        "vipe_results_dir": str(vipe_results_dir) if vipe_results_dir else None,
        "sample_points_per_view": int(sample_points),
        "max_error": max_error,
        "scale": float(scale),
        "total_points_after_filter": int(total_points),
        "per_view": per_view,
        "pairwise_overlap": overlap,
        "notes": [
            "Nearest-neighbor overlap is computed after the current rough rotation-based merge, not after Sim3 alignment.",
            "Use these metrics for relative comparison between settings, not as absolute 3D accuracy without GT.",
        ],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

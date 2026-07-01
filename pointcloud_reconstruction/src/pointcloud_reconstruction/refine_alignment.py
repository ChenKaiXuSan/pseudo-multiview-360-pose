from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .evaluate_fusion import describe_values, nearest_neighbor_distances, pairwise_overlap_metrics
from .merge_colmap import find_colmap_points3d, read_colmap_points3d, transform_point, write_ascii_ply


@dataclass
class ICPResult:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    aligned_points: np.ndarray
    iterations: int
    final_median_error: float | None
    final_mean_error: float | None
    inlier_count: int
    sampled_source_count: int
    sampled_reference_count: int

    def as_transform_json(self) -> dict[str, Any]:
        return {
            "scale": float(self.scale),
            "rotation": self.rotation.tolist(),
            "translation": [float(v) for v in self.translation],
            "iterations": int(self.iterations),
            "final_median_error": self.final_median_error,
            "final_mean_error": self.final_mean_error,
            "inlier_count": int(self.inlier_count),
            "sampled_source_count": int(self.sampled_source_count),
            "sampled_reference_count": int(self.sampled_reference_count),
        }


def finite_points(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return np.zeros((0, 3), dtype=np.float64)
    return arr[np.isfinite(arr).all(axis=1)]


def apply_similarity(points: np.ndarray, *, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    return float(scale) * (pts @ np.asarray(rotation, dtype=np.float64).T) + np.asarray(translation, dtype=np.float64)


def compose_similarity(
    *,
    base_scale: float,
    base_rotation: np.ndarray,
    base_translation: np.ndarray,
    delta_scale: float,
    delta_rotation: np.ndarray,
    delta_translation: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    scale = float(delta_scale) * float(base_scale)
    rotation = np.asarray(delta_rotation, dtype=np.float64) @ np.asarray(base_rotation, dtype=np.float64)
    translation = float(delta_scale) * (np.asarray(delta_rotation, dtype=np.float64) @ np.asarray(base_translation, dtype=np.float64)) + np.asarray(delta_translation, dtype=np.float64)
    return scale, rotation, translation


def estimate_similarity_umeyama(source: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    src = finite_points(source)
    dst = finite_points(target)
    if len(src) != len(dst) or len(src) < 3:
        raise ValueError("source and target must have the same length and at least 3 finite points")

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    variance = np.mean(np.sum(src_centered * src_centered, axis=1))
    if variance <= 0.0:
        raise ValueError("source points have zero variance")

    covariance = (dst_centered.T @ src_centered) / len(src)
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt
    scale = float(np.sum(singular_values * np.diag(correction)) / variance)
    translation = dst_mean - scale * (rotation @ src_mean)
    return {"scale": scale, "rotation": rotation, "translation": translation}


def deterministic_sample_array(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    pts = finite_points(points)
    if max_points <= 0 or len(pts) <= max_points:
        return pts
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(pts), size=max_points, replace=False)
    return pts[np.sort(indices)]


def initial_moment_similarity(source: np.ndarray, reference: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    src = finite_points(source)
    ref = finite_points(reference)
    if len(src) < 3 or len(ref) < 3:
        return 1.0, np.eye(3), np.zeros(3)
    src_center = np.median(src, axis=0)
    ref_center = np.median(ref, axis=0)
    src_radius = np.median(np.linalg.norm(src - src_center, axis=1))
    ref_radius = np.median(np.linalg.norm(ref - ref_center, axis=1))
    scale = float(ref_radius / src_radius) if src_radius > 0 else 1.0
    translation = ref_center - scale * src_center
    return scale, np.eye(3), translation


def nearest_neighbor_pairs(source: np.ndarray, target: np.ndarray, chunk_size: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    src = finite_points(source)
    dst = finite_points(target)
    if len(src) == 0 or len(dst) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
    indices = []
    distances = []
    dst32 = dst.astype(np.float32, copy=False)
    for start in range(0, len(src), chunk_size):
        chunk = src[start:start + chunk_size].astype(np.float32, copy=False)
        diff = chunk[:, None, :] - dst32[None, :, :]
        sq = np.einsum("ijk,ijk->ij", diff, diff)
        local = np.argmin(sq, axis=1)
        indices.append(local)
        distances.append(np.sqrt(sq[np.arange(len(local)), local]))
    return np.concatenate(indices).astype(np.int64), np.concatenate(distances).astype(np.float64)


def refine_similarity_icp(
    source_points: np.ndarray,
    reference_points: np.ndarray,
    *,
    max_iterations: int = 12,
    trim_fraction: float = 0.7,
    distance_threshold: float | None = None,
    min_inliers: int = 20,
    seed: int = 13,
) -> ICPResult:
    source = finite_points(source_points)
    reference = finite_points(reference_points)
    if len(source) < 3 or len(reference) < 3:
        return ICPResult(1.0, np.eye(3), np.zeros(3), source, 0, None, None, 0, len(source), len(reference))

    scale, rotation, translation = initial_moment_similarity(source, reference)
    aligned = apply_similarity(source, scale=scale, rotation=rotation, translation=translation)
    final_distances = np.zeros(0, dtype=np.float64)
    inlier_count = 0

    trim_fraction = min(max(float(trim_fraction), 0.05), 1.0)
    for iteration in range(int(max_iterations)):
        nn_indices, distances = nearest_neighbor_pairs(aligned, reference)
        if len(distances) == 0:
            break
        order = np.argsort(distances)
        keep_count = max(min_inliers, int(round(len(order) * trim_fraction)))
        keep_count = min(keep_count, len(order))
        keep = order[:keep_count]
        if distance_threshold is not None:
            keep = keep[distances[keep] <= distance_threshold]
        if len(keep) < 3:
            break

        matched = reference[nn_indices[keep]]
        delta = estimate_similarity_umeyama(aligned[keep], matched)
        scale, rotation, translation = compose_similarity(
            base_scale=scale,
            base_rotation=rotation,
            base_translation=translation,
            delta_scale=delta["scale"],
            delta_rotation=delta["rotation"],
            delta_translation=delta["translation"],
        )
        aligned = apply_similarity(source, scale=scale, rotation=rotation, translation=translation)
        final_distances = distances[keep]
        inlier_count = len(keep)
        if np.median(final_distances) < 1e-8:
            return ICPResult(scale, rotation, translation, aligned, iteration + 1, float(np.median(final_distances)), float(np.mean(final_distances)), inlier_count, len(source), len(reference))

    if len(final_distances) == 0:
        _, distances = nearest_neighbor_pairs(aligned, reference)
        final_distances = distances
        inlier_count = len(distances)
    return ICPResult(
        scale,
        rotation,
        translation,
        aligned,
        int(max_iterations),
        float(np.median(final_distances)) if len(final_distances) else None,
        float(np.mean(final_distances)) if len(final_distances) else None,
        inlier_count,
        len(source),
        len(reference),
    )


def similarity_warning(*, scale: float, median_error: float | None, max_scale_ratio: float, max_median_error: float) -> list[str]:
    warnings = []
    if scale <= 0 or scale < 1.0 / max_scale_ratio or scale > max_scale_ratio:
        warnings.append("scale_out_of_range")
    if median_error is not None and median_error > max_median_error:
        warnings.append("median_error_high")
    return warnings



def clockwise_yaw_delta_degrees(yaw: float, reference_yaw: float) -> float:
    return float((float(yaw) - float(reference_yaw)) % 360.0)


def build_adjacent_view_edges(views: list[dict[str, Any]], reference_view: str | None = None) -> list[tuple[str, str]]:
    """Build a circular horizontal adjacency graph from manifest view yaw values."""
    yaw_views = [
        {"name": str(view["name"]), "yaw_deg": float(view.get("yaw_deg", 0.0))}
        for view in views
        if "name" in view
    ]
    if len(yaw_views) < 2:
        return []

    ref_name = reference_view or yaw_views[0]["name"]
    ref = next((view for view in yaw_views if view["name"] == ref_name), yaw_views[0])
    ref_yaw = float(ref["yaw_deg"])
    ordered = sorted(yaw_views, key=lambda view: clockwise_yaw_delta_degrees(view["yaw_deg"], ref_yaw))
    return [
        (ordered[idx]["name"], ordered[(idx + 1) % len(ordered)]["name"])
        for idx in range(len(ordered))
    ]


def build_alignment_parent_order(
    view_names: list[str],
    *,
    reference_view: str,
    adjacency_edges: list[tuple[str, str]],
) -> tuple[dict[str, str | None], list[str]]:
    names = [str(name) for name in view_names]
    if reference_view not in names:
        raise ValueError(f"reference view '{reference_view}' is missing")

    neighbors: dict[str, list[str]] = {name: [] for name in names}
    for left, right in adjacency_edges:
        if left in neighbors and right in neighbors:
            neighbors[left].append(right)
            neighbors[right].append(left)

    parents: dict[str, str | None] = {reference_view: None}
    order = [reference_view]
    queue: deque[str] = deque([reference_view])
    while queue:
        current = queue.popleft()
        for candidate in neighbors.get(current, []):
            if candidate in parents:
                continue
            parents[candidate] = current
            order.append(candidate)
            queue.append(candidate)

    for name in names:
        if name not in parents:
            parents[name] = reference_view if name != reference_view else None
            order.append(name)
    return parents, order



def scale_in_range(scale: float, scale_min: float | None, scale_max: float | None) -> bool:
    value = float(scale)
    if scale_min is not None and value < float(scale_min):
        return False
    if scale_max is not None and value > float(scale_max):
        return False
    return True


def select_best_alignment_candidate(
    candidates: list[dict[str, Any]],
    *,
    scale_min: float | None = None,
    scale_max: float | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    rejected: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []
    for candidate in candidates:
        scale = float(candidate.get("scale", 1.0))
        item = dict(candidate)
        if not scale_in_range(scale, scale_min, scale_max):
            item["reason"] = "scale_out_of_range"
            rejected.append(item)
            continue
        valid.append(item)
    if not valid:
        return None, rejected
    return min(valid, key=lambda item: float(item.get("final_median_error") if item.get("final_median_error") is not None else np.inf)), rejected


def adjacency_neighbors(view_names: list[str], adjacency_edges: list[tuple[str, str]]) -> dict[str, list[str]]:
    neighbors: dict[str, list[str]] = {name: [] for name in view_names}
    for left, right in adjacency_edges:
        if left in neighbors and right in neighbors:
            neighbors[left].append(right)
            neighbors[right].append(left)
    return neighbors


def clamp_scale(scale: float, scale_min: float | None, scale_max: float | None) -> tuple[float, bool]:
    value = float(scale)
    clamped = value
    if scale_min is not None:
        clamped = max(float(scale_min), clamped)
    if scale_max is not None:
        clamped = min(float(scale_max), clamped)
    return clamped, abs(clamped - value) > 1e-12


def load_view_world_points(
    *,
    manifest_path: Path,
    colmap_root: Path,
    max_error: float | None = None,
    rough_scale: float = 1.0,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    view_points: dict[str, list[dict[str, Any]]] = {}
    view_meta: dict[str, Any] = {}
    for view in manifest.get("views", []):
        view_name = str(view["name"])
        points_path = find_colmap_points3d(colmap_root, view_name)
        if points_path is None:
            view_points[view_name] = []
            view_meta[view_name] = {"status": "missing", "points3D": str(colmap_root / view_name / "points3D.txt")}
            continue
        raw = read_colmap_points3d(points_path)
        transformed = []
        for point in raw:
            if max_error is not None and float(point.get("error", np.inf)) > max_error:
                continue
            transformed.append({
                "xyz": transform_point(point["xyz"], view["camera_to_world"], scale=rough_scale),
                "rgb": point.get("rgb", [255, 255, 255]),
                "error": float(point.get("error", np.nan)),
            })
        view_points[view_name] = transformed
        view_meta[view_name] = {"status": "used", "points3D": str(points_path), "points": len(transformed)}
    return view_points, view_meta


def points_to_array(points: list[dict[str, Any]]) -> np.ndarray:
    if not points:
        return np.zeros((0, 3), dtype=np.float64)
    return finite_points(np.asarray([point["xyz"] for point in points], dtype=np.float64))


def refine_manifest_colmap_points(
    *,
    manifest_path: Path,
    colmap_root: Path,
    output_ply: Path,
    alignment_json: Path,
    reference_view: str = "selfie",
    max_error: float | None = None,
    rough_scale: float = 1.0,
    sample_points: int = 8000,
    max_iterations: int = 12,
    trim_fraction: float = 0.7,
    distance_threshold: float | None = None,
    max_scale_ratio: float = 4.0,
    max_median_error: float = 10.0,
    alignment_mode: str = "direct",
    scale_min: float | None = None,
    scale_max: float | None = None,
    seed: int = 13,
    export_frame_plys: bool = True,
    frame_output_dir: Path | None = None,
    frame_max_frames: int | None = None,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    view_points, view_meta = load_view_world_points(
        manifest_path=manifest_path,
        colmap_root=colmap_root,
        max_error=max_error,
        rough_scale=rough_scale,
    )
    if reference_view not in view_points or len(view_points[reference_view]) == 0:
        raise ValueError(f"reference view '{reference_view}' is missing or empty")

    view_names = [str(view["name"]) for view in manifest.get("views", []) if str(view.get("name")) in view_points]
    mode = str(alignment_mode or "direct").lower()
    if mode == "adjacent":
        adjacency_edges = build_adjacent_view_edges(manifest.get("views", []), reference_view=reference_view)
        parents: dict[str, str | None] = {reference_view: None}
        alignment_order = [reference_view]
        graph_neighbors = adjacency_neighbors(view_names, adjacency_edges)
    elif mode == "direct":
        adjacency_edges = []
        alignment_order = view_names
        parents = {name: (None if name == reference_view else reference_view) for name in view_names}
        graph_neighbors = {name: [reference_view] for name in view_names}
    else:
        raise ValueError("alignment_mode must be 'adjacent' or 'direct'")

    refined_points: list[dict[str, Any]] = []
    transforms: dict[str, Any] = {}
    refined_arrays: dict[str, np.ndarray] = {}
    edge_scores: list[dict[str, Any]] = []
    rejected_edges: list[dict[str, Any]] = []

    def append_refined_points(view_name: str, points: list[dict[str, Any]], aligned: np.ndarray) -> None:
        refined_arrays[view_name] = aligned
        for point, coord in zip(points, aligned):
            refined_points.append({"xyz": [float(coord[0]), float(coord[1]), float(coord[2])], "rgb": point.get("rgb", [255, 255, 255])})

    reference_points = view_points.get(reference_view, [])
    reference_xyz = points_to_array(reference_points)
    transforms[reference_view] = {
        "status": "reference",
        "scale": 1.0,
        "rotation": np.eye(3).tolist(),
        "translation": [0.0, 0.0, 0.0],
        "warnings": [],
        "source_points": len(reference_points),
        "parent_view": None,
    }
    append_refined_points(reference_view, reference_points, reference_xyz)

    def align_view_to_parent(view_name: str, parent_view: str | None) -> None:
        points = view_points.get(view_name, [])
        xyz = points_to_array(points)
        if len(xyz) < 3:
            aligned = xyz
            transforms[view_name] = {"status": "skipped", "reason": "too_few_points", "warnings": ["too_few_points"], "source_points": len(points), "parent_view": parent_view}
        elif parent_view not in refined_arrays or len(refined_arrays[parent_view]) < 3:
            aligned = xyz
            transforms[view_name] = {"status": "skipped", "reason": "missing_parent", "warnings": ["missing_parent"], "source_points": len(points), "parent_view": parent_view}
        else:
            source_sample = deterministic_sample_array(xyz, sample_points, seed + len(transforms) + 1)
            parent_sample = deterministic_sample_array(refined_arrays[parent_view], sample_points, seed + len(transforms) + 101)
            icp = refine_similarity_icp(
                source_sample,
                parent_sample,
                max_iterations=max_iterations,
                trim_fraction=trim_fraction,
                distance_threshold=distance_threshold,
                seed=seed,
            )
            effective_scale, scale_was_clamped = clamp_scale(icp.scale, scale_min, scale_max)
            aligned = apply_similarity(xyz, scale=effective_scale, rotation=icp.rotation, translation=icp.translation)
            transform_json = icp.as_transform_json()
            transform_json["original_scale"] = float(icp.scale)
            transform_json["scale"] = float(effective_scale)
            transform_json["status"] = "aligned"
            transform_json["parent_view"] = parent_view
            warnings = similarity_warning(
                scale=effective_scale,
                median_error=icp.final_median_error,
                max_scale_ratio=max_scale_ratio,
                max_median_error=max_median_error,
            )
            if scale_was_clamped:
                warnings.append("scale_clamped")
            transform_json["warnings"] = warnings
            transform_json["source_points"] = len(points)
            transforms[view_name] = transform_json
        append_refined_points(view_name, points, aligned)

    if mode == "direct":
        for view_name in alignment_order:
            if view_name != reference_view:
                align_view_to_parent(view_name, parents.get(view_name))
    else:
        pending = [name for name in view_names if name != reference_view]
        while pending:
            best_pending: tuple[float, str, dict[str, Any], list[dict[str, Any]]] | None = None
            for view_name in pending:
                xyz = points_to_array(view_points.get(view_name, []))
                if len(xyz) < 3:
                    best_pending = (-np.inf, view_name, {"parent_view": None, "status": "too_few_points"}, [])
                    break
                candidates: list[dict[str, Any]] = []
                for parent_view in graph_neighbors.get(view_name, []):
                    if parent_view not in refined_arrays or len(refined_arrays[parent_view]) < 3:
                        continue
                    source_sample = deterministic_sample_array(xyz, sample_points, seed + len(transforms) + len(candidates) + 1)
                    parent_sample = deterministic_sample_array(refined_arrays[parent_view], sample_points, seed + len(transforms) + len(candidates) + 101)
                    icp = refine_similarity_icp(
                        source_sample,
                        parent_sample,
                        max_iterations=max_iterations,
                        trim_fraction=trim_fraction,
                        distance_threshold=distance_threshold,
                        seed=seed,
                    )
                    candidates.append({
                        "view": view_name,
                        "parent_view": parent_view,
                        "scale": float(icp.scale),
                        "final_median_error": icp.final_median_error,
                        "final_mean_error": icp.final_mean_error,
                    })
                selected, rejected = select_best_alignment_candidate(candidates, scale_min=scale_min, scale_max=scale_max)
                rejected_edges.extend({"view": view_name, **item} for item in rejected)
                if selected is None and rejected:
                    selected = min(
                        rejected,
                        key=lambda item: float(item.get("final_median_error") if item.get("final_median_error") is not None else np.inf),
                    )
                    selected = dict(selected)
                    selected["used_rejected_scale_edge"] = True
                if selected is None:
                    continue
                score = float(selected.get("final_median_error") if selected.get("final_median_error") is not None else np.inf)
                if best_pending is None or score < best_pending[0]:
                    best_pending = (score, view_name, selected, rejected)
            if best_pending is None:
                for view_name in pending:
                    parents[view_name] = reference_view
                    alignment_order.append(view_name)
                    align_view_to_parent(view_name, reference_view)
                break
            _, view_name, selected, _ = best_pending
            if selected.get("status") == "too_few_points":
                parents[view_name] = None
                alignment_order.append(view_name)
                align_view_to_parent(view_name, None)
            else:
                parent_view = str(selected["parent_view"])
                parents[view_name] = parent_view
                alignment_order.append(view_name)
                edge_scores.append(selected)
                align_view_to_parent(view_name, parent_view)
                if selected.get("used_rejected_scale_edge"):
                    transforms[view_name].setdefault("warnings", []).append("used_rejected_scale_edge")
            pending.remove(view_name)

    overlap = pairwise_overlap_metrics({name: arr.astype(np.float32) for name, arr in refined_arrays.items()}, sample_points=min(sample_points, 5000), seed=seed)
    summary = {
        "manifest_path": str(manifest_path),
        "colmap_root": str(colmap_root),
        "output_ply": str(output_ply),
        "reference_view": reference_view,
        "max_error": max_error,
        "rough_scale": float(rough_scale),
        "sample_points": int(sample_points),
        "max_iterations": int(max_iterations),
        "trim_fraction": float(trim_fraction),
        "distance_threshold": distance_threshold,
        "alignment_mode": mode,
        "alignment_graph": {
            "edges": [[left, right] for left, right in adjacency_edges],
            "parents": parents,
            "order": alignment_order,
            "scale_min": scale_min,
            "scale_max": scale_max,
            "edge_scores": edge_scores,
            "rejected_edges": rejected_edges,
        },
        "total_points": len(refined_points),
        "sources": view_meta,
        "transforms": transforms,
        "pairwise_overlap": overlap,
        "frame_plys": None,
        "notes": [
            "Transforms are estimated after the rough manifest rotation.",
            "This is no-GT self-consistency refinement; absolute world scale is not guaranteed.",
        ],
    }
    alignment_json.parent.mkdir(parents=True, exist_ok=True)
    alignment_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if export_frame_plys:
        from .frame_ply_export import export_refined_frame_plys

        resolved_frame_output_dir = frame_output_dir if frame_output_dir is not None else output_ply.parent / "refined_frame_plys"
        frame_summary = export_refined_frame_plys(
            manifest_path=manifest_path,
            colmap_root=colmap_root,
            alignment_json=alignment_json,
            output_dir=resolved_frame_output_dir,
            max_error=max_error,
            rough_scale=rough_scale,
            max_frames=frame_max_frames,
        )
        summary["frame_plys"] = {
            "output_dir": frame_summary["output_dir"],
            "summary_json": str(resolved_frame_output_dir / "frame_plys_summary.json"),
            "num_frames": frame_summary["num_frames"],
            "total_frame_observations": frame_summary["total_frame_observations"],
        }

    output_ply.parent.mkdir(parents=True, exist_ok=True)
    write_ascii_ply(output_ply, refined_points)
    alignment_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

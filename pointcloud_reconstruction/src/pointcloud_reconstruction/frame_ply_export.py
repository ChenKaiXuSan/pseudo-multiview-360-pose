from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .merge_colmap import find_colmap_points3d, transform_point, write_ascii_ply
from .refine_alignment import apply_similarity


FRAME_RE = re.compile(r"frame_(\d+)")


def parse_colmap_image_frames(path: Path) -> dict[int, int]:
    """Map COLMAP IMAGE_ID to frame index parsed from image names."""
    frames: dict[int, int] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        try:
            image_id = int(parts[0])
        except ValueError:
            continue
        match = FRAME_RE.search(parts[-1])
        if match:
            frames[image_id] = int(match.group(1))
    return frames


def parse_colmap_points3d_with_tracks(path: Path) -> list[dict[str, Any]]:
    """Read COLMAP points3D.txt with positive IMAGE_ID observations."""
    points: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        try:
            track_values = [int(float(value)) for value in parts[8:]]
        except ValueError:
            track_values = []
        image_ids = []
        for idx in range(0, len(track_values) - 1, 2):
            image_id = track_values[idx]
            if image_id > 0:
                image_ids.append(image_id)
        points.append({
            "xyz": [float(parts[1]), float(parts[2]), float(parts[3])],
            "rgb": [int(parts[4]), int(parts[5]), int(parts[6])],
            "error": float(parts[7]),
            "image_ids": sorted(set(image_ids)),
        })
    return points


def transform_from_alignment(alignment: dict[str, Any]) -> dict[str, Any]:
    return {
        "scale": float(alignment.get("scale", 1.0)),
        "rotation": np.asarray(alignment.get("rotation", np.eye(3)), dtype=np.float64),
        "translation": np.asarray(alignment.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64),
    }


def apply_refined_view_transform(point: list[float], transform: dict[str, Any]) -> list[float]:
    arr = np.asarray([point], dtype=np.float64)
    out = apply_similarity(
        arr,
        scale=transform["scale"],
        rotation=transform["rotation"],
        translation=transform["translation"],
    )[0]
    return [float(out[0]), float(out[1]), float(out[2])]


def export_refined_frame_plys(
    *,
    manifest_path: Path,
    colmap_root: Path,
    alignment_json: Path,
    output_dir: Path,
    max_error: float | None = None,
    rough_scale: float = 1.0,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Write one refined merged PLY per frame using COLMAP point tracks."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    alignment = json.loads(alignment_json.read_text(encoding="utf-8"))
    transforms = alignment.get("transforms", {})
    frame_points: dict[int, list[dict[str, Any]]] = defaultdict(list)
    sources = []

    for view in manifest.get("views", []):
        view_name = str(view["name"])
        points_path = find_colmap_points3d(colmap_root, view_name)
        images_path = colmap_root / view_name / "images.txt"
        if points_path is None or not images_path.exists():
            sources.append({"view": view_name, "status": "missing", "points": 0})
            continue
        image_to_frame = parse_colmap_image_frames(images_path)
        transform = transform_from_alignment(transforms.get(view_name, {}))
        points = parse_colmap_points3d_with_tracks(points_path)
        used = 0
        for point in points:
            if max_error is not None and float(point.get("error", np.inf)) > max_error:
                continue
            rough_world = transform_point(point["xyz"], view["camera_to_world"], scale=rough_scale)
            refined_world = apply_refined_view_transform(rough_world, transform)
            observed_frames = sorted({
                image_to_frame[image_id]
                for image_id in point.get("image_ids", [])
                if image_id in image_to_frame and (max_frames is None or image_to_frame[image_id] < max_frames)
            })
            for frame_idx in observed_frames:
                frame_points[frame_idx].append({"xyz": refined_world, "rgb": point.get("rgb", [255, 255, 255])})
                used += 1
        sources.append({
            "view": view_name,
            "status": "used",
            "points3D": str(points_path),
            "images": str(images_path),
            "points_read": len(points),
            "frame_observations_written": used,
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for frame_idx in sorted(frame_points):
        path = output_dir / f"frame_{frame_idx:06d}.ply"
        write_ascii_ply(path, frame_points[frame_idx])
        frames.append({"frame_index": frame_idx, "path": str(path), "points": len(frame_points[frame_idx])})

    summary = {
        "manifest_path": str(manifest_path),
        "colmap_root": str(colmap_root),
        "alignment_json": str(alignment_json),
        "output_dir": str(output_dir),
        "max_error": max_error,
        "rough_scale": float(rough_scale),
        "max_frames": max_frames,
        "num_frames": len(frames),
        "total_frame_observations": int(sum(frame["points"] for frame in frames)),
        "sources": sources,
        "frames": frames,
        "notes": [
            "Each frame PLY contains refined 3D points observed by that frame in COLMAP tracks.",
            "The same 3D map point may appear in multiple frame PLY files if it has multiple observations.",
        ],
    }
    (output_dir / "frame_plys_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def find_colmap_points3d(colmap_root: Path, view_name: str) -> Path | None:
    """Find points3D.txt for a view, including VIPE's nested sequence output layout."""
    direct = colmap_root / view_name / "points3D.txt"
    if direct.exists():
        return direct
    nested = colmap_root / view_name / view_name / "points3D.txt"
    if nested.exists():
        return nested
    matches = sorted((colmap_root / view_name).glob("*/points3D.txt"))
    if matches:
        return matches[0]
    return None


def read_colmap_points3d(path: Path) -> list[dict]:
    """Read COLMAP text-format points3D.txt records."""
    points = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        points.append(
            {
                "xyz": [float(parts[1]), float(parts[2]), float(parts[3])],
                "rgb": [int(parts[4]), int(parts[5]), int(parts[6])],
                "error": float(parts[7]),
            }
        )
    return points


def transform_point(point: list[float], matrix: list[list[float]], scale: float = 1.0) -> list[float]:
    vector = np.array([point[0] * scale, point[1] * scale, point[2] * scale, 1.0], dtype=np.float32)
    transformed = np.asarray(matrix, dtype=np.float32) @ vector
    return [float(transformed[0]), float(transformed[1]), float(transformed[2])]


def write_ascii_ply(path: Path, points: list[dict]) -> None:
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    for point in points:
        x, y, z = point["xyz"]
        r, g, b = point.get("rgb", [255, 255, 255])
        lines.append(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def merge_manifest_colmap_points(
    *,
    manifest_path: Path,
    colmap_root: Path,
    output_ply: Path,
    max_error: float | None = None,
    scale: float = 1.0,
) -> dict:
    """Roughly merge per-view COLMAP points using view camera-to-world rotations from the manifest."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    merged = []
    sources = []
    for view in manifest.get("views", []):
        view_name = view["name"]
        points_path = find_colmap_points3d(colmap_root, view_name)
        expected_path = colmap_root / view_name / "points3D.txt"
        if points_path is None:
            sources.append({"view": view_name, "points3D": str(expected_path), "status": "missing", "points": 0})
            continue
        raw_points = read_colmap_points3d(points_path)
        kept = 0
        for point in raw_points:
            if max_error is not None and point["error"] > max_error:
                continue
            merged.append(
                {
                    "xyz": transform_point(point["xyz"], view["camera_to_world"], scale=scale),
                    "rgb": point["rgb"],
                }
            )
            kept += 1
        sources.append({"view": view_name, "points3D": str(points_path), "status": "used", "points": kept})

    output_ply.parent.mkdir(parents=True, exist_ok=True)
    write_ascii_ply(output_ply, merged)
    summary = {
        "manifest_path": str(manifest_path),
        "colmap_root": str(colmap_root),
        "output_ply": str(output_ply),
        "total_points": len(merged),
        "scale": float(scale),
        "max_error": max_error,
        "sources": sources,
        "note": "This is a rough rotation-based merge. Similarity alignment should be estimated before quantitative use.",
    }
    output_ply.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

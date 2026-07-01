from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_BODY_EDGES: list[tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (1, 5),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (8, 11),
    (11, 12),
    (12, 13),
    (0, 14),
    (0, 15),
]


@dataclass(frozen=True)
class FusedKeypoints:
    values: np.ndarray
    mask_valid: np.ndarray
    frame_number: int | None = None
    track_id: int | None = None
    source_path: str | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    def __getitem__(self, index: int) -> "FusedKeypointRow":
        return FusedKeypointRow(self.values[index], bool(self.mask_valid[index]))


@dataclass(frozen=True)
class FusedKeypointRow:
    values: np.ndarray
    mask_valid: bool


def _rows_to_fused_keypoints(
    rows: Iterable[Iterable[float | int | None]],
    *,
    min_conf: float,
    frame_number: int | None = None,
    track_id: int | None = None,
    source_path: str | None = None,
) -> FusedKeypoints:
    converted: list[list[float]] = []
    for row in rows:
        values = []
        for value in list(row)[:4]:
            values.append(float("nan") if value is None else float(value))
        while len(values) < 4:
            values.append(1.0)
        converted.append(values)
    arr = np.asarray(converted, dtype=np.float64)
    if arr.size == 0:
        arr = np.zeros((0, 4), dtype=np.float64)
    mask = np.isfinite(arr[:, :3]).all(axis=1)
    if arr.shape[1] > 3:
        mask &= arr[:, 3] >= float(min_conf)
    return FusedKeypoints(
        values=arr,
        mask_valid=mask,
        frame_number=frame_number,
        track_id=track_id,
        source_path=source_path,
    )


def load_fused_keypoints(path: Path, min_conf: float = 0.3) -> FusedKeypoints:
    """Load one SAM3D multiview fused-keypoint JSON in world coordinates."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _rows_to_fused_keypoints(
        payload.get("fused_keypoints3d_world", []),
        min_conf=min_conf,
        frame_number=payload.get("frame_number"),
        track_id=payload.get("track_id"),
        source_path=str(path),
    )


load_fused_keypoints.from_rows = _rows_to_fused_keypoints  # type: ignore[attr-defined]


def _point(xyz: Iterable[float], rgb: Iterable[int]) -> dict[str, list[float] | list[int]]:
    return {
        "xyz": [float(value) for value in xyz],
        "rgb": [int(value) for value in rgb],
    }


def build_pose_overlay_points(
    keypoints: FusedKeypoints,
    *,
    edges: list[tuple[int, int]] | None = None,
    joint_radius: float = 0.08,
    bone_step: float = 0.04,
    joint_rgb: list[int] | None = None,
    bone_rgb: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Convert valid keypoints into dense colored points visible inside PLY viewers."""
    edges = DEFAULT_BODY_EDGES if edges is None else edges
    joint_rgb = [255, 40, 40] if joint_rgb is None else joint_rgb
    bone_rgb = [0, 255, 255] if bone_rgb is None else bone_rgb
    kpts = keypoints.values
    mask = keypoints.mask_valid
    points: list[dict[str, Any]] = []

    offsets = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [joint_radius, 0.0, 0.0],
            [-joint_radius, 0.0, 0.0],
            [0.0, joint_radius, 0.0],
            [0.0, -joint_radius, 0.0],
            [0.0, 0.0, joint_radius],
            [0.0, 0.0, -joint_radius],
        ],
        dtype=np.float64,
    )
    for idx, row in enumerate(kpts):
        if idx >= len(mask) or not mask[idx]:
            continue
        center = row[:3]
        for offset in offsets:
            points.append(_point(center + offset, joint_rgb))

    for a, b in edges:
        if a >= len(kpts) or b >= len(kpts) or not mask[a] or not mask[b]:
            continue
        start = kpts[a, :3]
        end = kpts[b, :3]
        length = float(np.linalg.norm(end - start))
        samples = max(2, int(math.ceil(length / max(float(bone_step), 1e-6))) + 1)
        for pos in np.linspace(0.0, 1.0, samples):
            points.append(_point(start * (1.0 - pos) + end * pos, bone_rgb))
    return points


def _read_ascii_ply_header(path: Path) -> tuple[list[str], int, int]:
    header: list[str] = []
    vertex_count: int | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.rstrip("\n")
            header.append(stripped)
            if stripped.startswith("element vertex "):
                vertex_count = int(stripped.split()[-1])
            if stripped == "end_header":
                if vertex_count is None:
                    raise ValueError(f"PLY header has no vertex count: {path}")
                return header, vertex_count, line_number
    raise ValueError(f"PLY header is incomplete: {path}")


def _format_ply_vertex(point: dict[str, Any]) -> str:
    x, y, z = point["xyz"]
    r, g, b = point.get("rgb", [255, 255, 255])
    return f"{float(x):.6f} {float(y):.6f} {float(z):.6f} {int(r)} {int(g)} {int(b)}"


def write_pose_pointcloud_overlay_frame(
    *,
    scene_ply: Path,
    pose_json: Path,
    output_ply: Path,
    min_conf: float = 0.3,
    edges: list[tuple[int, int]] | None = None,
    joint_radius: float = 0.08,
    bone_step: float = 0.04,
) -> dict[str, Any]:
    """Write one PLY containing the scene point cloud plus one fused 3D pose."""
    header, scene_points, header_lines = _read_ascii_ply_header(scene_ply)
    keypoints = load_fused_keypoints(pose_json, min_conf=min_conf)
    overlay_points = build_pose_overlay_points(
        keypoints,
        edges=edges,
        joint_radius=joint_radius,
        bone_step=bone_step,
    )

    output_ply.parent.mkdir(parents=True, exist_ok=True)
    with scene_ply.open("r", encoding="utf-8") as src, output_ply.open("w", encoding="utf-8") as dst:
        for idx, line in enumerate(header, start=1):
            if line.startswith("element vertex "):
                dst.write(f"element vertex {scene_points + len(overlay_points)}\n")
            else:
                dst.write(line + "\n")
        for _ in range(header_lines):
            next(src)
        for raw in src:
            dst.write(raw)
        for point in overlay_points:
            dst.write(_format_ply_vertex(point) + "\n")

    summary = {
        "scene_ply": str(scene_ply),
        "pose_json": str(pose_json),
        "output_ply": str(output_ply),
        "scene_points": scene_points,
        "overlay_points": len(overlay_points),
        "valid_keypoints": int(keypoints.mask_valid.sum()),
        "frame_number": keypoints.frame_number,
        "track_id": keypoints.track_id,
        "min_conf": float(min_conf),
        "joint_radius": float(joint_radius),
        "bone_step": float(bone_step),
    }
    output_ply.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def sample_ascii_ply_points(path: Path, max_points: int = 20000) -> tuple[np.ndarray, np.ndarray]:
    """Read a deterministic prefix sample from an ASCII xyzrgb PLY."""
    _, scene_points, header_lines = _read_ascii_ply_header(path)
    limit = min(scene_points, int(max_points))
    xyz: list[list[float]] = []
    rgb: list[list[float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for _ in range(header_lines):
            next(handle)
        for line in handle:
            if len(xyz) >= limit:
                break
            parts = line.split()
            if len(parts) < 6:
                continue
            xyz.append([float(parts[0]), float(parts[1]), float(parts[2])])
            rgb.append([int(parts[3]) / 255.0, int(parts[4]) / 255.0, int(parts[5]) / 255.0])
    return np.asarray(xyz, dtype=np.float64), np.asarray(rgb, dtype=np.float64)


def save_matplotlib_overlay_screenshot(
    *,
    scene_ply: Path,
    pose_json: Path,
    output_png: Path,
    min_conf: float = 0.3,
    max_scene_points: int = 20000,
) -> Path:
    """Save a static 3D preview without requiring an OpenGL display."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scene_xyz, scene_rgb = sample_ascii_ply_points(scene_ply, max_points=max_scene_points)
    keypoints = load_fused_keypoints(pose_json, min_conf=min_conf)
    valid = keypoints.values[keypoints.mask_valid, :3]

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    if len(scene_xyz):
        ax.scatter(scene_xyz[:, 0], scene_xyz[:, 1], scene_xyz[:, 2], s=0.15, c=scene_rgb, alpha=0.22)
    if len(valid):
        ax.scatter(valid[:, 0], valid[:, 1], valid[:, 2], s=28, c="#ff2828", depthshade=False)
        for a, b in DEFAULT_BODY_EDGES:
            if a < len(keypoints.values) and b < len(keypoints.values) and keypoints.mask_valid[a] and keypoints.mask_valid[b]:
                pts = keypoints.values[[a, b], :3]
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="#00ffff", linewidth=1.6)
    ax.set_title(f"frame {keypoints.frame_number} track {keypoints.track_id}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=18, azim=-68)
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)
    return output_png


def save_pillow_overlay_screenshot(
    *,
    scene_ply: Path,
    pose_json: Path,
    output_png: Path,
    min_conf: float = 0.3,
    max_scene_points: int = 30000,
    image_size: tuple[int, int] = (1400, 1000),
) -> Path:
    """Save a lightweight top-down x-z PNG preview using Pillow only."""
    from PIL import Image, ImageDraw

    scene_xyz, scene_rgb = sample_ascii_ply_points(scene_ply, max_points=max_scene_points)
    keypoints = load_fused_keypoints(pose_json, min_conf=min_conf)
    valid = keypoints.values[keypoints.mask_valid, :3]

    width, height = image_size
    image = Image.new("RGB", image_size, (248, 248, 246))
    draw = ImageDraw.Draw(image, "RGBA")

    if len(scene_xyz) or len(valid):
        all_xz = []
        if len(scene_xyz):
            all_xz.append(scene_xyz[:, [0, 2]])
        if len(valid):
            all_xz.append(valid[:, [0, 2]])
        coords = np.vstack(all_xz)
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        span = np.maximum(maxs - mins, 1e-6)
    else:
        mins = np.asarray([0.0, 0.0])
        span = np.asarray([1.0, 1.0])

    margin = 45

    def project(xyz: np.ndarray) -> tuple[int, int]:
        xz = xyz[[0, 2]]
        norm = (xz - mins) / span
        x = margin + norm[0] * (width - margin * 2)
        y = height - margin - norm[1] * (height - margin * 2)
        return int(round(x)), int(round(y))

    if len(scene_xyz):
        for xyz, rgb in zip(scene_xyz, scene_rgb):
            x, y = project(xyz)
            color = tuple(int(max(0.0, min(1.0, channel)) * 255) for channel in rgb)
            draw.point((x, y), fill=color + (70,))

    for a, b in DEFAULT_BODY_EDGES:
        if a < len(keypoints.values) and b < len(keypoints.values) and keypoints.mask_valid[a] and keypoints.mask_valid[b]:
            draw.line([project(keypoints.values[a, :3]), project(keypoints.values[b, :3])], fill=(0, 220, 240, 255), width=4)
    for xyz in valid:
        x, y = project(xyz)
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(255, 40, 40, 255), outline=(255, 255, 255, 255), width=2)

    draw.rectangle((0, 0, width - 1, height - 1), outline=(40, 40, 40, 255), width=1)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_png)
    return output_png


def save_open3d_overlay_screenshot(
    *,
    overlay_ply: Path,
    output_png: Path,
    width: int = 1400,
    height: int = 1000,
) -> Path:
    """Try to render a PLY preview with Open3D. Raises when Open3D/display is unavailable."""
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(overlay_ply))
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    vis.add_geometry(pcd)
    ctr = vis.get_view_control()
    ctr.set_front([0.45, -0.35, -0.82])
    ctr.set_up([0.0, 1.0, 0.0])
    ctr.set_zoom(0.65)
    vis.poll_events()
    vis.update_renderer()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    vis.capture_screen_image(str(output_png), do_render=True)
    vis.destroy_window()
    return output_png

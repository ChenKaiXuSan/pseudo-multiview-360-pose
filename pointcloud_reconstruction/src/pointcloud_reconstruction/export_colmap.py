from __future__ import annotations

import json
import math
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def load_view_names(manifest_path: Path) -> list[str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [str(view["name"]) for view in manifest.get("views", [])]


@dataclass(frozen=True)
class ColmapDepthPoint:
    point_id: int
    xyz: tuple[float, float, float]
    rgb: tuple[int, int, int]
    image_id: int
    point2d_idx: int
    xy: tuple[float, float]
    error: float = 0.0


def quaternion_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to COLMAP quaternion order (w, x, y, z)."""
    rot = np.asarray(matrix, dtype=np.float64)[:3, :3]
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rot[2, 1] - rot[1, 2]) / s
        qy = (rot[0, 2] - rot[2, 0]) / s
        qz = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        qw = (rot[2, 1] - rot[1, 2]) / s
        qx = 0.25 * s
        qy = (rot[0, 1] + rot[1, 0]) / s
        qz = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        qw = (rot[0, 2] - rot[2, 0]) / s
        qx = (rot[0, 1] + rot[1, 0]) / s
        qy = 0.25 * s
        qz = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        qw = (rot[1, 0] - rot[0, 1]) / s
        qx = (rot[0, 2] + rot[2, 0]) / s
        qy = (rot[1, 2] + rot[2, 1]) / s
        qz = 0.25 * s
    quat = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    return quat / norm if norm > 0 else quat


def matrix_to_colmap_pose(c2w_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert camera-to-world matrix to COLMAP world-to-camera quaternion and translation."""
    w2c = np.linalg.inv(np.asarray(c2w_matrix, dtype=np.float64))
    return quaternion_from_matrix(w2c), w2c[:3, 3]


def write_observation_colmap_text(
    *,
    output_dir: Path,
    image_records: list[dict[str, Any]],
    points: list[ColmapDepthPoint],
) -> None:
    """Write COLMAP images.txt and points3D.txt with real 2D-3D observation links."""
    output_dir.mkdir(parents=True, exist_ok=True)
    points_by_image: dict[int, list[ColmapDepthPoint]] = {}
    for point in points:
        points_by_image.setdefault(point.image_id, []).append(point)

    with (output_dir / "images.txt").open("w", encoding="utf-8") as handle:
        handle.write("# Image list with two lines of data per image:\n")
        handle.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        handle.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        handle.write(f"# Number of images: {len(image_records)}\n")
        for record in image_records:
            quat, trans = matrix_to_colmap_pose(np.asarray(record["pose"], dtype=np.float64))
            qw, qx, qy, qz = quat
            tx, ty, tz = trans
            image_id = int(record["image_id"])
            camera_id = int(record.get("camera_id", 1))
            handle.write(
                f"{image_id} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
                f"{tx:.9f} {ty:.9f} {tz:.9f} {camera_id} {record['name']}\n"
            )
            obs = sorted(points_by_image.get(image_id, []), key=lambda item: item.point2d_idx)
            handle.write(" ".join(f"{p.xy[0]:.6f} {p.xy[1]:.6f} {p.point_id}" for p in obs) + "\n")

    with (output_dir / "points3D.txt").open("w", encoding="utf-8") as handle:
        handle.write("# 3D point list with one line of data per point:\n")
        handle.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        handle.write(f"# Number of points: {len(points)}\n")
        for point in points:
            x, y, z = point.xyz
            r, g, b = point.rgb
            handle.write(
                f"{point.point_id} {x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)} "
                f"{point.error:.6f} {point.image_id} {point.point2d_idx}\n"
            )


def write_pinhole_cameras_txt(output_dir: Path, *, width: int, height: int, intrinsics: np.ndarray) -> None:
    fx, fy, cx, cy = [float(v) for v in intrinsics]
    with (output_dir / "cameras.txt").open("w", encoding="utf-8") as handle:
        handle.write("# Camera list with one line of data per camera:\n")
        handle.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        handle.write("# Number of cameras: 1\n")
        handle.write(f"1 PINHOLE {int(width)} {int(height)} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")


def _read_depth_zip(depth_zip: Path):
    try:
        import Imath
        import OpenEXR
    except ImportError as exc:
        raise RuntimeError("OpenEXR/Imath are required for observation-aware VIPE depth export") from exc
    with zipfile.ZipFile(depth_zip, "r") as archive:
        for name in sorted(archive.namelist()):
            frame_idx = int(Path(name).stem)
            with archive.open(name) as src:
                exr = OpenEXR.InputFile(src)
                header = exr.header()
                dw = header["dataWindow"]
                width = dw.max.x - dw.min.x + 1
                height = dw.max.y - dw.min.y + 1
                channels = exr.channels(["Z"], Imath.PixelType(Imath.PixelType.HALF))
                depth = np.frombuffer(channels[0], dtype=np.float16).reshape((height, width)).astype(np.float32)
                yield frame_idx, depth


def _read_rgb_frames(rgb_path: Path) -> dict[int, np.ndarray]:
    import imageio
    frames = {}
    reader = imageio.get_reader(rgb_path, "ffmpeg")
    for frame_idx, rgb in enumerate(reader):
        frames[frame_idx] = np.asarray(rgb, dtype=np.uint8)
    return frames


def _reliable_depth_mask(depth: np.ndarray) -> np.ndarray:
    return np.isfinite(depth) & (depth > 0.0)


def export_vipe_depth_colmap_with_observations(
    *,
    vipe_result_dir: Path,
    sequence: str,
    output_dir: Path,
    depth_step: int = 16,
    spatial_subsample: int = 4,
) -> dict[str, Any]:
    """Export VIPE depth artifacts to COLMAP text with x/y/POINT3D_ID observations."""
    base = Path(vipe_result_dir)
    pose_npz = np.load(base / "pose" / f"{sequence}.npz")
    intr_npz = np.load(base / "intrinsics" / f"{sequence}.npz")
    poses = pose_npz["data"]
    pose_inds = pose_npz["inds"]
    intrinsics = intr_npz["data"]
    rgb_frames = _read_rgb_frames(base / "rgb" / f"{sequence}.mp4")
    depth_items = list(_read_depth_zip(base / "depth" / f"{sequence}.zip"))
    depth_by_frame = {frame_idx: depth for frame_idx, depth in depth_items}

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    image_records: list[dict[str, Any]] = []
    points: list[ColmapDepthPoint] = []
    point_id = 1

    import imageio

    for image_pos, (pose, frame_idx) in enumerate(zip(poses, pose_inds), start=1):
        frame_idx = int(frame_idx)
        rgb = rgb_frames.get(frame_idx)
        if rgb is None:
            continue
        image_name = f"images/frame_{frame_idx:06d}.jpg"
        imageio.imwrite(output_dir / image_name, rgb)
        image_records.append({"image_id": image_pos, "pose": pose, "camera_id": 1, "name": image_name})
        if (image_pos - 1) % max(int(depth_step), 1) != 0:
            continue
        depth = depth_by_frame.get(frame_idx)
        if depth is None:
            continue
        h, w = depth.shape
        fx, fy, cx, cy = [float(v) for v in intrinsics[image_pos - 1]]
        sampled_v = np.arange(0, h, int(spatial_subsample), dtype=np.int32)
        sampled_u = np.arange(0, w, int(spatial_subsample), dtype=np.int32)
        uu, vv = np.meshgrid(sampled_u, sampled_v)
        z = depth[vv, uu]
        mask = _reliable_depth_mask(z)
        if not np.any(mask):
            continue
        uu = uu[mask].astype(np.float64)
        vv = vv[mask].astype(np.float64)
        z = z[mask].astype(np.float64)
        x = (uu - cx) / fx * z
        y = (vv - cy) / fy * z
        cam_points = np.stack([x, y, z], axis=1)
        world_points = cam_points @ pose[:3, :3].T + pose[:3, 3][None, :]
        rgb_sample = rgb[vv.astype(np.int32), uu.astype(np.int32), :]
        start_idx = len(points)
        for local_idx, (xyz, color, u, v) in enumerate(zip(world_points, rgb_sample, uu, vv)):
            points.append(
                ColmapDepthPoint(
                    point_id=point_id,
                    xyz=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
                    rgb=(int(color[0]), int(color[1]), int(color[2])),
                    image_id=image_pos,
                    point2d_idx=local_idx,
                    xy=(float(u), float(v)),
                )
            )
            point_id += 1

    if not image_records:
        raise ValueError(f"No RGB frames exported from {base}")
    first_rgb = next(iter(rgb_frames.values()))
    write_pinhole_cameras_txt(output_dir, width=first_rgb.shape[1], height=first_rgb.shape[0], intrinsics=intrinsics[0])
    write_observation_colmap_text(output_dir=output_dir, image_records=image_records, points=points)
    summary = {
        "vipe_result_dir": str(vipe_result_dir),
        "sequence": sequence,
        "output_dir": str(output_dir),
        "depth_step": int(depth_step),
        "spatial_subsample": int(spatial_subsample),
        "images": len(image_records),
        "points": len(points),
        "note": "images.txt contains x y POINT3D_ID observations for sampled depth points.",
    }
    (output_dir / "observation_export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_vipe_to_colmap_commands(
    *,
    manifest_path: Path,
    vipe_results_dir: Path,
    colmap_root: Path,
    vipe_repo: Path,
    python_command: str = "python",
    depth_step: int = 16,
    use_slam_map: bool = False,
) -> list[list[str]]:
    """Build commands that export each view directly to colmap_root/<view>."""
    commands: list[list[str]] = []
    script_path = vipe_repo / "scripts" / "vipe_to_colmap.py"
    for view_name in load_view_names(manifest_path):
        command = [
            python_command,
            str(script_path),
            str(vipe_results_dir / view_name),
            "--sequence",
            view_name,
            "--output",
            str(colmap_root),
            "--depth_step",
            str(int(depth_step)),
        ]
        if use_slam_map:
            command.append("--use_slam_map")
        commands.append(command)
    return commands


def flatten_nested_colmap_outputs(colmap_root: Path, view_names: list[str]) -> list[dict[str, str]]:
    """Move colmap/<view>/<view>/* up to colmap/<view> when old nested outputs exist."""
    actions: list[dict[str, str]] = []
    for view_name in view_names:
        view_dir = colmap_root / view_name
        nested_dir = view_dir / view_name
        if not nested_dir.exists() or not nested_dir.is_dir():
            continue
        view_dir.mkdir(parents=True, exist_ok=True)
        for child in sorted(nested_dir.iterdir()):
            target = view_dir / child.name
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite existing COLMAP output: {target}")
            shutil.move(str(child), str(target))
            actions.append({"from": str(child), "to": str(target)})
        nested_dir.rmdir()
    return actions


def export_colmap_views(
    *,
    manifest_path: Path,
    vipe_results_dir: Path,
    colmap_root: Path,
    vipe_repo: Path,
    python_command: str = "python",
    depth_step: int = 16,
    use_slam_map: bool = False,
    dry_run: bool = False,
) -> list[list[str]]:
    commands = build_vipe_to_colmap_commands(
        manifest_path=manifest_path,
        vipe_results_dir=vipe_results_dir,
        colmap_root=colmap_root,
        vipe_repo=vipe_repo,
        python_command=python_command,
        depth_step=depth_step,
        use_slam_map=use_slam_map,
    )
    colmap_root.mkdir(parents=True, exist_ok=True)
    for command in commands:
        print(" ".join(command))
        if not dry_run:
            subprocess.run(command, check=True, cwd=str(vipe_repo))
    if not dry_run:
        flatten_nested_colmap_outputs(colmap_root, load_view_names(manifest_path))
    return commands

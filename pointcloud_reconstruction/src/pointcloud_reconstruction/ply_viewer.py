from __future__ import annotations

import re
import socket
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PlyPointCloud:
    path: Path
    name: str
    xyz: np.ndarray
    rgb: np.ndarray
    total_points: int


def _parse_ascii_ply_header(path: Path) -> tuple[int, list[str], int]:
    vertex_count: int | None = None
    vertex_properties: list[str] = []
    in_vertex_element = False
    header_lines = 0
    saw_ascii_format = False

    with path.open("r", encoding="utf-8") as handle:
        first = handle.readline()
        header_lines += 1
        if first.strip() != "ply":
            raise ValueError(f"Not a PLY file: {path}")

        for line in handle:
            header_lines += 1
            stripped = line.strip()
            if stripped == "format ascii 1.0":
                saw_ascii_format = True
            elif stripped.startswith("format "):
                raise ValueError(f"Only ASCII PLY files are supported: {path}")
            elif stripped.startswith("element "):
                parts = stripped.split()
                in_vertex_element = len(parts) >= 3 and parts[1] == "vertex"
                if in_vertex_element:
                    vertex_count = int(parts[2])
                    vertex_properties = []
            elif in_vertex_element and stripped.startswith("property "):
                parts = stripped.split()
                if len(parts) >= 3:
                    vertex_properties.append(parts[-1])
            elif stripped == "end_header":
                break

    if not saw_ascii_format:
        raise ValueError(f"Only ASCII PLY files are supported: {path}")
    if vertex_count is None:
        raise ValueError(f"PLY header has no vertex count: {path}")

    return vertex_count, vertex_properties, header_lines


def _property_index(properties: list[str], *names: str) -> int | None:
    for name in names:
        if name in properties:
            return properties.index(name)
    return None


def read_ascii_ply_xyzrgb(path: Path, max_points: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Read xyz and optional rgb columns from an ASCII PLY file."""
    vertex_count, properties, header_lines = _parse_ascii_ply_header(path)
    x_idx = _property_index(properties, "x")
    y_idx = _property_index(properties, "y")
    z_idx = _property_index(properties, "z")
    if x_idx is None or y_idx is None or z_idx is None:
        raise ValueError(f"PLY vertex properties must include x/y/z: {path}")

    r_idx = _property_index(properties, "red", "r")
    g_idx = _property_index(properties, "green", "g")
    b_idx = _property_index(properties, "blue", "b")
    has_rgb = r_idx is not None and g_idx is not None and b_idx is not None

    if max_points is not None and max_points > 0 and vertex_count > max_points:
        sample_indices = np.linspace(0, vertex_count - 1, int(max_points), dtype=np.int64)
        sample_lookup = set(int(idx) for idx in sample_indices)
        target_count = len(sample_indices)
    else:
        sample_lookup = None
        target_count = vertex_count

    xyz = np.zeros((target_count, 3), dtype=np.float64)
    rgb = np.full((target_count, 3), 255, dtype=np.uint8)
    used = 0

    with path.open("r", encoding="utf-8") as handle:
        for _ in range(header_lines):
            next(handle)
        for row_idx, line in enumerate(handle):
            if row_idx >= vertex_count or used >= target_count:
                break
            if sample_lookup is not None and row_idx not in sample_lookup:
                continue
            parts = line.split()
            if len(parts) < len(properties):
                continue
            xyz[used] = [float(parts[x_idx]), float(parts[y_idx]), float(parts[z_idx])]
            if has_rgb:
                rgb[used] = [
                    np.clip(int(float(parts[r_idx])), 0, 255),
                    np.clip(int(float(parts[g_idx])), 0, 255),
                    np.clip(int(float(parts[b_idx])), 0, 255),
                ]
            used += 1

    return xyz[:used], rgb[:used]


def sample_evenly_spaced(xyz: np.ndarray, rgb: np.ndarray, max_points: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically sample at most max_points while preserving the cloud span."""
    if max_points is None or max_points <= 0 or len(xyz) <= max_points:
        return xyz, rgb
    indices = np.linspace(0, len(xyz) - 1, int(max_points), dtype=np.int64)
    return xyz[indices], rgb[indices]


def load_ply_point_cloud(path: Path, *, max_points: int | None = None, name: str | None = None) -> PlyPointCloud:
    total_points, _, _ = _parse_ascii_ply_header(path)
    xyz, rgb = read_ascii_ply_xyzrgb(path, max_points=max_points)
    return PlyPointCloud(
        path=path,
        name=name or path.stem,
        xyz=xyz,
        rgb=rgb,
        total_points=total_points,
    )


def frame_number_from_ply_path(path: Path) -> int:
    match = re.search(r"frame_(\d+)", path.stem)
    if match is None:
        raise ValueError(f"PLY filename does not contain frame number: {path}")
    return int(match.group(1))


def discover_frame_plys(frame_dir: Path, *, frame_stride: int = 1, max_frames: int | None = None) -> list[Path]:
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    paths = sorted(frame_dir.glob("frame_*.ply"), key=frame_number_from_ply_path)
    if not paths:
        raise ValueError(f"No frame_*.ply files found in: {frame_dir}")
    paths = paths[::frame_stride]
    if max_frames is not None and max_frames > 0:
        paths = paths[:max_frames]
    return paths


def get_host_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def run_ply_viewer(
    ply_paths: list[Path],
    *,
    names: list[str] | None = None,
    port: int = 20542,
    host: str | None = None,
    max_points: int | None = 200000,
    point_size: float = 0.01,
) -> None:
    """Start a viser server for one or more fused ASCII PLY point clouds."""
    if not ply_paths:
        raise ValueError("at least one --ply path is required")
    if names is not None and len(names) != len(ply_paths):
        raise ValueError("--name must be provided the same number of times as --ply")

    import viser

    clouds = [
        load_ply_point_cloud(path, max_points=max_points, name=None if names is None else names[idx])
        for idx, path in enumerate(ply_paths)
    ]
    server_host = host or get_host_ip()
    server = viser.ViserServer(host=server_host, port=port, verbose=False)
    handles = {}

    for cloud in clouds:
        handles[cloud.name] = server.scene.add_point_cloud(
            name=f"/point_clouds/{cloud.name}",
            points=cloud.xyz,
            colors=cloud.rgb,
            point_size=point_size,
            point_shape="rounded",
        )

    with server.gui.add_folder("PLY layers"):
        for cloud in clouds:
            label = f"{cloud.name} ({len(cloud.xyz)}/{cloud.total_points})"
            checkbox = server.gui.add_checkbox(label, initial_value=True)

            @checkbox.on_update
            def _(event, cloud_name: str = cloud.name) -> None:
                handles[cloud_name].visible = bool(event.target.value)

    print(f"Loaded {len(clouds)} PLY file(s).", flush=True)
    for cloud in clouds:
        print(f"- {cloud.name}: {len(cloud.xyz)} shown / {cloud.total_points} total points", flush=True)
        print(f"  {cloud.path}", flush=True)
    print(f"Viewer: http://{server_host}:{port}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)

    try:
        while True:
            time.sleep(10.0)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()

def run_ply_sequence_viewer(
    frame_dir: Path,
    *,
    port: int = 20542,
    host: str | None = None,
    max_points: int | None = 100000,
    point_size: float = 0.01,
    frame_stride: int = 1,
    max_frames: int | None = None,
    fps: float = 6.0,
) -> None:
    """Start a viser server that plays frame_*.ply files as a timeline."""
    import viser

    frame_paths = discover_frame_plys(frame_dir, frame_stride=frame_stride, max_frames=max_frames)
    clouds = [
        load_ply_point_cloud(path, max_points=max_points, name=path.stem)
        for path in frame_paths
    ]
    server_host = host or get_host_ip()
    server = viser.ViserServer(host=server_host, port=port, verbose=False)
    handles = []

    for idx, cloud in enumerate(clouds):
        handle = server.scene.add_point_cloud(
            name=f"/sequence/{cloud.name}",
            points=cloud.xyz,
            colors=cloud.rgb,
            point_size=point_size,
            point_shape="rounded",
        )
        handle.visible = idx == 0
        handles.append(handle)

    current_index = 0

    def show_frame(index: int) -> None:
        nonlocal current_index
        index = int(max(0, min(index, len(handles) - 1)))
        current_index = index
        for handle_idx, handle in enumerate(handles):
            handle.visible = handle_idx == index

    with server.gui.add_folder("Timeline"):
        slider = server.gui.add_slider("Frame", min=0, max=len(clouds) - 1, step=1, initial_value=0)
        play = server.gui.add_checkbox("Play", initial_value=False)
        fps_slider = server.gui.add_slider("FPS", min=1.0, max=30.0, step=1.0, initial_value=float(fps))
        frame_label = server.gui.add_text("Current", initial_value=clouds[0].name, disabled=True)

        @slider.on_update
        def _(event) -> None:
            index = int(event.target.value)
            show_frame(index)
            frame_label.value = clouds[index].name

    print(f"Loaded {len(clouds)} sequence frame(s) from {frame_dir}.", flush=True)
    for cloud in clouds:
        print(f"- {cloud.name}: {len(cloud.xyz)} shown / {cloud.total_points} total points", flush=True)
    print(f"Viewer: http://{server_host}:{port}", flush=True)
    print("Use the Timeline slider or Play checkbox. Press Ctrl+C to stop.", flush=True)

    last_step = time.monotonic()
    try:
        while True:
            time.sleep(0.02)
            if not bool(play.value):
                continue
            interval = 1.0 / max(float(fps_slider.value), 1.0)
            now = time.monotonic()
            if now - last_step < interval:
                continue
            next_index = (current_index + 1) % len(clouds)
            slider.value = next_index
            show_frame(next_index)
            frame_label.value = clouds[next_index].name
            last_step = now
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


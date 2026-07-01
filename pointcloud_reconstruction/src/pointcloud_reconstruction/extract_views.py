from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from .bbox_views import build_anchor_views, infer_anchor_yaw, save_view_manifest
from .projection import camera_to_world_matrix, equirectangular_to_perspective


def parse_yaw_offsets(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def extract_bbox_guided_views(
    *,
    video_path: Path,
    bbox_json_path: Path,
    output_dir: Path,
    target_id: int = 1,
    view_size: int = 1024,
    fov_deg: float = 100.0,
    yaw_offsets: Sequence[float] = (0.0, 60.0, 120.0, 180.0, -120.0, -60.0),
    max_frames: int | None = None,
    every_n: int = 1,
) -> dict:
    """Extract fixed perspective videos whose yaw layout is anchored by the selfie bbox."""
    import cv2

    if every_n <= 0:
        raise ValueError("every_n must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    views_dir = output_dir / "views"
    views_dir.mkdir(parents=True, exist_ok=True)

    anchor_yaw = infer_anchor_yaw(bbox_json_path, target_id=target_id)
    views = build_anchor_views(anchor_yaw, yaw_offsets=yaw_offsets, fov_deg=fov_deg, view_size=view_size)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    output_fps = source_fps / float(every_n)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writers = {}
    view_video_paths = {}
    try:
        for view in views:
            path = views_dir / f"{view.name}.mp4"
            view_video_paths[view.name] = path
            writers[view.name] = cv2.VideoWriter(str(path), fourcc, output_fps, (view.size, view.size))

        frame_index = 0
        written = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % every_n != 0:
                frame_index += 1
                continue
            if max_frames is not None and written >= max_frames:
                break
            for view in views:
                projected = equirectangular_to_perspective(
                    frame,
                    view_size=view.size,
                    yaw_deg=view.yaw_deg,
                    pitch_deg=view.pitch_deg,
                    fov_deg=view.fov_deg,
                )
                writers[view.name].write(projected)
            written += 1
            if written % 25 == 0:
                print(f"Extracted {written} frames per view")
            frame_index += 1
    finally:
        cap.release()
        for writer in writers.values():
            writer.release()

    manifest_path = output_dir / "view_manifest.json"
    save_view_manifest(
        manifest_path,
        source_video=video_path,
        bbox_json=bbox_json_path,
        anchor_yaw=anchor_yaw,
        views=views,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for view in manifest["views"]:
        view["video_path"] = str(view_video_paths[view["name"]])
        view["camera_to_world"] = camera_to_world_matrix(view["yaw_deg"], view["pitch_deg"]).tolist()
    manifest["frames_written_per_view"] = int(written)
    manifest["every_n"] = int(every_n)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest

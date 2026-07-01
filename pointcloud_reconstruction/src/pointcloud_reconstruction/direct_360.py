from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def limited_video_path(output_root: Path, max_frames: int) -> Path:
    return output_root / "input" / f"direct_360_first_{int(max_frames):06d}.mp4"


def clip_video_first_frames(
    *,
    video_path: Path,
    output_path: Path,
    max_frames: int,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write a cached video containing only the first max_frames frames."""
    import cv2

    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    if output_path.exists() and not overwrite:
        return {"status": "cached", "output_path": str(output_path), "frames_written": None, "max_frames": int(max_frames)}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Cannot read video dimensions: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    written = 0
    try:
        while written < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
            if written % 25 == 0:
                print(f"Clipped {written} frames for direct-360 VIPE input")
    finally:
        cap.release()
        writer.release()
    return {
        "status": "written",
        "source_video_path": str(video_path),
        "output_path": str(output_path),
        "frames_written": int(written),
        "max_frames": int(max_frames),
        "fps": fps,
        "width": width,
        "height": height,
    }


def build_direct_360_vipe_command(
    *,
    video_path: Path,
    output_dir: Path,
    vipe_command: str = "vipe",
    pipeline: str = "default",
    visualize: bool = False,
) -> list[str]:
    command = [
        vipe_command,
        "infer",
        str(video_path),
        "--output",
        str(output_dir),
        "--pipeline",
        pipeline,
    ]
    if visualize:
        command.append("--visualize")
    return command


def build_direct_360_colmap_command(
    *,
    vipe_results_dir: Path,
    colmap_root: Path,
    vipe_repo: Path,
    python_command: str = "python",
    sequence_name: str = "direct_360",
    depth_step: int = 16,
    use_slam_map: bool = False,
) -> list[str]:
    command = [
        python_command,
        str(vipe_repo / "scripts" / "vipe_to_colmap.py"),
        str(vipe_results_dir),
        "--sequence",
        sequence_name,
        "--output",
        str(colmap_root),
        "--depth_step",
        str(int(depth_step)),
    ]
    if use_slam_map:
        command.append("--use_slam_map")
    return command


def flatten_direct_360_colmap_output(colmap_root: Path, sequence_name: str = "direct_360") -> list[dict[str, str]]:
    """Move colmap/direct_360/* up to colmap/* if VIPE creates a nested sequence folder."""
    nested = colmap_root / sequence_name
    actions: list[dict[str, str]] = []
    if not nested.exists() or not nested.is_dir():
        return actions
    colmap_root.mkdir(parents=True, exist_ok=True)
    for child in sorted(nested.iterdir()):
        target = colmap_root / child.name
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite existing direct-360 COLMAP output: {target}")
        child.rename(target)
        actions.append({"from": str(child), "to": str(target)})
    nested.rmdir()
    return actions


def write_direct_360_summary(
    *,
    summary_path: Path,
    source_video_path: Path,
    vipe_input_video_path: Path,
    output_root: Path,
    vipe_results_dir: Path,
    colmap_root: Path,
    vipe_command: list[str],
    colmap_command: list[str] | None,
    ran_vipe: bool,
    ran_colmap: bool,
    flatten_actions: list[dict[str, str]] | None = None,
    max_frames: int | None = None,
    clip_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = {
        "mode": "direct_360_vipe_baseline",
        "source_video_path": str(source_video_path),
        "vipe_input_video_path": str(vipe_input_video_path),
        "max_frames": int(max_frames) if max_frames is not None else None,
        "clip_info": clip_info,
        "output_root": str(output_root),
        "vipe_results_dir": str(vipe_results_dir),
        "colmap_root": str(colmap_root),
        "vipe_command": vipe_command,
        "colmap_command": colmap_command,
        "ran_vipe": bool(ran_vipe),
        "ran_colmap": bool(ran_colmap),
        "flatten_actions": flatten_actions or [],
        "notes": [
            "This baseline sends the original equirectangular 360 video directly to VIPE.",
            "When max_frames is set, VIPE receives a cached clip containing only the first frames.",
            "Compare it against bbox-guided perspective-view VIPE fusion outputs.",
        ],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_direct_360_baseline(
    *,
    video_path: Path,
    output_root: Path,
    vipe_command: str = "vipe",
    pipeline: str = "default",
    visualize: bool = False,
    export_colmap: bool = False,
    vipe_repo: Path = Path("/mnt/dataset/skiing/vipe/vipe"),
    python_command: str = "python",
    sequence_name: str = "direct_360",
    depth_step: int = 16,
    use_slam_map: bool = False,
    max_frames: int | None = None,
    overwrite_clip: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    vipe_results_dir = output_root / "vipe_results"
    colmap_root = output_root / "colmap"
    summary_path = output_root / "direct_360_summary.json"
    vipe_input_video_path = video_path
    clip_info = None
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")
        vipe_input_video_path = limited_video_path(output_root, max_frames=max_frames)
        if dry_run:
            clip_info = {"status": "dry_run", "output_path": str(vipe_input_video_path), "max_frames": int(max_frames)}
        else:
            clip_info = clip_video_first_frames(
                video_path=video_path,
                output_path=vipe_input_video_path,
                max_frames=max_frames,
                overwrite=overwrite_clip,
            )
    vipe_cmd = build_direct_360_vipe_command(
        video_path=vipe_input_video_path,
        output_dir=vipe_results_dir,
        vipe_command=vipe_command,
        pipeline=pipeline,
        visualize=visualize,
    )
    colmap_cmd = None
    if export_colmap:
        colmap_cmd = build_direct_360_colmap_command(
            vipe_results_dir=vipe_results_dir,
            colmap_root=colmap_root,
            vipe_repo=vipe_repo,
            python_command=python_command,
            sequence_name=sequence_name,
            depth_step=depth_step,
            use_slam_map=use_slam_map,
        )

    output_root.mkdir(parents=True, exist_ok=True)
    vipe_results_dir.mkdir(parents=True, exist_ok=True)
    print(" ".join(vipe_cmd))
    ran_vipe = False
    ran_colmap = False
    flatten_actions: list[dict[str, str]] = []
    if not dry_run:
        subprocess.run(vipe_cmd, check=True)
        ran_vipe = True
    if colmap_cmd is not None:
        colmap_root.mkdir(parents=True, exist_ok=True)
        print(" ".join(colmap_cmd))
        if not dry_run:
            subprocess.run(colmap_cmd, check=True, cwd=str(vipe_repo))
            ran_colmap = True
            flatten_actions = flatten_direct_360_colmap_output(colmap_root, sequence_name=sequence_name)

    return write_direct_360_summary(
        summary_path=summary_path,
        source_video_path=video_path,
        vipe_input_video_path=vipe_input_video_path,
        output_root=output_root,
        vipe_results_dir=vipe_results_dir,
        colmap_root=colmap_root,
        vipe_command=vipe_cmd,
        colmap_command=colmap_cmd,
        ran_vipe=ran_vipe,
        ran_colmap=ran_colmap,
        flatten_actions=flatten_actions,
        max_frames=max_frames,
        clip_info=clip_info,
    )

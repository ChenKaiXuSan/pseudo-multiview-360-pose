#!/usr/bin/env python3
"""Render resized-XYZ direct-360 camera 3D plots for existing frame outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sam3d_body_multiview_fusion import (
    finite_keypoint_mask,
    keypoints3d_to_plot_coords,
    load_mhr70_visual_style,
    set_axes_equal,
    track_color_rgb01,
)


DEFAULT_OUTPUT_DIR = Path("/mnt/dataset/skiing/raw_new/sam3d_body_360_direct_compare")


def collect_frame_tracks(frame_dir: Path, min_conf: float) -> list[tuple[int, np.ndarray, np.ndarray]]:
    tracks = []
    for result_path in sorted(frame_dir.glob("track_*/direct_360_result.json")):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        rows = payload.get("keypoints3d_camera")
        if not rows:
            continue
        kpts = np.asarray(rows, dtype=np.float64)
        if kpts.ndim != 2 or kpts.shape[1] < 3:
            continue
        if kpts.shape[1] == 3:
            conf = np.ones((len(kpts), 1), dtype=np.float64)
            kpts = np.concatenate([kpts[:, :3], conf], axis=1)
        else:
            kpts = kpts[:, :4]
        mask = finite_keypoint_mask(kpts, min_conf)
        if np.any(mask):
            track_id = int(payload.get("track_id", len(tracks) + 1))
            tracks.append((track_id, kpts, mask))
    return tracks


def render_frame(
    frame_dir: Path,
    edges: list[tuple[int, int]],
    min_conf: float,
    output_name: str,
    point_size: float,
    xyz_scale: tuple[float, float, float],
) -> Path | None:
    tracks = collect_frame_tracks(frame_dir, min_conf)
    if not tracks:
        return None

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")
    all_pts = []
    axis_labels = ("camera X right", "camera Z depth", "camera -Y up")
    view_angles = (14, -70)

    frame_label = frame_dir.name.replace("frame_", "")
    scale = np.asarray(xyz_scale, dtype=np.float64).reshape(1, 3)
    for track_id, kpts, mask in tracks:
        plot_all, axis_labels, view_angles = keypoints3d_to_plot_coords(kpts, "camera")
        plot_all = plot_all * scale
        pts = plot_all[mask]
        color = track_color_rgb01(track_id)
        label = f"person {track_id:04d}"
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=[color], s=point_size, depthshade=True, label=label)
        for a, b in edges:
            if a < len(kpts) and b < len(kpts) and mask[a] and mask[b]:
                seg = plot_all[[a, b], :3]
                ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=color, linewidth=1.3, alpha=0.9)
        all_pts.append(pts)

    ax.set_title(f"frame {frame_label} direct 360 camera 3D - XYZ resized")
    ax.set_xlabel(f"{axis_labels[0]} x{xyz_scale[0]:g}")
    ax.set_ylabel(f"{axis_labels[1]} x{xyz_scale[1]:g}")
    ax.set_zlabel(f"{axis_labels[2]} x{xyz_scale[2]:g}")
    ax.view_init(elev=view_angles[0], azim=view_angles[1])
    set_axes_equal(ax, np.concatenate(all_pts, axis=0))
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()

    output_path = frame_dir / output_name
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default="direct_360_kpts3d_camera_xyz_resized.png")
    parser.add_argument("--min-conf", type=float, default=0.0)
    parser.add_argument("--point-size", type=float, default=10.0)
    parser.add_argument("--x-scale", type=float, default=1.8)
    parser.add_argument("--z-depth-scale", type=float, default=1.8)
    parser.add_argument("--y-up-scale", type=float, default=0.7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    style = load_mhr70_visual_style("")
    frame_dirs = sorted(path for path in args.output_dir.glob("frame_*") if path.is_dir())
    saved = []
    for frame_dir in frame_dirs:
        output_path = render_frame(
            frame_dir,
            style["edges"],
            args.min_conf,
            args.output_name,
            args.point_size,
            (args.x_scale, args.z_depth_scale, args.y_up_scale),
        )
        if output_path is not None:
            saved.append(output_path)
            print(f"saved {output_path}")
        else:
            print(f"skip {frame_dir}: no valid direct 360 keypoints")
    print(f"done: saved {len(saved)}/{len(frame_dirs)} frames")


if __name__ == "__main__":
    main()

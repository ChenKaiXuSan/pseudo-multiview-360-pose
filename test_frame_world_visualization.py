#!/usr/bin/env python3
"""Smoke tests for frame-level multi-track world visualization."""

from __future__ import annotations

import json
import io
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import cv2
import numpy as np

from sam3d_body_multiview_fusion import (
    CONFIG,
    collect_frame_world_tracks,
    draw_frame_tracks_overlay,
    draw_frame_tracks_world_axis,
    parse_args,
    resolve_sam3d_execution,
    save_frame_tracks_world_visualization,
    save_track_fused_summary_visualization,
    load_mhr70_visual_style,
    track_color_rgb01,
    world_keypoints_to_equirectangular_pixels,
)


def make_track(frame_dir: Path, track_id: int, offset: float) -> None:
    track_dir = frame_dir / f"track_{track_id:04d}"
    track_dir.mkdir(parents=True)
    kpts = [
        [offset + 0.0, 0.0, 0.0, 1.0],
        [offset + 0.2, 0.3, 0.1, 1.0],
        [offset - 0.2, 0.3, 0.1, 1.0],
    ]
    payload = {
        "frame_number": 92,
        "track_id": track_id,
        "fused_keypoints3d_world": kpts,
    }
    (track_dir / "fused_keypoints3d.json").write_text(json.dumps(payload), encoding="utf-8")


def test_load_mhr70_visual_style_from_repo_path_without_package_deps() -> None:
    repo = Path(__file__).resolve().parents[1] / "third_party" / "sam-3d-body"
    if not repo.exists():
        return

    style = load_mhr70_visual_style(str(repo))

    assert len(style["edges"]) > 0
    assert style["point_colors"] is not None


def test_cli_defaults_to_official_sam3d_without_run_flag() -> None:
    args = parse_args([])

    assert not hasattr(args, "run_sam3d")
    assert args.no_run_sam3d is False
    assert "360PoseFusion/third_party/sam-3d-body" in str(CONFIG["sam3d_repo"])


def test_no_run_sam3d_disables_direct_and_command_runners() -> None:
    args = parse_args(["--no-run-sam3d", "--sam3d-command", "cmd {image}"])

    use_direct, command = resolve_sam3d_execution(args)

    assert use_direct is False
    assert command is None


def test_collect_frame_world_tracks_reads_sorted_tracks() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        frame_dir = Path(tmp) / "frame_000092"
        make_track(frame_dir, 2, 1.0)
        make_track(frame_dir, 1, -1.0)

        tracks = collect_frame_world_tracks(frame_dir)

    assert [track["track_id"] for track in tracks] == [1, 2]
    assert tracks[0]["keypoints"].shape == (3, 4)


def test_world_keypoints_project_to_equirectangular_pixels() -> None:
    kpts = np.array([
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
    ], dtype=np.float64)

    pixels = world_keypoints_to_equirectangular_pixels(kpts, width=400, height=200)

    assert np.allclose(pixels[0], [200.0, 100.0])
    assert np.allclose(pixels[1], [300.0, 100.0])
    assert np.allclose(pixels[2], [200.0, 0.0])


def test_save_frame_tracks_world_visualization_writes_png() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        frame_dir = root / "frame_000092"
        make_track(frame_dir, 1, 0.0)
        make_track(frame_dir, 2, 1.0)
        frame = np.zeros((80, 160, 3), dtype=np.uint8)
        frame[:, :80] = (20, 80, 180)
        frame[:, 80:] = (180, 120, 20)
        output_path = frame_dir / "frame_tracks_world.png"

        saved = save_frame_tracks_world_visualization(
            frame,
            collect_frame_world_tracks(frame_dir),
            output_path,
            edges=[(0, 1), (0, 2)],
            edge_colors=[np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])],
            point_colors=None,
            show_indices=False,
            min_conf=0.0,
        )

        image = cv2.imread(str(output_path))

    assert saved is not None
    assert image is not None
    assert image.size > 0


def test_visualization_writes_metadata_and_uses_stable_track_colors() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        frame_dir = root / "frame_000092"
        make_track(frame_dir, 5, 1.0)
        make_track(frame_dir, 1, -1.0)
        frame = np.zeros((80, 160, 3), dtype=np.uint8)
        output_path = frame_dir / "frame_tracks_world.png"
        tracks = collect_frame_world_tracks(frame_dir)

        save_frame_tracks_world_visualization(
            frame,
            tracks,
            output_path,
            edges=[(0, 1), (0, 2)],
            edge_colors=[np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])],
            point_colors=None,
            show_indices=False,
            min_conf=0.0,
            frame_title="frame 000092 original 360",
        )

        metadata = json.loads(output_path.with_suffix(".json").read_text(encoding="utf-8"))

    assert np.allclose(track_color_rgb01(1), track_color_rgb01(1))
    assert not np.allclose(track_color_rgb01(1), track_color_rgb01(5))
    assert metadata["frame_title"] == "frame 000092 original 360"
    assert metadata["tracks"] == [1, 5]
    assert metadata["plot_views"] == ["original_360_with_kpts", "world_3d", "world_xz_topdown"]
    assert len(metadata["source_files"]) == 2


def test_frame_track_axes_use_skeleton_style_colors() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb

    track = {
        "track_id": 3,
        "keypoints": np.array([
            [0.0, 0.0, 1.0, 1.0],
            [0.1, 0.2, 1.0, 1.0],
            [-0.1, 0.2, 1.0, 1.0],
        ], dtype=np.float64),
    }
    edges = [(0, 1), (0, 2)]
    edge_colors = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])]
    point_colors = np.array([
        [0.2, 0.3, 0.4],
        [0.4, 0.5, 0.6],
        [0.6, 0.7, 0.8],
    ], dtype=np.float64)

    fig = plt.figure()
    overlay_ax = fig.add_subplot(121)
    draw_frame_tracks_overlay(
        overlay_ax,
        [track],
        edges,
        edge_colors,
        point_colors,
        width=400,
        height=200,
        min_conf=0.0,
    )
    world_ax = fig.add_subplot(122, projection="3d")
    draw_frame_tracks_world_axis(
        world_ax,
        [track],
        edges,
        edge_colors,
        point_colors,
        show_indices=False,
        min_conf=0.0,
    )

    overlay_line_colors = [to_rgb(line.get_color()) for line in overlay_ax.lines[:2]]
    world_line_colors = [to_rgb(line.get_color()) for line in world_ax.lines[:2]]
    overlay_point_colors = overlay_ax.collections[0].get_facecolors()[:, :3]
    world_point_colors = world_ax.collections[0].get_facecolors()[:, :3]
    plt.close(fig)

    assert np.allclose(overlay_line_colors, edge_colors)
    assert np.allclose(world_line_colors, edge_colors)
    assert np.allclose(overlay_point_colors, point_colors)
    assert np.allclose(world_point_colors, point_colors)


def test_track_fused_summary_visualization_writes_image_and_metadata() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        person_dir = root / "frame_000107" / "track_0009"
        view_dir = person_dir / "views" / "view_00"
        view_dir.mkdir(parents=True)
        image = np.full((48, 64, 3), 180, dtype=np.uint8)
        cv2.imwrite(str(view_dir / "frame_kpts2d.jpg"), image)
        fused = np.array([
            [0.0, 0.0, 1.0, 1.0],
            [0.1, 0.2, 1.0, 1.0],
            [-0.1, 0.2, 1.0, 1.0],
        ], dtype=np.float64)
        views = [{
            "view_index": 0,
            "yaw_offset_deg": 0.0,
            "pitch_offset_deg": -16.0,
            "image_path": str(view_dir / "frame.jpg"),
            "vis_path": str(view_dir / "frame_kpts2d.jpg"),
            "kpts2d_vis_path": str(view_dir / "frame_kpts2d.jpg"),
        }]

        saved = save_track_fused_summary_visualization(
            person_dir,
            views,
            fused,
            edges=[(0, 1), (0, 2)],
            edge_colors=[np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])],
            point_colors=None,
            show_indices=False,
            min_conf=0.0,
            title="frame 000107 track 0009",
        )
        metadata_path = person_dir / "fused" / "fused_views_summary.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        assert saved is not None
        assert (person_dir / "fused" / "fused_views_summary.png").exists()
        assert metadata["title"] == "frame 000107 track 0009"
        assert metadata["views"] == [0]
        assert metadata["plot_views"] == ["perspective_frames", "fused_world_3d"]


def test_verbose_frame_visualization_reports_progress() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        frame_dir = root / "frame_000092"
        make_track(frame_dir, 1, 0.0)
        frame = np.zeros((80, 160, 3), dtype=np.uint8)
        output_path = frame_dir / "frame_tracks_world.png"
        buf = io.StringIO()

        with redirect_stdout(buf):
            tracks = collect_frame_world_tracks(frame_dir, verbose=True)
            save_frame_tracks_world_visualization(
                frame,
                tracks,
                output_path,
                edges=[(0, 1), (0, 2)],
                edge_colors=[np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])],
                point_colors=None,
                show_indices=False,
                min_conf=0.0,
                verbose=True,
            )

    output = buf.getvalue()
    assert "[frame-vis] scan tracks:" in output
    assert "loaded track 0001" in output
    assert "[frame-vis] saved combined world view:" in output


if __name__ == "__main__":
    test_load_mhr70_visual_style_from_repo_path_without_package_deps()
    test_cli_defaults_to_official_sam3d_without_run_flag()
    test_no_run_sam3d_disables_direct_and_command_runners()
    test_collect_frame_world_tracks_reads_sorted_tracks()
    test_world_keypoints_project_to_equirectangular_pixels()
    test_save_frame_tracks_world_visualization_writes_png()
    test_visualization_writes_metadata_and_uses_stable_track_colors()
    test_frame_track_axes_use_skeleton_style_colors()
    test_track_fused_summary_visualization_writes_image_and_metadata()
    test_verbose_frame_visualization_reports_progress()

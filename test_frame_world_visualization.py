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
    collect_frame_world_tracks,
    save_frame_tracks_world_visualization,
    track_color_rgb01,
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


def test_collect_frame_world_tracks_reads_sorted_tracks() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        frame_dir = Path(tmp) / "frame_000092"
        make_track(frame_dir, 2, 1.0)
        make_track(frame_dir, 1, -1.0)

        tracks = collect_frame_world_tracks(frame_dir)

    assert [track["track_id"] for track in tracks] == [1, 2]
    assert tracks[0]["keypoints"].shape == (3, 4)


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
    assert metadata["plot_views"] == ["original_360", "world_3d", "world_xz_topdown"]
    assert len(metadata["source_files"]) == 2


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
    test_collect_frame_world_tracks_reads_sorted_tracks()
    test_save_frame_tracks_world_visualization_writes_png()
    test_visualization_writes_metadata_and_uses_stable_track_colors()
    test_verbose_frame_visualization_reports_progress()

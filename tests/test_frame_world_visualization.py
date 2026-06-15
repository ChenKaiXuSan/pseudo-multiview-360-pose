#!/usr/bin/env python3
"""Smoke tests for frame-level multi-track world visualization."""

from __future__ import annotations

import json
import io
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from sam3d_body_multiview_fusion import (
    CONFIG,
    collect_frame_world_tracks,
    draw_frame_tracks_overlay,
    draw_frame_tracks_world_axis,
    parse_args,
    expand_sam3d_runner_devices,
    resolve_sam3d_execution,
    resolve_sam3d_devices,
    resolve_sam3d_result_output_dir,
    resolve_video_output_dir,
    run_sam3d_for_view,
    run_sam3d_for_views,
    save_fused_keypoints_npz,
    save_frame_tracks_world_visualization,
    save_sam3d_payload_npz,
    save_track_fused_summary_visualization,
    load_mhr70_visual_style,
    person_center_to_lon_lat,
    track_color_rgb01,
    write_person_result,
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


def test_output_root_resolves_to_video_named_subdirectory() -> None:
    output_dir = resolve_video_output_dir(
        Path("/tmp/sam3d_body_multiview"),
        Path("/mnt/dataset/skiing/360test/kimura2_360.mp4"),
    )

    assert output_dir == Path("/tmp/sam3d_body_multiview/kimura2_360")


def test_sam3d_results_are_stored_in_video_level_results_folder() -> None:
    person_dir = Path("/tmp/sam3d_body_multiview/kimura2_360/frame_000123/track_0007")

    result_dir = resolve_sam3d_result_output_dir(person_dir, 3)

    assert result_dir == Path(
        "/tmp/sam3d_body_multiview/kimura2_360/sam3d_results/frame_000123/track_0007/view_03"
    )


def test_cli_exposes_device_pool_without_legacy_single_device_flag() -> None:
    args = parse_args([])

    assert args.sam3d_devices == "auto"
    assert not hasattr(args, "sam3d_device")


def test_auto_sam3d_devices_use_all_available_cuda_devices() -> None:
    assert resolve_sam3d_devices("auto", cuda_available=True, cuda_count=2) == ["cuda:0", "cuda:1"]
    assert resolve_sam3d_devices("auto", cuda_available=True, cuda_count=1) == ["cuda:0"]
    assert resolve_sam3d_devices("auto", cuda_available=False, cuda_count=0) == ["cpu"]


def test_expand_sam3d_runner_devices_respects_estimators_per_device_and_cap() -> None:
    assert expand_sam3d_runner_devices(["cuda:0", "cuda:1"], 2, 8) == [
        "cuda:0",
        "cuda:0",
        "cuda:1",
        "cuda:1",
    ]
    assert expand_sam3d_runner_devices(["cuda:0", "cuda:1"], 2, 3) == [
        "cuda:0",
        "cuda:0",
        "cuda:1",
    ]
    assert expand_sam3d_runner_devices(["cuda:0", "cuda:1"], 0, 8) == ["cuda:0", "cuda:1"]


def test_sam3d_view_pool_uses_one_runner_per_worker_and_preserves_order() -> None:
    class RecordingRunner:
        def __init__(self, name):
            self.name = name
            self.calls = []

        def run(self, image_path, bbox_xyxy, output_json_path):
            self.calls.append(Path(image_path).name)
            return np.array([[float(self.name), 0.0, 0.0, 1.0]], dtype=np.float64)

    views = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for idx in range(4):
            view_dir = root / f"view_{idx:02d}"
            view_dir.mkdir()
            views.append({
                "view_index": idx,
                "image_path": str(view_dir / f"frame_{idx}.jpg"),
                "bbox_json_path": str(view_dir / "bbox.json"),
                "sam3d_output_path": str(view_dir / "sam3d.json"),
                "bbox_xyxy": [10, 20, 30, 40],
            })

        runners = [RecordingRunner(0), RecordingRunner(1)]
        results = run_sam3d_for_views(views, {"sam3d_view_workers": 0}, None, runners)

    assert [view["view_index"] for view, _ in results] == [0, 1, 2, 3]
    assert [int(kpts[0, 0]) for _, kpts in results] == [0, 1, 0, 1]
    assert len(runners[0].calls) == 2
    assert len(runners[1].calls) == 2


def test_sam3d_view_failure_is_recorded_and_skipped() -> None:
    class FailingRunner:
        def run(self, image_path, bbox_xyxy, output_json_path):
            raise IndexError("index is out of bounds for dimension with size 0")

    with tempfile.TemporaryDirectory() as tmp:
        view_dir = Path(tmp) / "view_00"
        view_dir.mkdir()
        view = {
            "view_index": 0,
            "image_path": str(view_dir / "frame.jpg"),
            "bbox_json_path": str(view_dir / "bbox.json"),
            "sam3d_output_path": str(view_dir / "sam3d.json"),
            "bbox_xyxy": [10, 20, 30, 40],
        }

        kpts = run_sam3d_for_view(view, {}, None, FailingRunner())
        payload = json.loads((view_dir / "sam3d.json").read_text(encoding="utf-8"))

    assert kpts is None
    assert payload["status"] == "failed"
    assert payload["error_type"] == "IndexError"
    assert "index is out of bounds" in payload["error"]


def test_save_sam3d_payload_npz_preserves_full_payload_and_common_arrays() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        output_path = Path(tmp) / "view_00" / "sam3d.npz"
        payload = {
            "image_path": "/tmp/frame.jpg",
            "outputs": [{
                "pred_keypoints_3d": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                "pred_cam_t": [0.1, 0.2, 0.3],
                "scores": [0.9],
            }],
            "keypoints3d": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            "keypoints3d_camera": [[1.1, 2.2, 3.3], [4.1, 5.2, 6.3]],
            "keypoints2d": [[10.0, 20.0], [30.0, 40.0]],
            "joint_coords": [[7.0, 8.0, 9.0]],
        }

        saved = save_sam3d_payload_npz(payload, output_path)
        with np.load(saved, allow_pickle=False) as data:
            payload_roundtrip = json.loads(str(data["payload_json"]))
            keys = set(data.files)
            keypoints3d = data["keypoints3d"]

    assert saved == output_path
    assert payload_roundtrip["outputs"][0]["scores"] == [0.9]
    assert "keypoints3d" in keys
    assert "keypoints3d_camera" in keys
    assert "keypoints2d" in keys
    assert "joint_coords" in keys
    assert keypoints3d.shape == (2, 3)


def test_write_person_result_defaults_to_compact_fused_dir_only() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        person_dir = Path(tmp) / "frame_000001" / "track_0001"
        result = {
            "frame_number": 1,
            "track_id": 1,
            "fused_keypoints3d_world": [
                [1.0, 2.0, 3.0, 0.9],
                [None, None, None, 0.0],
            ],
        }

        write_person_result(person_dir, result)
        root_json = person_dir / "fused_keypoints3d.json"
        root_npz = person_dir / "fused_keypoints3d_world.npz"
        fused_json = person_dir / "fused" / "fused_keypoints3d.json"
        fused_npz = person_dir / "fused" / "fused_keypoints3d_world.npz"
        with np.load(fused_npz, allow_pickle=False) as data:
            kpts = data["fused_keypoints3d_world"]
            meta = json.loads(str(data["metadata_json"]))

        assert not root_json.exists()
        assert not root_npz.exists()
        assert fused_json.exists()
        assert fused_npz.exists()
        assert np.allclose(kpts[0], [1.0, 2.0, 3.0, 0.9])
        assert np.isnan(kpts[1, 0])
        assert meta["frame_number"] == 1
        assert meta["track_id"] == 1


def test_write_person_result_can_write_legacy_root_duplicates() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        person_dir = Path(tmp) / "frame_000001" / "track_0001"
        result = {
            "frame_number": 1,
            "track_id": 1,
            "fused_keypoints3d_world": [[1.0, 2.0, 3.0, 0.9]],
        }

        write_person_result(person_dir, result, write_legacy_root=True)

        assert (person_dir / "fused_keypoints3d.json").exists()
        assert (person_dir / "fused_keypoints3d_world.npz").exists()
        assert (person_dir / "fused" / "fused_keypoints3d.json").exists()
        assert (person_dir / "fused" / "fused_keypoints3d_world.npz").exists()


def test_save_fused_keypoints_npz_handles_missing_fused_array() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        output_path = Path(tmp) / "fused_keypoints3d_world.npz"

        saved = save_fused_keypoints_npz({"frame_number": 1}, output_path)

    assert saved is None
    assert not output_path.exists()


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


def test_person_center_prefers_track_points_across_360_seam() -> None:
    box = {
        "bbox_xyxy": [180, 40, 220, 120],
        "track_points_source": "pose",
        "track_points_xy": [
            [395.0, 100.0],
            [5.0, 102.0],
            [398.0, 98.0],
        ],
    }

    lon, lat, source = person_center_to_lon_lat(box, width=400, height=200)

    assert source == "pose"
    assert abs(abs(np.degrees(lon)) - 180.0) < 3.0
    assert abs(np.degrees(lat)) < 5.0


def test_person_center_uses_bbox_when_track_points_are_not_pose() -> None:
    box = {
        "bbox_xyxy": [180, 40, 220, 120],
        "track_points_source": "grid",
        "track_points_xy": [
            [395.0, 100.0],
            [5.0, 102.0],
            [398.0, 98.0],
        ],
    }

    lon, lat, source = person_center_to_lon_lat(box, width=400, height=200)

    assert source == "bbox"
    assert np.allclose([np.degrees(lon), np.degrees(lat)], [0.0, 18.0])


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
    test_sam3d_results_are_stored_in_video_level_results_folder()
    test_cli_exposes_device_pool_without_legacy_single_device_flag()
    test_auto_sam3d_devices_use_all_available_cuda_devices()
    test_expand_sam3d_runner_devices_respects_estimators_per_device_and_cap()
    test_sam3d_view_pool_uses_one_runner_per_worker_and_preserves_order()
    test_sam3d_view_failure_is_recorded_and_skipped()
    test_save_sam3d_payload_npz_preserves_full_payload_and_common_arrays()
    test_write_person_result_defaults_to_compact_fused_dir_only()
    test_write_person_result_can_write_legacy_root_duplicates()
    test_save_fused_keypoints_npz_handles_missing_fused_array()
    test_collect_frame_world_tracks_reads_sorted_tracks()
    test_world_keypoints_project_to_equirectangular_pixels()
    test_person_center_prefers_track_points_across_360_seam()
    test_person_center_uses_bbox_when_track_points_are_not_pose()
    test_save_frame_tracks_world_visualization_writes_png()
    test_visualization_writes_metadata_and_uses_stable_track_colors()
    test_frame_track_axes_use_skeleton_style_colors()
    test_track_fused_summary_visualization_writes_image_and_metadata()
    test_verbose_frame_visualization_reports_progress()

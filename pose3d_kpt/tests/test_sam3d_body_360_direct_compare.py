#!/usr/bin/env python3
"""Smoke tests for direct 360-frame SAM3D Body comparison."""

from __future__ import annotations

import json
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.modules.setdefault("cv2", types.SimpleNamespace(imwrite=lambda path, image: True, rectangle=lambda *args, **kwargs: None, circle=lambda *args, **kwargs: None, line=lambda *args, **kwargs: None, putText=lambda *args, **kwargs: None, FONT_HERSHEY_SIMPLEX=0, LINE_AA=0))


class FakeAxis:
    def scatter(self, *args, **kwargs):
        return None

    def plot(self, *args, **kwargs):
        return None

    def text(self, *args, **kwargs):
        return None

    def set_title(self, *args, **kwargs):
        return None

    def set_xlabel(self, *args, **kwargs):
        return None

    def set_ylabel(self, *args, **kwargs):
        return None

    def set_zlabel(self, *args, **kwargs):
        return None

    def view_init(self, *args, **kwargs):
        return None

    def legend(self, *args, **kwargs):
        return None

    def set_xlim(self, *args, **kwargs):
        return None

    def set_ylim(self, *args, **kwargs):
        return None

    def set_zlim(self, *args, **kwargs):
        return None


class FakeFigure:
    def add_subplot(self, *args, **kwargs):
        return FakeAxis()

    def tight_layout(self):
        return None

    def savefig(self, path, *args, **kwargs):
        Path(path).write_bytes(b"fake png")


def fake_figure(*args, **kwargs):
    return FakeFigure()


fake_matplotlib = types.ModuleType("matplotlib")
fake_pyplot = types.ModuleType("matplotlib.pyplot")
fake_matplotlib.use = lambda *args, **kwargs: None
fake_matplotlib.pyplot = fake_pyplot
fake_pyplot.figure = fake_figure
fake_pyplot.close = lambda *args, **kwargs: None
sys.modules.setdefault("matplotlib", fake_matplotlib)
sys.modules.setdefault("matplotlib.pyplot", fake_pyplot)

from sam3d_body_360_direct_compare import (
    parse_args,
    process_frame_direct_360_with_bbox,
    save_frame_direct_visualizations,
    write_direct_track_result,
)
from sam3d_body_multiview_fusion import Sam3DBodyDirectRunner


class FakeEstimator:
    def __init__(self) -> None:
        self.calls = []

    def process_one_image(self, image_path, **kwargs):
        self.calls.append((image_path, kwargs))
        return [
            {
                "bbox": np.array([1, 2, 11, 22], dtype=np.float32),
                "pred_keypoints_3d": np.array([[0.0, 0.0, 1.0], [0.1, 0.2, 1.2]], dtype=np.float32),
                "pred_keypoints_2d": np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
                "pred_cam_t": np.array([0.0, 0.0, 0.5], dtype=np.float32),
            },
            {
                "bbox": np.array([30, 40, 50, 70], dtype=np.float32),
                "pred_keypoints_3d": np.array([[1.0, 0.0, 1.0], [1.1, 0.2, 1.2]], dtype=np.float32),
                "pred_keypoints_2d": np.array([[33.0, 44.0], [55.0, 66.0]], dtype=np.float32),
                "pred_cam_t": np.array([0.0, 0.0, 0.25], dtype=np.float32),
            },
        ]


def test_direct_compare_cli_defaults_to_official_sam3d_without_run_flag() -> None:
    args = parse_args([])

    assert not hasattr(args, "run_sam3d")
    assert args.no_run_sam3d is False


def test_direct_compare_cli_accepts_bbox_driven_single_person_mode() -> None:
    args = parse_args(["--bbox-json", "selfie_bboxes.json", "--track-id", "1"])

    assert args.bbox_json == "selfie_bboxes.json"
    assert args.track_id == 1


def test_direct_track_result_does_not_write_fusion_outputs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        result = write_direct_track_result(
            frame_bgr=np.zeros((8, 8, 3), dtype=np.uint8),
            frame_number=56,
            detection_index=8,
            output={
                "bbox": np.array([1, 2, 6, 7], dtype=np.float32),
                "pred_keypoints_3d": np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
                "pred_cam_t": np.array([0.0, 0.0, 0.5], dtype=np.float32),
            },
            frame_image_path=output_dir / "frame_000056" / "frame_360.jpg",
            output_dir=output_dir,
            config={"visualize_keypoints": False, "sam3d_use_camera_translation": True},
        )
        track_dir = output_dir / "frame_000056" / "track_0008"

        assert not (track_dir / "fused").exists()
        assert "fused_keypoints3d_world" not in result
        assert "num_fused_views" not in result
        assert result["keypoints3d_camera"] == [[0.0, 0.0, 1.5, 1.0]]


def test_frame_direct_visualizations_return_combined_paths() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        frame_dir = Path(tmp) / "frame_000056"
        tracks = [
            {
                "track_id": 1,
                "source_bbox_xyxy": [1, 2, 6, 7],
                "keypoints2d": [[2.0, 3.0, 1.0]],
                "keypoints3d_camera": [[0.0, 0.0, 1.0, 1.0]],
            },
            {
                "track_id": 2,
                "source_bbox_xyxy": [10, 12, 16, 19],
                "keypoints2d": [[11.0, 13.0, 1.0]],
                "keypoints3d_camera": [[1.0, 0.0, 1.0, 1.0]],
            },
        ]

        vis = save_frame_direct_visualizations(
            np.zeros((24, 32, 3), dtype=np.uint8),
            frame_number=56,
            frame_dir=frame_dir,
            track_results=tracks,
            config={"visualize_keypoints": False, "min_kpt_conf": 0.0},
        )

        assert vis["detections_kpts2d_path"].endswith("frame_360_detections_kpts2d.jpg")
        assert vis["kpts3d_camera_path"].endswith("direct_360_kpts3d_camera.png")
        assert vis["num_visualized_tracks"] == 2


def test_sam3d_runner_can_use_internal_detector_without_bbox() -> None:
    runner = Sam3DBodyDirectRunner.__new__(Sam3DBodyDirectRunner)
    runner.estimator = FakeEstimator()
    runner.config = {
        "sam3d_use_known_intrinsics": False,
        "sam3d_inference_type": "body",
        "sam3d_use_camera_translation": True,
    }

    with tempfile.TemporaryDirectory() as tmp:
        output_path = Path(tmp) / "sam3d.json"
        kpts = runner.run(Path(tmp) / "frame.jpg", None, output_path)
        payload = json.loads(output_path.read_text(encoding="utf-8"))

    _, kwargs = runner.estimator.calls[0]
    assert kwargs["bboxes"] is None
    assert payload["detection_mode"] == "sam3d_internal_detector"
    assert payload["detected_bboxes_xyxy"] == [[1, 2, 11, 22], [30, 40, 50, 70]]
    assert payload["bbox_xyxy"] == [1, 2, 11, 22]
    assert kpts.shape == (2, 4)


def test_process_frame_direct_360_with_bbox_uses_provided_tracking_box() -> None:
    runner = Sam3DBodyDirectRunner.__new__(Sam3DBodyDirectRunner)
    runner.estimator = FakeEstimator()
    runner.config = {
        "sam3d_use_known_intrinsics": False,
        "sam3d_inference_type": "body",
        "sam3d_use_camera_translation": True,
    }

    with tempfile.TemporaryDirectory() as tmp:
        results = process_frame_direct_360_with_bbox(
            frame_bgr=np.zeros((24, 32, 3), dtype=np.uint8),
            frame_number=56,
            box={"track_id": 7, "bbox_xyxy": [4, 5, 14, 25]},
            output_dir=Path(tmp),
            config={"visualize_keypoints": False, "sam3d_use_camera_translation": True},
            sam3d_runner=runner,
        )

    _, kwargs = runner.estimator.calls[0]
    assert kwargs["bboxes"].tolist() == [[4.0, 5.0, 14.0, 25.0]]
    assert len(results) == 1
    assert results[0]["track_id"] == 7
    assert results[0]["source_bbox_xyxy"] == [4, 5, 14, 25]
    assert results[0]["detection_mode"] == "provided_bbox"


if __name__ == "__main__":
    test_direct_compare_cli_defaults_to_official_sam3d_without_run_flag()
    test_direct_compare_cli_accepts_bbox_driven_single_person_mode()
    test_direct_track_result_does_not_write_fusion_outputs()
    test_frame_direct_visualizations_return_combined_paths()
    test_sam3d_runner_can_use_internal_detector_without_bbox()
    test_process_frame_direct_360_with_bbox_uses_provided_tracking_box()

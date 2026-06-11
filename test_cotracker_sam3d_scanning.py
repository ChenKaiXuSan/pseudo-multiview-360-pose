#!/usr/bin/env python3
"""Smoke tests for SAM3D-backed cubemap scanning in cotracker tracking."""

from __future__ import annotations

import json
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

sys.modules.setdefault(
    "cv2",
    types.SimpleNamespace(
        imwrite=lambda path, image: Path(path).write_bytes(b"fake image") or True,
        remap=lambda image, map_x, map_y, interpolation, borderMode=None: np.zeros(
            (map_x.shape[0], map_x.shape[1], image.shape[2]), dtype=image.dtype
        ),
        INTER_LINEAR=1,
        BORDER_WRAP=3,
    ),
)

from cotracker_person_tracking import sam3d_cubemap_sliding_detection, validate_sam3d_scanner_detector


class FakeSam3DRunner:
    def __init__(self) -> None:
        self.calls = []
        self.config = {
            "view_width": 0,
            "view_height": 0,
            "hfov_deg": 0.0,
            "vfov_deg": 0.0,
        }

    def run(self, image_path: Path, bbox_xyxy, output_json_path: Path):
        self.calls.append((Path(image_path), bbox_xyxy, Path(output_json_path), dict(self.config)))
        payload = {
            "outputs": [
                {
                    "bbox": [2.0, 3.0, 8.0, 10.0],
                    "pred_keypoints_2d": [
                        [3.0, 4.0, 0.9],
                        [5.0, 6.0, 0.8],
                        [7.0, 8.0, 0.7],
                    ],
                }
            ]
        }
        output_json_path.write_text(json.dumps(payload), encoding="utf-8")
        return None


def test_sam3d_scanning_projects_window_bbox_and_keypoints_to_360() -> None:
    frame = np.zeros((32, 64, 3), dtype=np.uint8)
    runner = FakeSam3DRunner()

    with tempfile.TemporaryDirectory() as tmp:
        detections, _ = sam3d_cubemap_sliding_detection(
            runner,
            frame,
            cache_dir=Path(tmp),
            frame_number=7,
            clip_index=2,
            window_width_ratio=0.5,
            step_size=16,
            face_size=16,
            edge_samples=4,
            min_keypoints=2,
            nms_angle_threshold_deg=0.0,
            nms_iou_threshold=1.0,
            nms_containment_threshold=1.0,
        )

    assert runner.calls
    assert detections
    det = detections[0]
    assert det["class"] == "person"
    assert det["source"] == "sam3d_body"
    assert det["x1"] < det["x2"]
    assert det["y1"] < det["y2"]
    assert len(det["keypoints"]) == 3
    for x, y, score in det["keypoints"]:
        assert 0 <= x < frame.shape[1]
        assert 0 <= y < frame.shape[0]
        assert score > 0


def test_scanner_detector_validation_rejects_empty_detector() -> None:
    try:
        validate_sam3d_scanner_detector({"sam3d_detector_name": ""})
    except RuntimeError as exc:
        assert "needs a real person detector" in str(exc)
    else:
        raise AssertionError("empty detector should be rejected for scanning")


def test_scanner_detector_validation_reports_missing_vitdet_dependency() -> None:
    try:
        validate_sam3d_scanner_detector({"sam3d_detector_name": "vitdet"})
    except RuntimeError as exc:
        message = str(exc)
        assert "vitdet" in message
        assert "detectron2" in message
    else:
        # Environment already has detectron2; in that case validation correctly passes.
        return


if __name__ == "__main__":
    test_sam3d_scanning_projects_window_bbox_and_keypoints_to_360()
    test_scanner_detector_validation_rejects_empty_detector()
    test_scanner_detector_validation_reports_missing_vitdet_dependency()

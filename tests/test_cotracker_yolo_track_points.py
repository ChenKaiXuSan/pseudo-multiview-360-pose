#!/usr/bin/env python3
"""Tests for CoTracker YOLO tracked-point metadata serialization."""

from __future__ import annotations

import sys
import types


sys.modules.setdefault("cv2", types.SimpleNamespace())
sys.modules.setdefault(
    "torch",
    types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
        hub=types.SimpleNamespace(load=lambda *args, **kwargs: None),
        no_grad=lambda: None,
        from_numpy=lambda value: value,
        argmax=lambda value: types.SimpleNamespace(item=lambda: 0),
    ),
)
sys.modules.setdefault("ultralytics", types.SimpleNamespace(YOLO=lambda *args, **kwargs: None))
sys.modules.setdefault(
    "test_360_detection",
    types.SimpleNamespace(cubemap_sliding_detection=lambda *args, **kwargs: ([], None)),
)

from cotracker_person_tracking_yolo import serialize_track_boxes


def test_serialize_track_boxes_preserves_visible_track_points() -> None:
    rows = serialize_track_boxes([
        {
            "id": 7,
            "box": (10, 20, 30, 80),
            "score": 0.91,
            "source": "pose",
            "visible_points": 3,
            "track_points_xy": [[12.4, 32.8], [16.2, 42.1], [28.9, 70.5]],
            "center_xy": [16.2, 42.1],
            "center_source": "track_points",
        }
    ])

    assert rows == [
        {
            "track_id": 7,
            "bbox_xyxy": [10, 20, 30, 80],
            "conf": 0.91,
            "source": "pose",
            "visible_points": 3,
            "track_points_xy": [[12.4, 32.8], [16.2, 42.1], [28.9, 70.5]],
            "track_points_source": "pose",
            "center_xy": [16.2, 42.1],
            "center_source": "track_points",
        }
    ]


if __name__ == "__main__":
    test_serialize_track_boxes_preserves_visible_track_points()

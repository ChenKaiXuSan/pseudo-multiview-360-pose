#!/usr/bin/env python
"""Focused checks for selfie bbox selection and source semantics."""

from cotracker_selfie_bbox_tracking_yolo import (
    CONFIG,
    select_selfie_detection,
    select_selfie_frame_box,
    serialize_selfie_track_boxes,
)


def test_default_selfie_selection_prefers_bottom_near_largest():
    detections = [
        {"class": "person", "x1": 10, "y1": 10, "x2": 90, "y2": 210, "conf": 0.95},
        {"class": "person", "x1": 20, "y1": 330, "x2": 95, "y2": 520, "conf": 0.80},
    ]

    selected = select_selfie_detection(detections, (540, 960, 3), CONFIG)

    assert selected is detections[1], selected


def test_frame_box_matches_when_reference_center_is_inside_detection():
    config = dict(CONFIG)
    config.update({
        "id_match_iou_threshold": 0.5,
        "id_match_center_threshold": 10.0,
        "id_match_area_ratio_threshold": 0.9,
    })
    detections = [
        {"class": "person", "x1": 0, "y1": 0, "x2": 120, "y2": 120, "conf": 0.9},
    ]
    refs = [
        {"id": 1, "box": (50, 50, 60, 60), "source": "grid", "visible_points": 6, "score": 0.7},
    ]

    selected = select_selfie_frame_box(detections, refs, config, track_id=1)

    assert selected["box"] == (0, 0, 120, 120)
    assert selected["source"] == "yolo_matched"
    assert selected["ref_source"] == "grid"


def test_frame_box_carries_reference_track_points_when_matched():
    config = dict(CONFIG)
    detections = [
        {"class": "person", "x1": 0, "y1": 0, "x2": 120, "y2": 120, "conf": 0.9},
    ]
    refs = [
        {
            "id": 1,
            "box": (10, 10, 100, 110),
            "source": "pose",
            "visible_points": 3,
            "score": 0.7,
            "track_points_xy": [[12.0, 30.0], [18.0, 42.0], [20.0, 55.0]],
            "track_points_source": "pose",
            "center_xy": [18.0, 42.0],
            "center_source": "track_points",
        },
    ]

    selected = select_selfie_frame_box(detections, refs, config, track_id=1)
    serialized = serialize_selfie_track_boxes([selected])

    assert selected["track_points_xy"] == refs[0]["track_points_xy"]
    assert serialized[0]["track_points_xy"] == refs[0]["track_points_xy"]
    assert serialized[0]["track_points_source"] == "pose"
    assert serialized[0]["center_source"] == "track_points"


def test_frame_box_fallback_keeps_tracker_fallback_source():
    config = dict(CONFIG)
    detections = [
        {"class": "person", "x1": 300, "y1": 300, "x2": 420, "y2": 420, "conf": 0.9},
    ]
    refs = [
        {"id": 1, "box": (10, 10, 90, 160), "source": "pose", "visible_points": 5, "score": 0.8},
    ]

    selected = select_selfie_frame_box(detections, refs, config, track_id=1)

    assert selected["box"] == (10, 10, 90, 160)
    assert selected["source"] == "tracker_fallback"


def test_serialized_boxes_keep_ref_source_when_present():
    boxes = [{
        "id": 1,
        "box": (0, 0, 120, 120),
        "source": "yolo_matched",
        "ref_source": "grid",
        "visible_points": 6,
        "score": 0.9,
    }]

    serialized = serialize_selfie_track_boxes(boxes)

    assert serialized[0]["source"] == "yolo_matched"
    assert serialized[0]["ref_source"] == "grid"


if __name__ == "__main__":
    test_default_selfie_selection_prefers_bottom_near_largest()
    test_frame_box_matches_when_reference_center_is_inside_detection()
    test_frame_box_carries_reference_track_points_when_matched()
    test_frame_box_fallback_keeps_tracker_fallback_source()
    test_serialized_boxes_keep_ref_source_when_present()
    print("selfie bbox tracking checks passed")

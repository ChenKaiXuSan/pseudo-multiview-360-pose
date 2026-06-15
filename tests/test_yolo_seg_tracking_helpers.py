#!/usr/bin/env python3
"""Unit tests for YOLO-seg CoTracker helper behavior."""

from __future__ import annotations

import sys
import types

import numpy as np


def _connected_components_with_stats(mask, connectivity):
    mask = np.asarray(mask, dtype=np.uint8)
    labels = np.zeros(mask.shape, dtype=np.int32)
    stats = [[0, 0, 0, 0, int((mask == 0).sum())]]
    centroids = [[0.0, 0.0]]
    label_id = 0
    h, w = mask.shape
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        neighbors += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    for y in range(h):
        for x in range(w):
            if mask[y, x] == 0 or labels[y, x] != 0:
                continue
            label_id += 1
            stack = [(y, x)]
            labels[y, x] = label_id
            pixels = []
            while stack:
                cy, cx = stack.pop()
                pixels.append((cy, cx))
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = label_id
                        stack.append((ny, nx))
            ys = np.array([p[0] for p in pixels])
            xs = np.array([p[1] for p in pixels])
            stats.append([int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1), len(pixels)])
            centroids.append([float(xs.mean()), float(ys.mean())])

    return label_id + 1, labels, np.asarray(stats, dtype=np.int32), np.asarray(centroids, dtype=np.float64)

sys.modules.setdefault(
    "cv2",
    types.SimpleNamespace(
        resize=lambda image, size, interpolation=None: np.zeros((size[1], size[0]), dtype=image.dtype),
        remap=lambda image, map_x, map_y, interpolation=None: np.zeros(
            (map_x.shape[0], map_x.shape[1], image.shape[2]), dtype=image.dtype
        ),
        connectedComponentsWithStats=_connected_components_with_stats,
        CC_STAT_AREA=4,
        INTER_NEAREST=0,
        INTER_LINEAR=1,
    ),
)
sys.modules.setdefault("torch", types.SimpleNamespace())
sys.modules.setdefault("ultralytics", types.SimpleNamespace(YOLO=object))
matplotlib_module = types.ModuleType("matplotlib")
matplotlib_module.use = lambda *_args, **_kwargs: None
pyplot_module = types.ModuleType("matplotlib.pyplot")
pyplot_module.subplots = lambda *args, **kwargs: (None, np.empty((2, 3), dtype=object))
pyplot_module.tight_layout = lambda: None
pyplot_module.savefig = lambda *args, **kwargs: None
pyplot_module.close = lambda *args, **kwargs: None
pyplot_module.show = lambda: None
matplotlib_module.pyplot = pyplot_module
sys.modules.setdefault("matplotlib", matplotlib_module)
sys.modules.setdefault("matplotlib.pyplot", pyplot_module)

from cotracker_person_tracking_yolo import (  # noqa: E402
    CONFIG,
    apply_video_name_output_paths,
    assign_frame_detection_boxes,
    box_iou,
    filter_overlapping_track_boxes,
    grid_points_in_box,
    make_query_points,
    resolve_model_path,
    serialize_track_boxes,
)

from cotracker_selfie_bbox_tracking_yolo import (  # noqa: E402
    select_selfie_detection,
    select_selfie_frame_box,
)
from test_360_detection import build_detection_views, cluster_view_detections  # noqa: E402


def test_grid_points_in_box_keeps_points_inside_margin() -> None:
    points = grid_points_in_box(10, 20, 30, 60, points_per_axis=2, margin_ratio=0.25)

    assert points == [[15.0, 30.0], [25.0, 30.0], [15.0, 50.0], [25.0, 50.0]]


def test_make_query_points_prefers_pose_keypoints() -> None:
    detections = [
        {
            "class": "person",
            "x1": 10,
            "y1": 20,
            "x2": 30,
            "y2": 60,
            "conf": 0.8,
            "keypoints": [[12, 22, 0.9], [18, 30, 0.9], [24, 42, 0.9]],
            "source": "pose",
        }
    ]
    config = {"use_pose_keypoints": True, "min_pose_keypoints": 3, "fallback_to_grid_points": True}

    queries, groups = make_query_points(
        detections,
        scale_x=0.5,
        scale_y=0.25,
        points_per_axis=2,
        margin_ratio=0.25,
        config=config,
        query_time=4,
    )

    assert queries.tolist() == [[4.0, 6.0, 5.5], [4.0, 9.0, 7.5], [4.0, 12.0, 10.5]]
    assert groups[0]["source"] == "pose"
    assert groups[0]["seed_box"].tolist() == [5.0, 5.0, 15.0, 15.0]


def test_make_query_points_falls_back_to_bbox_grid_when_pose_missing() -> None:
    detections = [
        {
            "class": "person",
            "x1": 10,
            "y1": 20,
            "x2": 30,
            "y2": 60,
            "conf": 0.8,
            "keypoints": [],
            "source": "grid",
        }
    ]
    config = {"use_pose_keypoints": True, "min_pose_keypoints": 3, "fallback_to_grid_points": True}

    queries, groups = make_query_points(
        detections,
        scale_x=1.0,
        scale_y=1.0,
        points_per_axis=2,
        margin_ratio=0.25,
        config=config,
        query_time=0,
    )

    assert queries.tolist() == [[0.0, 15.0, 30.0], [0.0, 25.0, 30.0], [0.0, 15.0, 50.0], [0.0, 25.0, 50.0]]
    assert groups[0]["source"] == "grid"


def test_serialize_track_boxes_outputs_bbox_xyxy() -> None:
    boxes = [{"id": 4, "box": (1, 2, 30, 40), "score": 0.8, "source": "pose", "visible_points": 5}]

    payload = serialize_track_boxes(boxes)

    assert payload == [
        {
            "track_id": 4,
            "bbox_xyxy": [1, 2, 30, 40],
            "conf": 0.8,
            "source": "pose",
            "visible_points": 5,
        }
    ]


def test_box_iou_returns_overlap_ratio() -> None:
    assert abs(box_iou((0, 0, 10, 10), (5, 5, 15, 15)) - (25 / 175)) < 1e-6


def test_assign_frame_detection_boxes_outputs_detector_box_with_track_id() -> None:
    detections = [{"x1": 120, "y1": 80, "x2": 180, "y2": 220, "conf": 0.91, "class": "person"}]
    track_refs = [
        {
            "id": 7,
            "box": (90, 40, 220, 280),
            "source": "pose",
            "visible_points": 12,
            "score": 0.8,
        }
    ]
    config = {"id_match_iou_threshold": 0.1, "id_match_center_threshold": 80.0, "id_match_area_ratio_threshold": 0.1}

    boxes, next_id, matched = assign_frame_detection_boxes(detections, track_refs, config, next_track_id=10)

    assert next_id == 10
    assert matched == 1
    assert boxes == [
        {
            "id": 7,
            "box": (120, 80, 180, 220),
            "source": "pose",
            "visible_points": 12,
            "score": 0.91,
        }
    ]


def test_assign_frame_detection_boxes_reuses_previous_detection_ref() -> None:
    detections = [{"x1": 102, "y1": 104, "x2": 158, "y2": 220, "conf": 0.75, "class": "person"}]
    track_refs = [{"id": 3, "box": (100, 100, 160, 220), "source": "yolo", "visible_points": 0, "score": 0.7}]
    config = {"id_match_iou_threshold": 0.1, "id_match_center_threshold": 80.0, "id_match_area_ratio_threshold": 0.1}

    boxes, next_id, matched = assign_frame_detection_boxes(detections, track_refs, config, next_track_id=4)

    assert next_id == 4
    assert matched == 1
    assert boxes[0]["id"] == 3
    assert boxes[0]["box"] == (102, 104, 158, 220)


def test_assign_frame_detection_boxes_assigns_new_id_to_unmatched_detection() -> None:
    detections = [{"x1": 300, "y1": 100, "x2": 360, "y2": 220, "conf": 0.66, "class": "person"}]
    track_refs = [{"id": 3, "box": (100, 100, 160, 220), "source": "pose", "visible_points": 8, "score": 0.7}]
    config = {"id_match_iou_threshold": 0.2, "id_match_center_threshold": 40.0, "id_match_area_ratio_threshold": 0.2}

    boxes, next_id, matched = assign_frame_detection_boxes(detections, track_refs, config, next_track_id=4)

    assert matched == 0
    assert next_id == 5
    assert boxes[0]["id"] == 4
    assert boxes[0]["source"] == "yolo"


def test_filter_overlapping_track_boxes_drops_projection_duplicate() -> None:
    boxes = [
        {"id": 1, "box": (2836, 1044, 3288, 2205), "source": "pose", "visible_points": 17, "score": 0.96},
        {"id": 8, "box": (3104, 1051, 3469, 2201), "source": "yolo", "visible_points": 0, "score": 0.92},
    ]
    kept = filter_overlapping_track_boxes(boxes, CONFIG)

    assert [box["id"] for box in kept] == [1]


def test_filter_overlapping_track_boxes_drops_side_by_side_split_person() -> None:
    boxes = [
        {"id": 201, "box": (935, 270, 1115, 558), "source": "pose", "visible_points": 14, "score": 0.95},
        {"id": 202, "box": (1080, 278, 1205, 553), "source": "pose", "visible_points": 12, "score": 0.93},
    ]

    kept = filter_overlapping_track_boxes(boxes, CONFIG)

    assert [box["id"] for box in kept] == [201]


def test_filter_overlapping_track_boxes_drops_gapped_split_person() -> None:
    boxes = [
        {"id": 4, "box": (3680, 1405, 3747, 1905), "source": "pose", "visible_points": 13, "score": 0.945},
        {"id": 7, "box": (3582, 1401, 3667, 1910), "source": "pose", "visible_points": 12, "score": 0.927},
    ]

    kept = filter_overlapping_track_boxes(boxes, CONFIG)

    assert [box["id"] for box in kept] == [4]


def test_filter_overlapping_track_boxes_unions_split_person_boxes() -> None:
    boxes = [
        {"id": 11, "box": (1200, 500, 1285, 820), "source": "pose", "visible_points": 14, "score": 0.94},
        {"id": 12, "box": (1260, 505, 1390, 825), "source": "pose", "visible_points": 13, "score": 0.92},
    ]

    kept = filter_overlapping_track_boxes(boxes, CONFIG)

    assert len(kept) == 1
    assert kept[0]["id"] == 11
    assert kept[0]["box"] == (1200, 500, 1390, 825)
    assert kept[0]["visible_points"] == 14
    assert kept[0]["score"] == 0.94


def test_filter_overlapping_track_boxes_unions_wide_gapped_split_person() -> None:
    boxes = [
        {"id": 3, "box": (3600, 1326, 3898, 2109), "source": "pose", "visible_points": 14, "score": 0.943},
        {"id": 4, "box": (3390, 1330, 3587, 2047), "source": "pose", "visible_points": 12, "score": 0.922},
    ]

    kept = filter_overlapping_track_boxes(boxes, CONFIG)

    assert len(kept) == 1
    assert kept[0]["id"] == 3
    assert kept[0]["box"] == (3390, 1326, 3898, 2109)


def test_filter_overlapping_track_boxes_drops_nearby_partial_fragment() -> None:
    boxes = [
        {"id": 2, "box": (1799, 1484, 2045, 2123), "source": "pose", "visible_points": 14, "score": 0.966},
        {"id": 3, "box": (2160, 1620, 2234, 2018), "source": "grid", "visible_points": 0, "score": 0.899},
    ]

    kept = filter_overlapping_track_boxes(boxes, CONFIG)

    assert len(kept) == 1
    assert kept[0]["id"] == 2


def test_select_selfie_detection_prefers_largest_person() -> None:
    detections = [
        {"class": "person", "x1": 10, "y1": 10, "x2": 30, "y2": 80, "conf": 0.9},
        {"class": "person", "x1": 100, "y1": 20, "x2": 220, "y2": 300, "conf": 0.8},
        {"class": "car", "x1": 0, "y1": 0, "x2": 500, "y2": 500, "conf": 1.0},
    ]

    selected = select_selfie_detection(detections, frame_shape=(400, 600, 3), config={})

    assert selected == detections[1]


def test_select_selfie_frame_box_returns_only_matched_detector_box() -> None:
    detections = [
        {"class": "person", "x1": 98, "y1": 100, "x2": 205, "y2": 330, "conf": 0.91},
        {"class": "person", "x1": 350, "y1": 100, "x2": 430, "y2": 260, "conf": 0.95},
    ]
    refs = [{"id": 1, "box": (100, 90, 210, 340), "source": "pose", "visible_points": 11, "score": 0.8}]
    config = {"id_match_iou_threshold": 0.1, "id_match_center_threshold": 80.0, "id_match_area_ratio_threshold": 0.1}

    box = select_selfie_frame_box(detections, refs, config, track_id=1)

    assert box == {
        "id": 1,
        "box": (98, 100, 205, 330),
        "source": "pose",
        "visible_points": 11,
        "score": 0.91,
    }


def test_select_selfie_frame_box_uses_tracker_fallback_when_detection_missing() -> None:
    refs = [{"id": 1, "box": (100, 90, 210, 340), "source": "pose", "visible_points": 11, "score": 0.8}]
    config = {"id_match_iou_threshold": 0.1, "id_match_center_threshold": 80.0, "id_match_area_ratio_threshold": 0.1}

    box = select_selfie_frame_box([], refs, config, track_id=1)

    assert box == {
        "id": 1,
        "box": (100, 90, 210, 340),
        "source": "tracker_fallback",
        "visible_points": 11,
        "score": 0.8,
    }


def test_apply_video_name_output_paths_uses_input_stem_for_person_tracker() -> None:
    config = {
        "video_path": "/data/custom_run.mp4",
        "output_root_dir": "/tmp/out",
        "output_suffix": "cotracker_yolo",
    }

    updated = apply_video_name_output_paths(config)

    assert updated["output_path"] == "/tmp/out/custom_run_cotracker_yolo_tracked.mp4"
    assert updated["frames_output_dir"] == "/tmp/out/custom_run_cotracker_yolo_frames"
    assert updated["bbox_output_path"] == "/tmp/out/custom_run_cotracker_yolo_bboxes.json"
    assert config.get("output_path") is None


def test_apply_video_name_output_paths_preserves_explicit_paths() -> None:
    config = {
        "video_path": "/data/custom_run.mp4",
        "output_root_dir": "/tmp/out",
        "output_suffix": "cotracker_yolo",
        "output_path": "/explicit/video.mp4",
        "frames_output_dir": "/explicit/frames",
        "bbox_output_path": "/explicit/boxes.json",
    }

    updated = apply_video_name_output_paths(config)

    assert updated["output_path"] == "/explicit/video.mp4"
    assert updated["frames_output_dir"] == "/explicit/frames"
    assert updated["bbox_output_path"] == "/explicit/boxes.json"


def test_resolve_model_path_prefers_script_dir() -> None:
    path = resolve_model_path("yolo26xseg.pt")

    assert str(path).endswith("360PoseFusion/yolo26xseg.pt")


def test_build_detection_views_uses_18_view_layout() -> None:
    views = build_detection_views(
        enable_extra_views=True,
        horizontal_extra_yaws=[45, 135, 225, 315],
        upper_extra_pitch=55,
        lower_extra_pitch=-55,
        vertical_extra_yaws=[0, 90, 180, 270],
        extra_view_fov_deg=100,
    )

    assert len(views) == 18
    assert [view["type"] for view in views[:6]] == ["cubemap"] * 6
    assert [view["type"] for view in views[6:]] == ["perspective"] * 12
    assert views[6]["name"] == "yaw45_pitch0"
    assert views[10]["name"] == "yaw0_pitch55"
    assert views[14]["name"] == "yaw0_pitch-55"


def test_apply_video_name_output_paths_adds_debug_view_dir() -> None:
    config = {
        "video_path": "/data/custom_run.mp4",
        "output_root_dir": "/tmp/out",
        "output_suffix": "cotracker_yolo",
        "save_view_debug": True,
    }

    updated = apply_video_name_output_paths(config)

    assert updated["view_debug_dir"] == "/tmp/out/custom_run_cotracker_yolo_views"


def test_filter_overlapping_track_boxes_drops_upper_body_overlay_fragment() -> None:
    boxes = [
        {"id": 1, "box": (2740, 1030, 3370, 2160), "source": "pose", "visible_points": 14, "score": 0.94},
        {"id": 5, "box": (2860, 1045, 3840, 1375), "source": "yolo", "visible_points": 0, "score": 0.89},
    ]

    kept = filter_overlapping_track_boxes(boxes, CONFIG)

    assert len(kept) == 1
    assert kept[0]["id"] == 1
    assert kept[0]["box"] == (2740, 1030, 3370, 2160)


def test_cluster_view_detections_keeps_best_full_body_box() -> None:
    detections = [
        {
            "class": "person",
            "x1": 2740,
            "y1": 1030,
            "x2": 3370,
            "y2": 2160,
            "conf": 0.84,
            "view_name": "face0",
            "view_type": "cubemap",
            "_phi": 0.10,
            "_lat": -0.05,
        },
        {
            "class": "person",
            "x1": 2860,
            "y1": 1045,
            "x2": 3840,
            "y2": 1375,
            "conf": 0.93,
            "view_name": "yaw45_pitch0",
            "view_type": "perspective",
            "_phi": 0.12,
            "_lat": -0.04,
        },
    ]

    clustered = cluster_view_detections(
        detections,
        angle_threshold_deg=8.0,
        iou_threshold=0.10,
        containment_threshold=0.30,
    )

    assert len(clustered) == 1
    assert clustered[0]["view_name"] == "face0"
    assert clustered[0]["x1"] == 2740
    assert clustered[0]["y2"] == 2160


if __name__ == "__main__":
    test_grid_points_in_box_keeps_points_inside_margin()
    test_make_query_points_prefers_pose_keypoints()
    test_make_query_points_falls_back_to_bbox_grid_when_pose_missing()
    test_serialize_track_boxes_outputs_bbox_xyxy()
    test_box_iou_returns_overlap_ratio()
    test_assign_frame_detection_boxes_outputs_detector_box_with_track_id()
    test_assign_frame_detection_boxes_reuses_previous_detection_ref()
    test_assign_frame_detection_boxes_assigns_new_id_to_unmatched_detection()
    test_filter_overlapping_track_boxes_drops_projection_duplicate()
    test_filter_overlapping_track_boxes_drops_side_by_side_split_person()
    test_filter_overlapping_track_boxes_drops_gapped_split_person()
    test_filter_overlapping_track_boxes_unions_split_person_boxes()
    test_filter_overlapping_track_boxes_unions_wide_gapped_split_person()
    test_filter_overlapping_track_boxes_drops_nearby_partial_fragment()
    test_filter_overlapping_track_boxes_drops_upper_body_overlay_fragment()
    test_select_selfie_detection_prefers_largest_person()
    test_select_selfie_frame_box_returns_only_matched_detector_box()
    test_select_selfie_frame_box_uses_tracker_fallback_when_detection_missing()
    test_apply_video_name_output_paths_uses_input_stem_for_person_tracker()
    test_apply_video_name_output_paths_preserves_explicit_paths()
    test_resolve_model_path_prefers_script_dir()
    test_build_detection_views_uses_18_view_layout()
    test_cluster_view_detections_keeps_best_full_body_box()
    test_apply_video_name_output_paths_adds_debug_view_dir()

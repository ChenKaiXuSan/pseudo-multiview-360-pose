import math
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pointcloud_reconstruction.bbox_views import (
    box_center_x,
    build_anchor_views,
    circular_mean_degrees,
    equirect_x_to_yaw,
    normalize_yaw,
    select_selfie_box,
)


class BBoxViewTests(unittest.TestCase):
    def test_equirect_x_to_yaw_uses_front_at_image_center(self):
        self.assertAlmostEqual(equirect_x_to_yaw(0, 4000), -180.0)
        self.assertAlmostEqual(equirect_x_to_yaw(2000, 4000), 0.0)
        self.assertAlmostEqual(equirect_x_to_yaw(3000, 4000), 90.0)
        self.assertAlmostEqual(equirect_x_to_yaw(4000, 4000), -180.0)

    def test_box_center_prefers_track_point_center_when_available(self):
        box = {"box": [100, 10, 300, 400], "center_xy": [360, 200]}
        self.assertAlmostEqual(box_center_x(box), 360.0)

    def test_select_selfie_box_uses_requested_id_then_falls_back(self):
        frame = {
            "boxes": [
                {"id": 7, "box": [0, 0, 20, 20], "score": 0.9},
                {"id": 1, "box": [100, 0, 200, 50], "score": 0.2},
            ]
        }
        self.assertEqual(select_selfie_box(frame, target_id=1)["id"], 1)
        self.assertEqual(select_selfie_box(frame, target_id=3)["id"], 7)

    def test_circular_mean_handles_dateline_wrap(self):
        mean = circular_mean_degrees([179.0, -179.0])
        self.assertTrue(abs(abs(mean) - 180.0) < 1e-6)


    def test_build_anchor_views_defaults_to_six_horizontal_views(self):
        views = build_anchor_views(anchor_yaw=10.0, fov_deg=100)

        self.assertEqual(
            [view.name for view in views],
            ["selfie", "front_right", "back_right", "back", "back_left", "front_left"],
        )
        self.assertEqual(
            [normalize_yaw(view.yaw_deg) for view in views],
            [10.0, 70.0, 130.0, -170.0, -110.0, -50.0],
        )

    def test_build_anchor_views_centers_offsets_on_selfie_yaw(self):
        views = build_anchor_views(anchor_yaw=170.0, yaw_offsets=[0, 90, -90, 180], fov_deg=100)
        self.assertEqual([view.name for view in views], ["selfie", "right", "left", "back"])
        self.assertEqual([normalize_yaw(view.yaw_deg) for view in views], [170.0, -100.0, 80.0, -10.0])
        self.assertTrue(all(math.isclose(view.fov_deg, 100.0) for view in views))


if __name__ == "__main__":
    unittest.main()

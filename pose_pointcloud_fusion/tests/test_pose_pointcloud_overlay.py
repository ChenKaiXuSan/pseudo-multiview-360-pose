import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pose_pointcloud_fusion.pose_pointcloud_overlay import (
    build_pose_overlay_points,
    load_fused_keypoints,
    write_pose_pointcloud_overlay_frame,
)


class PosePointcloudOverlayTests(unittest.TestCase):
    def test_load_fused_keypoints_filters_invalid_and_low_confidence_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fused_keypoints3d.json"
            path.write_text(
                json.dumps(
                    {
                        "frame_number": 1,
                        "track_id": 1,
                        "fused_keypoints3d_world": [
                            [1.0, 2.0, 3.0, 0.9],
                            [4.0, 5.0, 6.0, 0.1],
                            [None, 8.0, 9.0, 0.9],
                        ],
                    }
                ),
                encoding="utf-8",
            )

            kpts = load_fused_keypoints(path, min_conf=0.5)

        self.assertEqual(kpts.shape, (3, 4))
        self.assertTrue(kpts[0].mask_valid)
        self.assertFalse(kpts[1].mask_valid)
        self.assertFalse(kpts[2].mask_valid)

    def test_build_pose_overlay_points_adds_joint_markers_and_bone_samples(self):
        keypoints = load_fused_keypoints.from_rows(
            [
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
            ],
            min_conf=0.5,
        )

        points = build_pose_overlay_points(
            keypoints,
            edges=[(0, 1)],
            joint_radius=0.1,
            bone_step=0.5,
            joint_rgb=[255, 0, 0],
            bone_rgb=[0, 255, 255],
        )

        self.assertGreaterEqual(len(points), 14)
        self.assertIn({"xyz": [0.0, 0.0, 0.0], "rgb": [255, 0, 0]}, points)
        self.assertIn({"xyz": [0.5, 0.0, 0.0], "rgb": [0, 255, 255]}, points)

    def test_write_pose_pointcloud_overlay_frame_appends_pose_points_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene_ply = root / "scene.ply"
            scene_ply.write_text(
                "\n".join(
                    [
                        "ply",
                        "format ascii 1.0",
                        "element vertex 1",
                        "property float x",
                        "property float y",
                        "property float z",
                        "property uchar red",
                        "property uchar green",
                        "property uchar blue",
                        "end_header",
                        "10.000000 20.000000 30.000000 1 2 3",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            pose_json = root / "fused_keypoints3d.json"
            pose_json.write_text(
                json.dumps(
                    {
                        "frame_number": 1,
                        "track_id": 1,
                        "fused_keypoints3d_world": [
                            [0.0, 0.0, 0.0, 1.0],
                            [1.0, 0.0, 0.0, 1.0],
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output_ply = root / "overlay.ply"

            summary = write_pose_pointcloud_overlay_frame(
                scene_ply=scene_ply,
                pose_json=pose_json,
                output_ply=output_ply,
                min_conf=0.5,
                edges=[(0, 1)],
                joint_radius=0.1,
                bone_step=0.5,
            )

            text = output_ply.read_text(encoding="utf-8")
            self.assertTrue(output_ply.exists())
            self.assertIn("element vertex {}".format(1 + summary["overlay_points"]), text)
            self.assertEqual(summary["scene_points"], 1)
            self.assertEqual(summary["valid_keypoints"], 2)
            self.assertGreater(summary["overlay_points"], 2)


if __name__ == "__main__":
    unittest.main()

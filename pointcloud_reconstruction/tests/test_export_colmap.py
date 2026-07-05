import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pointcloud_reconstruction.export_colmap import (
    ColmapDepthPoint,
    build_vipe_to_colmap_commands,
    flatten_nested_colmap_outputs,
    matrix_to_colmap_pose,
    write_observation_colmap_text,
)


class ExportColmapTests(unittest.TestCase):
    def test_build_commands_use_colmap_root_not_view_subdir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "view_manifest.json"
            manifest.write_text('{"views": [{"name": "selfie"}, {"name": "right"}]}', encoding="utf-8")
            commands = build_vipe_to_colmap_commands(
                manifest_path=manifest,
                vipe_results_dir=root / "vipe_results",
                colmap_root=root / "colmap",
                vipe_repo=root / "vipe",
                python_command="python3",
            )

        self.assertEqual(commands[0][2], str(root / "vipe_results" / "selfie"))
        self.assertEqual(commands[0][3:6], ["--sequence", "selfie", "--output"])
        self.assertEqual(commands[0][6], str(root / "colmap"))
        self.assertNotEqual(commands[0][6], str(root / "colmap" / "selfie"))

    def test_flatten_nested_colmap_outputs_moves_sequence_contents_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "colmap"
            nested = root / "selfie" / "selfie"
            nested.mkdir(parents=True)
            (nested / "points3D.txt").write_text("points", encoding="utf-8")
            (nested / "cameras.txt").write_text("cameras", encoding="utf-8")

            actions = flatten_nested_colmap_outputs(root, ["selfie"])

            self.assertEqual(len(actions), 2)
            self.assertTrue((root / "selfie" / "points3D.txt").exists())
            self.assertTrue((root / "selfie" / "cameras.txt").exists())
            self.assertFalse((root / "selfie" / "selfie").exists())

    def test_write_observation_colmap_text_keeps_image_point_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            points = [
                ColmapDepthPoint(
                    point_id=1,
                    xyz=(1.0, 2.0, 3.0),
                    rgb=(10, 20, 30),
                    image_id=1,
                    point2d_idx=0,
                    xy=(12.0, 34.0),
                ),
                ColmapDepthPoint(
                    point_id=2,
                    xyz=(4.0, 5.0, 6.0),
                    rgb=(40, 50, 60),
                    image_id=1,
                    point2d_idx=1,
                    xy=(56.0, 78.0),
                ),
            ]

            write_observation_colmap_text(
                output_dir=root,
                image_records=[{
                    "image_id": 1,
                    "pose": np.eye(4),
                    "camera_id": 1,
                    "name": "images/frame_000000.jpg",
                }],
                points=points,
            )

            images_text = (root / "images.txt").read_text(encoding="utf-8")
            points_text = (root / "points3D.txt").read_text(encoding="utf-8")

        self.assertIn("1 1.000000000 0.000000000 0.000000000 0.000000000", images_text)
        self.assertIn("12.000000 34.000000 1 56.000000 78.000000 2", images_text)
        self.assertIn("1 1.000000 2.000000 3.000000 10 20 30 0.000000 1 0", points_text)
        self.assertIn("2 4.000000 5.000000 6.000000 40 50 60 0.000000 1 1", points_text)

    def test_matrix_to_colmap_pose_identity(self):
        quat, trans = matrix_to_colmap_pose(np.eye(4))

        np.testing.assert_allclose(quat, [1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(trans, [0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()

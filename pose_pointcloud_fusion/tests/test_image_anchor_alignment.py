import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pose_pointcloud_fusion.image_anchor_alignment import (
    align_sam3d_keypoints_to_colmap,
    load_view_points3d_by_id,
    parse_colmap_images_observations,
    parse_colmap_points3d_by_id,
    write_image_anchor_alignment,
)


class ImageAnchorAlignmentTests(unittest.TestCase):
    def test_parse_colmap_images_observations_keeps_2d_point3d_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            images_path = Path(tmp) / "images.txt"
            images_path.write_text(
                "\n".join(
                    [
                        "# Image list with two lines per image",
                        "1 1 0 0 0 0 0 0 1 view_00_frame_000003.jpg",
                        "10.0 20.0 101 30.0 40.0 -1 50.0 60.0 102",
                        "2 1 0 0 0 0 0 0 1 view_00_frame_000004.jpg",
                        "15.0 25.0 201",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            parsed = parse_colmap_images_observations(images_path)

        self.assertEqual(sorted(parsed), [1, 2])
        self.assertEqual(parsed[1].frame_index, 3)
        self.assertEqual(len(parsed[1].observations), 2)
        self.assertEqual(parsed[1].observations[0].point3d_id, 101)
        np.testing.assert_allclose(parsed[1].observations[1].xy, [50.0, 60.0])

    def test_parse_colmap_images_observations_handles_empty_observation_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            images_path = Path(tmp) / "images.txt"
            images_path.write_text(
                "# header\n"
                "1 1 0 0 0 0 0 0 1 images/frame_000000.jpg\n"
                "\n"
                "2 1 0 0 0 0 0 0 1 images/frame_000001.jpg\n"
                "10.0 20.0 101\n",
                encoding="utf-8",
            )

            parsed = parse_colmap_images_observations(images_path)

        self.assertEqual(sorted(parsed), [1, 2])
        self.assertEqual(len(parsed[1].observations), 0)
        self.assertEqual(len(parsed[2].observations), 1)
        self.assertEqual(parsed[2].observations[0].point3d_id, 101)

    def test_load_view_points_applies_manifest_and_refined_alignment_transforms(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            points_path = root / "points3D.txt"
            points_path.write_text("7 1.0 0.0 0.0 255 0 0 0.1 1 0\n", encoding="utf-8")
            manifest_path = root / "view_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "views": [
                            {
                                "name": "view_00",
                                "camera_to_world": [
                                    [1.0, 0.0, 0.0, 10.0],
                                    [0.0, 1.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0, 0.0],
                                    [0.0, 0.0, 0.0, 1.0],
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            alignment_path = root / "refined_alignment.json"
            alignment_path.write_text(
                json.dumps(
                    {
                        "transforms": {
                            "view_00": {
                                "scale": 2.0,
                                "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                                "translation": [0.0, 5.0, 0.0],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            points = load_view_points3d_by_id(
                points_path,
                manifest_path=manifest_path,
                view_name="view_00",
                alignment_json=alignment_path,
            )

        np.testing.assert_allclose(points[7].xyz, [22.0, 5.0, 0.0])

    def test_align_uses_2d_observations_to_estimate_transform_without_replacing_skeleton_shape(self):
        keypoints2d = np.array(
            [
                [10.0, 20.0, 1.0],
                [30.0, 20.0, 1.0],
                [10.0, 50.0, 1.0],
            ],
            dtype=np.float64,
        )
        sam3d_keypoints = np.array(
            [
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_path = root / "images.txt"
            images_path.write_text(
                "\n".join(
                    [
                        "1 1 0 0 0 0 0 0 1 frame_000000.jpg",
                        "10.5 20.0 11 30.0 20.5 12 10.0 50.5 13 200.0 200.0 99",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            points_path = root / "points3D.txt"
            points_path.write_text(
                "\n".join(
                    [
                        "11 10.0 20.0 30.0 255 0 0 0.1 1 0",
                        "12 12.0 20.0 30.0 0 255 0 0.1 1 1",
                        "13 10.0 22.0 30.0 0 0 255 0.1 1 2",
                        "99 99.0 99.0 99.0 1 2 3 0.1 1 3",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            images = parse_colmap_images_observations(images_path)
            points = parse_colmap_points3d_by_id(points_path)
            result = align_sam3d_keypoints_to_colmap(
                keypoints2d=keypoints2d,
                keypoints3d=sam3d_keypoints,
                image_observations=images[1],
                points3d_by_id=points,
                radius_px=2.0,
                min_conf=0.5,
            )

        self.assertEqual(result.num_matches, 3)
        np.testing.assert_allclose(result.transform.scale, 2.0, atol=1e-6)
        np.testing.assert_allclose(result.transform.translation, [10.0, 20.0, 30.0], atol=1e-6)
        np.testing.assert_allclose(result.aligned_keypoints[:, :3], [[10.0, 20.0, 30.0], [12.0, 20.0, 30.0], [10.0, 22.0, 30.0]], atol=1e-6)
        np.testing.assert_allclose(result.aligned_keypoints[:, 3], [1.0, 1.0, 1.0])

        original_bone = np.linalg.norm(sam3d_keypoints[1, :3] - sam3d_keypoints[0, :3])
        aligned_bone = np.linalg.norm(result.aligned_keypoints[1, :3] - result.aligned_keypoints[0, :3])
        self.assertAlmostEqual(aligned_bone / original_bone, 2.0)

    def test_write_image_anchor_alignment_loads_sam3d_json_and_writes_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sam3d_json = root / "sam3d.json"
            sam3d_json.write_text(
                json.dumps(
                    {
                        "frame_number": 1,
                        "track_id": 7,
                        "keypoints2d": [[5.0, 5.0, 1.0], [15.0, 5.0, 1.0], [5.0, 15.0, 1.0]],
                        "keypoints3d_camera": [[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]],
                    }
                ),
                encoding="utf-8",
            )
            images_path = root / "images.txt"
            images_path.write_text(
                "1 1 0 0 0 0 0 0 1 frame_000001.jpg\n"
                "5.0 5.0 1 15.0 5.0 2 5.0 15.0 3\n",
                encoding="utf-8",
            )
            points_path = root / "points3D.txt"
            points_path.write_text(
                "1 1.0 2.0 3.0 255 0 0 0.1 1 0\n"
                "2 2.0 2.0 3.0 0 255 0 0.1 1 1\n"
                "3 1.0 3.0 3.0 0 0 255 0.1 1 2\n",
                encoding="utf-8",
            )
            output_json = root / "aligned.json"

            summary = write_image_anchor_alignment(
                sam3d_json=sam3d_json,
                images_txt=images_path,
                points3d_txt=points_path,
                image_id=1,
                output_json=output_json,
                radius_px=1.0,
                min_conf=0.5,
            )

            self.assertTrue(output_json.exists())
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(summary["num_matches"], 3)
            self.assertEqual(payload["frame_number"], 1)
            self.assertEqual(payload["track_id"], 7)
            self.assertEqual(len(payload["aligned_keypoints3d_world"]), 3)
            np.testing.assert_allclose(payload["transform"]["translation"], [1.0, 2.0, 3.0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()

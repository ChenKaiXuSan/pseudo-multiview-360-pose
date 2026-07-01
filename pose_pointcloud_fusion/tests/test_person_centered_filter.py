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

from pose_pointcloud_fusion.person_centered_filter import (
    filter_person_centered_points,
    load_person_keypoints,
    point_to_segments_distance,
    read_ascii_xyzrgb_ply,
    statistical_outlier_mask,
    voxel_density_outlier_mask,
    write_ascii_xyzrgb_ply,
)


class PersonCenteredFilterTests(unittest.TestCase):
    def test_load_person_keypoints_accepts_fused_and_frame_tracks_world_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fused = root / "track_0007" / "fused_keypoints3d.json"
            fused.parent.mkdir(parents=True)
            fused.write_text(json.dumps({
                "frame_number": 3,
                "track_id": 7,
                "fused_keypoints3d_world": [[1.0, 2.0, 3.0, 0.9], [4.0, 5.0, 6.0, 0.1]],
            }), encoding="utf-8")
            frame_tracks = root / "frame_tracks_world.json"
            frame_tracks.write_text(json.dumps({
                "tracks": [7],
                "source_files": [str(fused)],
            }), encoding="utf-8")

            from_fused = load_person_keypoints(fused, track_id=None, min_conf=0.5)
            from_tracks = load_person_keypoints(frame_tracks, track_id=7, min_conf=0.5)

        self.assertEqual(from_fused.values.shape, (2, 4))
        self.assertTrue(from_fused.mask_valid[0])
        self.assertFalse(from_fused.mask_valid[1])
        np.testing.assert_allclose(from_tracks.values[0, :3], [1.0, 2.0, 3.0])

    def test_point_to_segments_distance_measures_capsule_distance(self):
        points = np.array([[0.5, 0.2, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
        starts = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        ends = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)

        distances = point_to_segments_distance(points, starts, ends)

        np.testing.assert_allclose(distances, [0.2, 1.0], atol=1e-6)

    def test_filter_person_centered_points_scene_removes_body_and_far_points(self):
        keypoints = load_person_keypoints.from_rows(
            [[0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]],
            min_conf=0.5,
        )
        xyz = np.array([
            [0.05, 0.5, 0.0],
            [1.0, 0.5, 0.0],
            [10.0, 0.5, 0.0],
        ], dtype=np.float64)
        rgb = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)

        result = filter_person_centered_points(
            xyz,
            rgb,
            [keypoints],
            mode="scene",
            trajectory_radius=2.0,
            height_below=2.0,
            height_above=2.0,
            body_radius=0.2,
            outlier_filter="none",
            edges=[(0, 1)],
        )

        self.assertEqual(result.xyz.shape, (1, 3))
        np.testing.assert_allclose(result.xyz[0], [1.0, 0.5, 0.0])
        self.assertEqual(result.summary["removed_by_body"], 1)
        self.assertEqual(result.summary["removed_by_roi"], 1)

    def test_statistical_outlier_mask_removes_isolated_point(self):
        cluster = np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.0, 0.02, 0.0], [0.02, 0.02, 0.0], [10.0, 0.0, 0.0]], dtype=np.float64)

        mask = statistical_outlier_mask(cluster, k=2, std_ratio=0.5)

        self.assertEqual(mask.tolist(), [True, True, True, True, False])

    def test_voxel_density_outlier_mask_removes_sparse_point(self):
        points = np.array([
            [0.00, 0.00, 0.00],
            [0.01, 0.00, 0.00],
            [0.00, 0.01, 0.00],
            [5.00, 0.00, 0.00],
        ], dtype=np.float64)

        mask = voxel_density_outlier_mask(points, voxel_size=0.1, min_neighbors=3)

        self.assertEqual(mask.tolist(), [True, True, True, False])

    def test_ascii_ply_round_trip_xyzrgb(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cloud.ply"
            xyz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
            rgb = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)

            write_ascii_xyzrgb_ply(path, xyz, rgb)
            loaded_xyz, loaded_rgb = read_ascii_xyzrgb_ply(path)

        np.testing.assert_allclose(loaded_xyz, xyz)
        np.testing.assert_array_equal(loaded_rgb, rgb)


if __name__ == "__main__":
    unittest.main()

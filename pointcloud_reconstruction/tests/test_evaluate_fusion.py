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

from pointcloud_reconstruction.evaluate_fusion import (
    describe_values,
    nearest_neighbor_distances,
    point_cloud_metrics,
    trajectory_metrics,
)


class EvaluateFusionTests(unittest.TestCase):
    def test_describe_values_reports_basic_percentiles(self):
        stats = describe_values([1, 2, 3, 4])
        self.assertEqual(stats["count"], 4)
        self.assertAlmostEqual(stats["mean"], 2.5)
        self.assertAlmostEqual(stats["median"], 2.5)
        self.assertAlmostEqual(stats["min"], 1.0)
        self.assertAlmostEqual(stats["max"], 4.0)

    def test_nearest_neighbor_distances_uses_exact_distances(self):
        src = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        dst = np.array([[1.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
        distances = nearest_neighbor_distances(src, dst, chunk_size=1)
        self.assertTrue(np.allclose(distances, [1.0, 1.0]))

    def test_point_cloud_metrics_counts_points_and_error_stats(self):
        points = [
            {"xyz": [0.0, 0.0, 0.0], "error": 0.5},
            {"xyz": [1.0, 0.0, 0.0], "error": 1.5},
        ]
        metrics = point_cloud_metrics(points)
        self.assertEqual(metrics["point_count"], 2)
        self.assertAlmostEqual(metrics["reprojection_error"]["mean"], 1.0)
        self.assertEqual(metrics["bbox"]["min"], [0.0, 0.0, 0.0])
        self.assertEqual(metrics["bbox"]["max"], [1.0, 0.0, 0.0])

    def test_trajectory_metrics_reports_motion_steps(self):
        poses = np.tile(np.eye(4, dtype=np.float32), (3, 1, 1))
        poses[1, 0, 3] = 1.0
        poses[2, 0, 3] = 3.0
        metrics = trajectory_metrics(poses)
        self.assertEqual(metrics["frame_count"], 3)
        self.assertAlmostEqual(metrics["path_length"], 3.0)
        self.assertAlmostEqual(metrics["translation_step"]["mean"], 1.5)


if __name__ == "__main__":
    unittest.main()

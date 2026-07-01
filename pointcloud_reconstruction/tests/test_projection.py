import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pointcloud_reconstruction.projection import camera_to_world_matrix, xyz_to_equirectangular


class ProjectionTests(unittest.TestCase):
    def test_xyz_to_equirectangular_places_forward_at_image_center(self):
        _lon, _lat, x, y = xyz_to_equirectangular(0.0, 0.0, 1.0, 4000, 2000)
        self.assertAlmostEqual(float(x), 2000.0)
        self.assertAlmostEqual(float(y), 1000.0)

    def test_camera_to_world_matrix_has_forward_column(self):
        matrix = camera_to_world_matrix(yaw_deg=90.0, pitch_deg=0.0)
        self.assertAlmostEqual(float(matrix[0][2]), 1.0, places=6)
        self.assertAlmostEqual(float(matrix[1][2]), 0.0, places=6)
        self.assertAlmostEqual(float(matrix[2][2]), 0.0, places=6)
        self.assertEqual(matrix[3].tolist(), [0.0, 0.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()

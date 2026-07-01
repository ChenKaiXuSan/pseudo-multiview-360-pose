import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pointcloud_reconstruction.merge_colmap import find_colmap_points3d, read_colmap_points3d, transform_point, write_ascii_ply


class MergeColmapTests(unittest.TestCase):
    def test_read_colmap_points3d_skips_comments(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "points3D.txt"
            sample = "# header\n1 1.0 2.0 3.0 10 20 30 0.5 1 2 3\n"
            path.write_text(sample, encoding="utf-8")
            points = read_colmap_points3d(path)
        self.assertEqual(points[0]["xyz"], [1.0, 2.0, 3.0])
        self.assertEqual(points[0]["rgb"], [10, 20, 30])

    def test_find_colmap_points3d_accepts_vipe_nested_sequence_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "selfie" / "selfie" / "points3D.txt"
            nested.parent.mkdir(parents=True)
            nested.write_text("# points\n", encoding="utf-8")

            self.assertEqual(find_colmap_points3d(root, "selfie"), nested)

    def test_transform_point_applies_homogeneous_matrix(self):
        matrix = [
            [1.0, 0.0, 0.0, 10.0],
            [0.0, 1.0, 0.0, 20.0],
            [0.0, 0.0, 1.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        self.assertEqual(transform_point([1.0, 2.0, 3.0], matrix), [11.0, 22.0, 33.0])

    def test_write_ascii_ply_writes_vertex_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "merged.ply"
            write_ascii_ply(path, [{"xyz": [1, 2, 3], "rgb": [4, 5, 6]}])
            text = path.read_text(encoding="utf-8")
        self.assertIn("element vertex 1", text)
        self.assertIn("1.000000 2.000000 3.000000 4 5 6", text)


if __name__ == "__main__":
    unittest.main()

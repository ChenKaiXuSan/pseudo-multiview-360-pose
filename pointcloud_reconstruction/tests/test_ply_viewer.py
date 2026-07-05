import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pointcloud_reconstruction.ply_viewer import (
    discover_frame_plys,
    frame_number_from_ply_path,
    read_ascii_ply_xyzrgb,
    sample_evenly_spaced,
)


class PlyViewerTests(unittest.TestCase):
    def test_read_ascii_ply_xyzrgb_uses_header_property_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cloud.ply"
            path.write_text(
                "\n".join(
                    [
                        "ply",
                        "format ascii 1.0",
                        "element vertex 2",
                        "property uchar red",
                        "property float z",
                        "property uchar blue",
                        "property float x",
                        "property uchar green",
                        "property float y",
                        "end_header",
                        "10 3.0 30 1.0 20 2.0",
                        "40 6.0 60 4.0 50 5.0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            xyz, rgb = read_ascii_ply_xyzrgb(path)

        np.testing.assert_allclose(xyz, np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        np.testing.assert_array_equal(rgb, np.asarray([[10, 20, 30], [40, 50, 60]], dtype=np.uint8))

    def test_read_ascii_ply_xyzrgb_defaults_to_white_when_color_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cloud.ply"
            path.write_text(
                "\n".join(
                    [
                        "ply",
                        "format ascii 1.0",
                        "element vertex 1",
                        "property float x",
                        "property float y",
                        "property float z",
                        "end_header",
                        "1.0 2.0 3.0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            xyz, rgb = read_ascii_ply_xyzrgb(path)

        np.testing.assert_allclose(xyz, np.asarray([[1.0, 2.0, 3.0]]))
        np.testing.assert_array_equal(rgb, np.asarray([[255, 255, 255]], dtype=np.uint8))

    def test_read_ascii_ply_xyzrgb_can_sample_during_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cloud.ply"
            rows = [
                "ply",
                "format ascii 1.0",
                "element vertex 5",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "end_header",
            ]
            for idx in range(5):
                rows.append(f"{idx}.0 {idx + 1}.0 {idx + 2}.0 {idx} {idx + 10} {idx + 20}")
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")

            xyz, rgb = read_ascii_ply_xyzrgb(path, max_points=3)

        np.testing.assert_allclose(xyz, np.asarray([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0], [4.0, 5.0, 6.0]]))
        np.testing.assert_array_equal(rgb, np.asarray([[0, 10, 20], [2, 12, 22], [4, 14, 24]], dtype=np.uint8))

    def test_sample_evenly_spaced_is_deterministic_and_keeps_limit(self):
        xyz = np.arange(30, dtype=np.float64).reshape(10, 3)
        rgb = np.arange(30, dtype=np.uint8).reshape(10, 3)

        sampled_xyz, sampled_rgb = sample_evenly_spaced(xyz, rgb, max_points=4)

        np.testing.assert_array_equal(sampled_xyz, xyz[[0, 3, 6, 9]])
        np.testing.assert_array_equal(sampled_rgb, rgb[[0, 3, 6, 9]])

    def test_rejects_binary_ply(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cloud.ply"
            path.write_text(
                "\n".join(
                    [
                        "ply",
                        "format binary_little_endian 1.0",
                        "element vertex 1",
                        "property float x",
                        "property float y",
                        "property float z",
                        "end_header",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Only ASCII PLY"):
                read_ascii_ply_xyzrgb(path)


    def test_frame_number_from_ply_path_reads_zero_padded_frame_name(self):
        self.assertEqual(frame_number_from_ply_path(Path("frame_000064.ply")), 64)

    def test_discover_frame_plys_sorts_by_frame_number_and_applies_stride(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ["frame_000032.ply", "frame_000000.ply", "frame_000016.ply", "other.ply"]:
                (root / name).write_text("", encoding="utf-8")

            paths = discover_frame_plys(root, frame_stride=2)

        self.assertEqual([p.name for p in paths], ["frame_000000.ply", "frame_000032.ply"])

    def test_discover_frame_plys_applies_max_frames_after_stride(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for idx in [0, 16, 32, 48]:
                (root / f"frame_{idx:06d}.ply").write_text("", encoding="utf-8")

            paths = discover_frame_plys(root, frame_stride=1, max_frames=2)

        self.assertEqual([p.name for p in paths], ["frame_000000.ply", "frame_000016.ply"])


if __name__ == "__main__":
    unittest.main()

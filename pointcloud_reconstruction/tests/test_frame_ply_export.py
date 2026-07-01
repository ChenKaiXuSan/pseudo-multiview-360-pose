import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pointcloud_reconstruction.frame_ply_export import (
    apply_refined_view_transform,
    parse_colmap_image_frames,
    parse_colmap_points3d_with_tracks,
    transform_from_alignment,
)


class FramePlyExportTests(unittest.TestCase):
    def test_parse_colmap_image_frames_reads_frame_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "images.txt"
            path.write_text(
                "# header\n"
                "1 1 0 0 0 0 0 0 1 images/frame_000000.jpg\n\n"
                "2 1 0 0 0 0 0 0 1 images/frame_000017.jpg\n\n",
                encoding="utf-8",
            )

            frames = parse_colmap_image_frames(path)

        self.assertEqual(frames, {1: 0, 2: 17})

    def test_parse_colmap_points3d_with_tracks_keeps_positive_image_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "points3D.txt"
            path.write_text(
                "# header\n"
                "1 1.0 2.0 3.0 10 20 30 0.0 1 5 0 0 2 8\n",
                encoding="utf-8",
            )

            points = parse_colmap_points3d_with_tracks(path)

        self.assertEqual(points[0]["xyz"], [1.0, 2.0, 3.0])
        self.assertEqual(points[0]["image_ids"], [1, 2])

    def test_apply_refined_view_transform_uses_scale_rotation_translation(self):
        alignment = {
            "scale": 2.0,
            "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "translation": [10.0, 20.0, 30.0],
        }
        transform = transform_from_alignment(alignment)

        out = apply_refined_view_transform([1.0, 2.0, 3.0], transform)

        self.assertEqual(out, [12.0, 24.0, 36.0])


if __name__ == "__main__":
    unittest.main()

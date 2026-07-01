import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pointcloud_reconstruction.refine_alignment import (
    apply_similarity,
    build_adjacent_view_edges,
    select_best_alignment_candidate,
    estimate_similarity_umeyama,
    refine_manifest_colmap_points,
    refine_similarity_icp,
    similarity_warning,
)


class RefineAlignmentTests(unittest.TestCase):
    def test_build_adjacent_view_edges_orders_views_by_yaw(self):
        views = [
            {"name": "selfie", "yaw_deg": 5.0},
            {"name": "front_right", "yaw_deg": 65.0},
            {"name": "back_right", "yaw_deg": 125.0},
            {"name": "back", "yaw_deg": -175.0},
            {"name": "back_left", "yaw_deg": -115.0},
            {"name": "front_left", "yaw_deg": -55.0},
        ]

        edges = build_adjacent_view_edges(views)

        self.assertEqual(
            edges,
            [
                ("selfie", "front_right"),
                ("front_right", "back_right"),
                ("back_right", "back"),
                ("back", "back_left"),
                ("back_left", "front_left"),
                ("front_left", "selfie"),
            ],
        )

    def test_select_best_alignment_candidate_prefers_valid_lower_error_scale(self):
        candidates = [
            {"parent_view": "bad_scale", "scale": 0.05, "final_median_error": 0.01},
            {"parent_view": "noisy", "scale": 1.0, "final_median_error": 0.8},
            {"parent_view": "good", "scale": 0.9, "final_median_error": 0.2},
        ]

        selected, rejected = select_best_alignment_candidate(candidates, scale_min=0.2, scale_max=2.5)

        self.assertEqual(selected["parent_view"], "good")
        self.assertEqual(rejected[0]["parent_view"], "bad_scale")
        self.assertEqual(rejected[0]["reason"], "scale_out_of_range")

    def test_umeyama_recovers_known_similarity(self):
        src = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        )
        angle = np.deg2rad(30.0)
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        target = apply_similarity(src, scale=2.5, rotation=rotation, translation=np.array([3.0, -2.0, 0.5]))

        transform = estimate_similarity_umeyama(src, target)

        self.assertAlmostEqual(transform["scale"], 2.5, places=6)
        np.testing.assert_allclose(transform["rotation"], rotation, atol=1e-6)
        np.testing.assert_allclose(transform["translation"], [3.0, -2.0, 0.5], atol=1e-6)

    def test_icp_reduces_alignment_error(self):
        rng = np.random.default_rng(7)
        reference = rng.normal(size=(200, 3))
        shifted = reference * 1.7 + np.array([4.0, -3.0, 2.0])

        result = refine_similarity_icp(shifted, reference, max_iterations=20, trim_fraction=0.8, distance_threshold=1.0)

        before = np.linalg.norm(shifted - reference, axis=1).mean()
        after = np.linalg.norm(result.aligned_points - reference, axis=1).mean()
        self.assertLess(after, before * 0.05)
        self.assertLess(result.final_median_error, 1e-5)

    def test_similarity_warning_flags_extreme_scale(self):
        warning = similarity_warning(scale=0.02, median_error=12.0, max_scale_ratio=4.0, max_median_error=5.0)

        self.assertIn("scale_out_of_range", warning)
        self.assertIn("median_error_high", warning)

    def test_refine_manifest_colmap_points_writes_outputs(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {
                "views": [
                    {"name": "selfie", "camera_to_world": np.eye(4).tolist()},
                    {"name": "right", "camera_to_world": np.eye(4).tolist()},
                ]
            }
            manifest_path = root / "view_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            colmap_root = root / "colmap"
            for view_name, offset in [("selfie", 0.0), ("right", 10.0)]:
                view_dir = colmap_root / view_name
                view_dir.mkdir(parents=True)
                image_rows = [
                    "# header",
                    "1 1 0 0 0 0 0 0 1 images/frame_000000.jpg",
                    "",
                ]
                (view_dir / "images.txt").write_text("\n".join(image_rows), encoding="utf-8")
                rows = [
                    "# header",
                    f"1 {0 + offset} 0 0 255 0 0 0.0 1 0",
                    f"2 {1 + offset} 0 0 0 255 0 0.0 1 1",
                    f"3 {0 + offset} 1 0 0 0 255 0.0 1 2",
                    f"4 {0 + offset} 0 1 255 255 255 0.0 1 3",
                ]
                (view_dir / "points3D.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
            output_ply = root / "refined.ply"
            alignment_json = root / "alignment.json"

            summary = refine_manifest_colmap_points(
                manifest_path=manifest_path,
                colmap_root=colmap_root,
                output_ply=output_ply,
                alignment_json=alignment_json,
                reference_view="selfie",
                sample_points=100,
                max_iterations=8,
            )

            self.assertTrue(output_ply.exists())
            self.assertTrue(alignment_json.exists())
            self.assertEqual(summary["total_points"], 8)
            self.assertEqual(summary["transforms"]["selfie"]["status"], "reference")
            self.assertEqual(summary["transforms"]["right"]["status"], "aligned")
            self.assertEqual(summary["frame_plys"]["num_frames"], 1)
            self.assertTrue((root / "refined_frame_plys" / "frame_000000.ply").exists())


if __name__ == "__main__":
    unittest.main()

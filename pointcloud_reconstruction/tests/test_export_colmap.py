import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pointcloud_reconstruction.export_colmap import build_vipe_to_colmap_commands, flatten_nested_colmap_outputs


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


if __name__ == "__main__":
    unittest.main()

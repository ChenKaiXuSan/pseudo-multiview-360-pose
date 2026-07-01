import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pointcloud_reconstruction.direct_360 import (
    build_direct_360_colmap_command,
    build_direct_360_vipe_command,
    limited_video_path,
    write_direct_360_summary,
)


class Direct360Tests(unittest.TestCase):
    def test_build_direct_360_vipe_command_uses_source_video(self):
        command = build_direct_360_vipe_command(
            video_path=Path("/data/input.mp4"),
            output_dir=Path("/out/vipe_results"),
            vipe_command="vipe",
            pipeline="default",
            visualize=True,
        )

        self.assertEqual(command[:4], ["vipe", "infer", "/data/input.mp4", "--output"])
        self.assertEqual(command[4], "/out/vipe_results")
        self.assertIn("--visualize", command)

    def test_build_direct_360_colmap_command_exports_single_sequence(self):
        command = build_direct_360_colmap_command(
            vipe_results_dir=Path("/out/vipe_results"),
            colmap_root=Path("/out/colmap"),
            vipe_repo=Path("/repo/vipe"),
            python_command="python3",
            sequence_name="direct_360",
            depth_step=8,
            use_slam_map=True,
        )

        self.assertEqual(command[0], "python3")
        self.assertEqual(command[1], "/repo/vipe/scripts/vipe_to_colmap.py")
        self.assertEqual(command[2], "/out/vipe_results")
        self.assertEqual(command[3:6], ["--sequence", "direct_360", "--output"])
        self.assertEqual(command[6], "/out/colmap")
        self.assertIn("--use_slam_map", command)

    def test_write_direct_360_summary_records_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            summary = write_direct_360_summary(
                summary_path=path,
                source_video_path=Path("/data/input.mp4"),
                vipe_input_video_path=Path("/out/direct_360/input/direct_360_first_000120.mp4"),
                output_root=Path("/out/direct_360"),
                vipe_results_dir=Path("/out/direct_360/vipe_results"),
                colmap_root=Path("/out/direct_360/colmap"),
                vipe_command=["vipe", "infer"],
                colmap_command=["python3", "vipe_to_colmap.py"],
                ran_vipe=False,
                ran_colmap=False,
                max_frames=120,
            )

            self.assertTrue(path.exists())
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["source_video_path"], "/data/input.mp4")
            self.assertEqual(loaded["vipe_input_video_path"], "/out/direct_360/input/direct_360_first_000120.mp4")
            self.assertEqual(loaded["max_frames"], 120)
            self.assertEqual(loaded["vipe_results_dir"], "/out/direct_360/vipe_results")
            self.assertFalse(summary["ran_vipe"])

    def test_limited_video_path_is_inside_output_root(self):
        path = limited_video_path(Path("/out/direct_360"), max_frames=120)

        self.assertEqual(path, Path("/out/direct_360/input/direct_360_first_000120.mp4"))


if __name__ == "__main__":
    unittest.main()

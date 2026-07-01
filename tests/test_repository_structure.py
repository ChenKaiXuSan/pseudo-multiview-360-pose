from __future__ import annotations

import configparser
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class RepositoryStructureTests(unittest.TestCase):
    def test_root_developer_entrypoints_exist(self) -> None:
        self.assertTrue((REPO_ROOT / "README.md").exists())
        self.assertTrue((REPO_ROOT / "pyproject.toml").exists())
        self.assertTrue((REPO_ROOT / ".gitignore").exists())

    def test_pipeline_modules_have_expected_directories(self) -> None:
        for module in ("pose3d_kpt", "pointcloud_reconstruction", "pose_pointcloud_fusion"):
            with self.subTest(module=module):
                module_root = REPO_ROOT / module
                self.assertTrue(module_root.is_dir())
                self.assertTrue((module_root / "README.md").exists())
                self.assertTrue((module_root / "scripts").is_dir())
                self.assertTrue((module_root / "tests").is_dir())

    def test_pointcloud_and_fusion_use_named_packages_not_top_level_src(self) -> None:
        self.assertTrue((REPO_ROOT / "pointcloud_reconstruction" / "src" / "pointcloud_reconstruction").is_dir())
        self.assertFalse((REPO_ROOT / "pointcloud_reconstruction" / "src" / "__init__.py").exists())
        self.assertTrue((REPO_ROOT / "pose_pointcloud_fusion" / "src" / "pose_pointcloud_fusion").is_dir())
        self.assertFalse((REPO_ROOT / "pose_pointcloud_fusion" / "src" / "__init__.py").exists())

    def test_sam3d_submodule_mapping_matches_new_location(self) -> None:
        gitmodules = REPO_ROOT / ".gitmodules"
        self.assertTrue(gitmodules.exists())
        parser = configparser.ConfigParser()
        parser.read(gitmodules)
        section = 'submodule "pose3d_kpt/third_party/sam-3d-body"'
        self.assertTrue(parser.has_section(section))
        self.assertEqual(parser.get(section, "path"), "pose3d_kpt/third_party/sam-3d-body")
        self.assertEqual(parser.get(section, "url"), "https://github.com/facebookresearch/sam-3d-body.git")


if __name__ == "__main__":
    unittest.main()

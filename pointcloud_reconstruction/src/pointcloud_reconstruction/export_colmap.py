from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def load_view_names(manifest_path: Path) -> list[str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [str(view["name"]) for view in manifest.get("views", [])]


def build_vipe_to_colmap_commands(
    *,
    manifest_path: Path,
    vipe_results_dir: Path,
    colmap_root: Path,
    vipe_repo: Path,
    python_command: str = "python",
    depth_step: int = 16,
    use_slam_map: bool = False,
) -> list[list[str]]:
    """Build commands that export each view directly to colmap_root/<view>."""
    commands: list[list[str]] = []
    script_path = vipe_repo / "scripts" / "vipe_to_colmap.py"
    for view_name in load_view_names(manifest_path):
        command = [
            python_command,
            str(script_path),
            str(vipe_results_dir / view_name),
            "--sequence",
            view_name,
            "--output",
            str(colmap_root),
            "--depth_step",
            str(int(depth_step)),
        ]
        if use_slam_map:
            command.append("--use_slam_map")
        commands.append(command)
    return commands


def flatten_nested_colmap_outputs(colmap_root: Path, view_names: list[str]) -> list[dict[str, str]]:
    """Move colmap/<view>/<view>/* up to colmap/<view> when old nested outputs exist."""
    actions: list[dict[str, str]] = []
    for view_name in view_names:
        view_dir = colmap_root / view_name
        nested_dir = view_dir / view_name
        if not nested_dir.exists() or not nested_dir.is_dir():
            continue
        view_dir.mkdir(parents=True, exist_ok=True)
        for child in sorted(nested_dir.iterdir()):
            target = view_dir / child.name
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite existing COLMAP output: {target}")
            shutil.move(str(child), str(target))
            actions.append({"from": str(child), "to": str(target)})
        nested_dir.rmdir()
    return actions


def export_colmap_views(
    *,
    manifest_path: Path,
    vipe_results_dir: Path,
    colmap_root: Path,
    vipe_repo: Path,
    python_command: str = "python",
    depth_step: int = 16,
    use_slam_map: bool = False,
    dry_run: bool = False,
) -> list[list[str]]:
    commands = build_vipe_to_colmap_commands(
        manifest_path=manifest_path,
        vipe_results_dir=vipe_results_dir,
        colmap_root=colmap_root,
        vipe_repo=vipe_repo,
        python_command=python_command,
        depth_step=depth_step,
        use_slam_map=use_slam_map,
    )
    colmap_root.mkdir(parents=True, exist_ok=True)
    for command in commands:
        print(" ".join(command))
        if not dry_run:
            subprocess.run(command, check=True, cwd=str(vipe_repo))
    if not dry_run:
        flatten_nested_colmap_outputs(colmap_root, load_view_names(manifest_path))
    return commands

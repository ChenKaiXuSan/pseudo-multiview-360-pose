#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ViPE once for each extracted virtual view.")
    parser.add_argument("--manifest", required=True, type=Path, help="view_manifest.json from extract_dynamic_views.py.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for per-view ViPE outputs.")
    parser.add_argument("--vipe-command", default="vipe", help="ViPE CLI command, usually 'vipe'.")
    parser.add_argument("--pipeline", default="default", help="ViPE pipeline name.")
    parser.add_argument("--visualize", action="store_true", help="Pass --visualize to vipe infer.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser.parse_args()


def build_commands(manifest: dict, output_dir: Path, vipe_command: str, pipeline: str, visualize: bool) -> list[list[str]]:
    commands = []
    for view in manifest.get("views", []):
        view_output = output_dir / view["name"]
        command = [
            vipe_command,
            "infer",
            view["video_path"],
            "--output",
            str(view_output),
            "--pipeline",
            pipeline,
        ]
        if visualize:
            command.append("--visualize")
        commands.append(command)
    return commands


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commands = build_commands(manifest, args.output_dir, args.vipe_command, args.pipeline, args.visualize)
    for command in commands:
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

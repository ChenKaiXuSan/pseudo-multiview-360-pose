#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pointcloud_reconstruction.ply_viewer import run_ply_sequence_viewer, run_ply_viewer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View fused ASCII PLY point clouds or frame PLY sequences in a browser with viser.")
    parser.add_argument(
        "--ply",
        action="append",
        default=None,
        type=Path,
        help="ASCII xyzrgb PLY path. Repeat to overlay several static clouds in the same viewer.",
    )
    parser.add_argument(
        "--frame-dir",
        type=Path,
        default=None,
        help="Directory containing frame_*.ply files to play as a timeline.",
    )
    parser.add_argument(
        "--name",
        action="append",
        default=None,
        help="Optional layer name. Repeat in the same order as --ply.",
    )
    parser.add_argument("--port", type=int, default=20542, help="viser server port.")
    parser.add_argument("--host", default=None, help="Server host/IP. Defaults to the machine LAN IP when available.")
    parser.add_argument(
        "--max-points",
        type=int,
        default=200000,
        help="Deterministic per-PLY point limit for responsive browser rendering. Use 0 to load all points.",
    )
    parser.add_argument("--point-size", type=float, default=0.01, help="Point size used by viser.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Use every Nth frame when --frame-dir is set.")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of sequence frames to load when --frame-dir is set.")
    parser.add_argument("--fps", type=float, default=6.0, help="Initial playback FPS when --frame-dir is set.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_points = None if args.max_points <= 0 else args.max_points
    if args.frame_dir is not None:
        run_ply_sequence_viewer(
            args.frame_dir,
            port=args.port,
            host=args.host,
            max_points=max_points,
            point_size=args.point_size,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
            fps=args.fps,
        )
        return

    if not args.ply:
        raise SystemExit("Either --ply or --frame-dir is required.")
    run_ply_viewer(
        args.ply,
        names=args.name,
        port=args.port,
        host=args.host,
        max_points=max_points,
        point_size=args.point_size,
    )


if __name__ == "__main__":
    main()

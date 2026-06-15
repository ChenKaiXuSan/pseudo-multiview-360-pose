#!/usr/bin/env python3
"""Remove exact duplicate files from SAM3D multiview output trees.

Dry-run is the default. Use --execute only after reviewing the printed summary.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def same_file_content(a: Path, b: Path) -> bool:
    if not a.exists() or not b.exists():
        return False
    if a.stat().st_size != b.stat().st_size:
        return False
    return file_sha256(a) == file_sha256(b)


def duplicate_candidates(root: Path, include_view_sam3d: bool = False) -> list[tuple[Path, Path, str]]:
    """Return (duplicate_path, canonical_path, reason) for exact duplicates."""
    candidates: list[tuple[Path, Path, str]] = []
    for track_dir in sorted(root.glob("frame_*/track_*")):
        for name in ("fused_keypoints3d.json", "fused_keypoints3d_world.npz"):
            duplicate = track_dir / name
            canonical = track_dir / "fused" / name
            if same_file_content(duplicate, canonical):
                candidates.append((duplicate, canonical, "legacy fused root duplicate"))

    if include_view_sam3d:
        for duplicate in sorted(root.glob("frame_*/track_*/views/view_*/sam3d.json")):
            rel = duplicate.relative_to(root)
            parts = rel.parts
            if len(parts) < 5:
                continue
            canonical = root / "sam3d_results" / parts[0] / parts[1] / parts[3] / "sam3d.json"
            if same_file_content(duplicate, canonical):
                candidates.append((duplicate, canonical, "legacy per-view sam3d duplicate"))
    return candidates


def cleanup(root: Path, include_view_sam3d: bool, execute: bool) -> dict[str, int]:
    candidates = duplicate_candidates(root, include_view_sam3d=include_view_sam3d)
    bytes_to_remove = sum(path.stat().st_size for path, _canonical, _reason in candidates)
    by_reason: dict[str, int] = {}
    for _path, _canonical, reason in candidates:
        by_reason[reason] = by_reason.get(reason, 0) + 1

    print(f"Root: {root}")
    print(f"Exact duplicate files: {len(candidates)}")
    print(f"Bytes removable: {bytes_to_remove}")
    for reason, count in sorted(by_reason.items()):
        print(f"  {reason}: {count}")
    for path, canonical, reason in candidates[:20]:
        print(f"  {reason}: remove {path} (same as {canonical})")
    if len(candidates) > 20:
        print(f"  ... {len(candidates) - 20} more")

    if execute:
        for path, _canonical, _reason in candidates:
            path.unlink()
        print(f"Deleted {len(candidates)} duplicate files.")
    else:
        print("Dry-run only. Re-run with --execute to delete these files.")
    return {"files": len(candidates), "bytes": bytes_to_remove}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean exact duplicate SAM3D multiview output files.")
    parser.add_argument("root", type=Path, help="Video-level output root, e.g. sam3d_body_multiview/kimura2_360")
    parser.add_argument("--include-view-sam3d", action="store_true", help="also delete old views/view_xx/sam3d.json files when identical to sam3d_results")
    parser.add_argument("--execute", action="store_true", help="delete files; default only prints a dry-run summary")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.root.exists():
        raise FileNotFoundError(args.root)
    cleanup(args.root, include_view_sam3d=args.include_view_sam3d, execute=args.execute)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

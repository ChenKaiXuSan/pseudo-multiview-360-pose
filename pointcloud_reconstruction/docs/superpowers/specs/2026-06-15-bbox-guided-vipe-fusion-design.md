# BBox-Guided VIPE Fusion Design

## Goal

Create an independent wrapper project under `/mnt/dataset/skiing/360PoseFusion/pointcloud_reconstruction` that cuts perspective views from a 360 video based on the selfie skier bbox, runs VIPE on each view, and prepares merged scene geometry artifacts.

## Architecture

The project does not copy or modify `/mnt/dataset/skiing/vipe/vipe`. It calls VIPE as an external tool and stores all generated artifacts under `pointcloud_reconstruction/outputs` by default.

The bbox JSON from `360PoseFusion` is the control signal. The initial implementation uses a sequence-level circular mean of the selfie bbox yaw to define a stable anchor. Fixed virtual cameras are generated around that anchor so VIPE receives stable per-view videos.

## Data Flow

1. Read `360PoseFusion` bbox JSON with `target=selfie` and `bbox_format=xyxy`.
2. Select `target_id=1` when present; otherwise fall back to the highest scoring bbox in each frame.
3. Convert bbox horizontal centers to equirectangular yaw using the same convention as `test_360_detection.py`: image center is yaw 0, left/right wrap at +/-180.
4. Compute circular mean yaw across valid frames.
5. Generate fixed views: selfie, right, left, back.
6. Project each source frame to square perspective mp4 files.
7. Save `view_manifest.json` with source paths, view yaw/pitch/FOV, output video paths, and camera-to-world rotation matrices.
8. Run VIPE per view through `vipe infer`.
9. Convert VIPE outputs to COLMAP with VIPE's `scripts/vipe_to_colmap.py`.
10. Rough-merge COLMAP `points3D.txt` files into an ASCII PLY using view rotations.

## Known Limitation

Independent VIPE/COLMAP reconstructions can differ in scale, origin, and drift. The first merge is a qualitative inspection artifact, not a final metric reconstruction. A later version should estimate similarity transforms between views using overlapping geometry, camera trajectories, or shared 360-view constraints.

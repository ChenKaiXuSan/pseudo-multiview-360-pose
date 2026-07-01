# pointcloud_reconstruction

Multi-view VIPE scene reconstruction from 360 video.

This module cuts several fixed perspective views from a 360 equirectangular video, runs VIPE independently on each view, exports VIPE results to COLMAP text format, and merges/refines the per-view point clouds in a shared 360/world frame.

## Inputs

- 360 video, for example `/mnt/dataset/skiing/360test/kimura2_360.mp4`
- Selfie/person bbox JSON from `pose3d_kpt`, used only to choose a stable yaw anchor
- External VIPE checkout, for example `/mnt/dataset/skiing/vipe/vipe`

## Main Pipeline

From repository root `/mnt/dataset/skiing/360PoseFusion`:

```bash
python3 pointcloud_reconstruction/scripts/extract_dynamic_views.py   --video /mnt/dataset/skiing/360test/kimura2_360.mp4   --bbox-json /mnt/dataset/skiing/360tracker_outputs/kimura2_360/kimura2_360_cotracker_selfie_yolo_bboxes.json   --output-dir /mnt/dataset/skiing/360PoseFusion/pointcloud_reconstruction/outputs/kimura2_360   --view-size 1024   --fov-deg 100   --yaw-offsets 0,60,120,180,-120,-60   --max-frames 120
```

```bash
python3 pointcloud_reconstruction/scripts/run_vipe_views.py   --manifest /mnt/dataset/skiing/360PoseFusion/pointcloud_reconstruction/outputs/kimura2_360/view_manifest.json   --output-dir /mnt/dataset/skiing/360PoseFusion/pointcloud_reconstruction/outputs/kimura2_360/vipe_results   --pipeline default
```

```bash
python3 pointcloud_reconstruction/scripts/export_colmap_views.py   --manifest /mnt/dataset/skiing/360PoseFusion/pointcloud_reconstruction/outputs/kimura2_360/view_manifest.json   --vipe-results-dir /mnt/dataset/skiing/360PoseFusion/pointcloud_reconstruction/outputs/kimura2_360/vipe_results   --colmap-root /mnt/dataset/skiing/360PoseFusion/pointcloud_reconstruction/outputs/kimura2_360/colmap   --vipe-repo /mnt/dataset/skiing/vipe/vipe
```

```bash
python3 pointcloud_reconstruction/scripts/refine_vipe_views.py   --manifest /mnt/dataset/skiing/360PoseFusion/pointcloud_reconstruction/outputs/kimura2_360/view_manifest.json   --colmap-root /mnt/dataset/skiing/360PoseFusion/pointcloud_reconstruction/outputs/kimura2_360/colmap   --output-ply /mnt/dataset/skiing/360PoseFusion/pointcloud_reconstruction/outputs/kimura2_360/refined_views_world.ply   --alignment-json /mnt/dataset/skiing/360PoseFusion/pointcloud_reconstruction/outputs/kimura2_360/refined_alignment.json   --reference-view selfie   --sample-points 8000   --max-iterations 12   --trim-fraction 0.7   --max-error 5.0
```

For six-view experiments, `--alignment-mode adjacent --scale-min -1 --scale-max -1` enables adjacent-view propagation without scale clamping.

## Outputs

```text
outputs/<sequence>/views/                         # extracted perspective videos
outputs/<sequence>/vipe_results/                  # raw VIPE outputs
outputs/<sequence>/colmap/                        # flat COLMAP text exports
outputs/<sequence>/merged_views_world.ply         # rough point cloud merge
outputs/<sequence>/refined_views_world*.ply       # refined point cloud merge
outputs/<sequence>/refined_frame_plys*/           # frame-level PLY files when available
```

## File Map

- `src/bbox_views.py`: bbox JSON parsing and yaw-anchor view layout.
- `src/projection.py`: equirectangular-to-perspective projection and camera matrices.
- `src/extract_views.py`: perspective video extraction.
- `src/merge_colmap.py`: COLMAP `points3D.txt` parsing and rough PLY merge.
- `src/refine_alignment.py`: Sim3/ICP view alignment and frame-level PLY export hooks.
- `src/frame_ply_export.py`: refined per-frame PLY export.
- `src/evaluate_fusion.py`: no-GT point-cloud consistency metrics.
- `scripts/`: CLI wrappers for each pipeline stage.

Person-centered filtering and pose overlays live in `../pose_pointcloud_fusion`.

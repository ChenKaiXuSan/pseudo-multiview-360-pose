# pseudo-multiview-360-pose

This repository is organized into three pipeline modules for 360-video human pose and scene reconstruction work.

## Repository Layout

```text
pose3d_kpt/
  Multi-perspective 360 human 3D keypoint estimation.

pointcloud_reconstruction/
  Multi-perspective VIPE scene reconstruction and point-cloud alignment.

pose_pointcloud_fusion/
  Person-centered filtering and visualization that combines 3D keypoints with point clouds.
```

## Developer Setup

From the repository root, install the Python packages in editable mode when you want importable modules and `posefusion360-*` console scripts:

```bash
cd /mnt/dataset/skiing/360PoseFusion
python3 -m pip install -e . --no-deps
```

The point-cloud and pose-pointcloud tools can also be run directly via their `scripts/` paths shown below.

## 1. `pose3d_kpt`: Multi-View 3D KPT

Goal: track a person in a 360 equirectangular video, render multiple perspective views around the person, run SAM3D Body per view, transform predictions into a shared world frame, and fuse 3D keypoints.

Main entry points:

```bash
cd /mnt/dataset/skiing/360PoseFusion/pose3d_kpt
python3 scripts/run_yolo_tracking.py
python3 scripts/run_multiview_fusion.py --max-frames 1
python3 scripts/run_full_pipeline.py
```

Important outputs:

```text
sam3d_body_multiview/<sequence>/frame_XXXXXX/track_XXXX/fused/fused_keypoints3d.json
sam3d_body_multiview/<sequence>/multiview_fused_keypoints3d.json
```

## 2. `pointcloud_reconstruction`: VIPE Point Cloud

Goal: cut fixed perspective views from the 360 video, run VIPE on each view, export COLMAP text outputs, merge them into a shared world point cloud, and optionally refine view alignment.

Main entry points:

```bash
cd /mnt/dataset/skiing/360PoseFusion
python3 pointcloud_reconstruction/scripts/extract_dynamic_views.py --help
python3 pointcloud_reconstruction/scripts/run_vipe_views.py --help
python3 pointcloud_reconstruction/scripts/export_colmap_views.py --help
python3 pointcloud_reconstruction/scripts/refine_vipe_views.py --help
```

Important outputs:

```text
pointcloud_reconstruction/outputs/<sequence>/views/
pointcloud_reconstruction/outputs/<sequence>/vipe_results/
pointcloud_reconstruction/outputs/<sequence>/colmap/
pointcloud_reconstruction/outputs/<sequence>/refined_views_world*.ply
pointcloud_reconstruction/outputs/<sequence>/refined_frame_plys*/frame_XXXXXX.ply
```

## 3. `pose_pointcloud_fusion`: 3D KPT + Point Cloud

Goal: use 3D human keypoints as an anchor to filter or visualize scene point clouds around a person.

Main entry points:

```bash
cd /mnt/dataset/skiing/360PoseFusion
python3 pose_pointcloud_fusion/scripts/filter_person_centered_cloud.py --help
python3 pose_pointcloud_fusion/scripts/overlay_pose_pointcloud.py --help
```

Typical use cases:

- `scene` mode: keep scene points around the person trajectory and remove points close to the moving body skeleton.
- `human` mode: keep only points close to the body skeleton.
- overlay mode: append visible 3D pose joints and bones to a PLY for inspection.

## Testing

Run module tests separately:

```bash
cd /mnt/dataset/skiing/360PoseFusion
python3 -m unittest discover -s pointcloud_reconstruction/tests -v
python3 -m unittest discover -s pose_pointcloud_fusion/tests -v
```

For `pose3d_kpt`, many tests depend on local SAM3D/CUDA/model assets. Use targeted tests when those dependencies are available.

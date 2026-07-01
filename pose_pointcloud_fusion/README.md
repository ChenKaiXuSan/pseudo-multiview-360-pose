# pose_pointcloud_fusion

Tools for combining 3D human keypoints with reconstructed point clouds.

This module consumes outputs from:

- `pose3d_kpt`: fused SAM3D world keypoints, usually `fused_keypoints3d.json`
- `pointcloud_reconstruction`: global or frame-level VIPE/COLMAP PLY files

## Person-Centered Filtering

Use a 3D pose as the center of a point-cloud filter. `scene` mode keeps nearby scene points and removes points close to the moving body skeleton. `human` mode keeps points near the skeleton.

```bash
python3 pose_pointcloud_fusion/scripts/filter_person_centered_cloud.py   --input-ply /mnt/dataset/skiing/360PoseFusion/pointcloud_reconstruction/outputs/kimura2_360/refined_frame_plys/frame_000000.ply   --pose-json /mnt/dataset/skiing/sam3d_body_multiview/kimura2_360/frame_000001/track_0001/fused/fused_keypoints3d.json   --output-ply /mnt/dataset/skiing/360PoseFusion/pose_pointcloud_fusion/outputs/person_centered_frame_000000_track_0001.ply   --mode scene   --trajectory-radius 8.0   --height-below 2.0   --height-above 3.0   --body-radius 0.35   --outlier-filter voxel
```

## Pose Overlay

Append visible joint and bone marker points to a PLY for inspection:

```bash
python3 pose_pointcloud_fusion/scripts/overlay_pose_pointcloud.py   --scene-ply /mnt/dataset/skiing/360PoseFusion/pose_pointcloud_fusion/outputs/person_centered_frame_000000_track_0001.ply   --pose-root /mnt/dataset/skiing/sam3d_body_multiview/kimura2_360   --output-dir /mnt/dataset/skiing/360PoseFusion/pose_pointcloud_fusion/outputs/overlay   --start-frame 1   --max-frames 1   --track-id 1   --joint-radius 0.25   --bone-step 0.01
```

## File Map

- `src/person_centered_filter.py`: PLY IO, pose loading, trajectory ROI, skeleton capsule filtering, voxel/statistical outlier filtering.
- `src/pose_pointcloud_overlay.py`: append pose markers/bone samples to scene PLYs and save previews.
- `scripts/filter_person_centered_cloud.py`: CLI for person-centered scene/human filtering.
- `scripts/overlay_pose_pointcloud.py`: CLI for pose and point-cloud overlay.
- `tests/`: unit tests for filtering and overlay helpers.

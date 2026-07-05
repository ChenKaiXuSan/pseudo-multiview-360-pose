# pose_pointcloud_fusion

Tools for combining 3D human keypoints with reconstructed point clouds.

This module consumes outputs from:

- `pose3d_kpt`: fused SAM3D world keypoints, usually `fused_keypoints3d.json`
- `pointcloud_reconstruction`: global or frame-level VIPE/COLMAP PLY files

## Person-Centered Filtering

Use a 3D pose as the center of a point-cloud filter. `scene` mode keeps nearby scene points and removes points close to the moving body skeleton. `human` mode keeps points near the skeleton.

```bash
python3 pose_pointcloud_fusion/scripts/filter_person_centered_cloud.py   --input-ply /mnt/dataset/skiing/360PoseFusion/output/pointcloud_reconstruction/multiview_4view/kimura2_360/refined_frame_plys/frame_000000.ply   --pose-json /mnt/dataset/skiing/360PoseFusion/output/pose3d_kpt/sam3d_multiview/kimura2_360/frame_000001/track_0001/fused/fused_keypoints3d.json   --output-ply /mnt/dataset/skiing/360PoseFusion/output/pose_pointcloud_fusion/person_centered_frame_000000_track_0001.ply   --mode scene   --trajectory-radius 8.0   --height-below 2.0   --height-above 3.0   --body-radius 0.35   --outlier-filter voxel
```

## Pose Overlay

Append visible joint and bone marker points to a PLY for inspection:

```bash
python3 pose_pointcloud_fusion/scripts/overlay_pose_pointcloud.py   --scene-ply /mnt/dataset/skiing/360PoseFusion/output/pose_pointcloud_fusion/person_centered_frame_000000_track_0001.ply   --pose-root /mnt/dataset/skiing/360PoseFusion/output/pose3d_kpt/sam3d_multiview/kimura2_360   --output-dir /mnt/dataset/skiing/360PoseFusion/output/pose_pointcloud_fusion/overlay   --start-frame 1   --max-frames 1   --track-id 1   --joint-radius 0.25   --bone-step 0.01
```

## Image-Anchored Pose Alignment

Use the image plane as a bridge between SAM3D and VIPE/COLMAP. This keeps the
SAM3D 3D skeleton as the human-shape source, matches SAM3D 2D joints to nearby
COLMAP 2D observations in the same perspective image, then estimates one
similarity transform that anchors the whole skeleton to the point-cloud
coordinate system.

```bash
python3 pose_pointcloud_fusion/scripts/align_pose_to_pointcloud_by_image.py \
  --sam3d-json /mnt/dataset/skiing/360PoseFusion/output/pose3d_kpt/sam3d_multiview/kimura2_360/sam3d_results/frame_000001/track_0001/view_00/sam3d.json \
  --images-txt /mnt/dataset/skiing/360PoseFusion/output/pointcloud_reconstruction/multiview_4view/kimura2_360/colmap/view_00/images.txt \
  --points3d-txt /mnt/dataset/skiing/360PoseFusion/output/pointcloud_reconstruction/multiview_4view/kimura2_360/colmap/view_00/points3D.txt \
  --manifest /mnt/dataset/skiing/360PoseFusion/output/pointcloud_reconstruction/multiview_4view/kimura2_360/view_manifest.json \
  --view-name view_00 \
  --alignment-json /mnt/dataset/skiing/360PoseFusion/output/pointcloud_reconstruction/multiview_4view/kimura2_360/refined_alignment.json \
  --image-id 1 \
  --output-json /mnt/dataset/skiing/360PoseFusion/output/pose_pointcloud_fusion/image_anchor/image_anchor/frame_000001_track_0001_view_00.json \
  --radius-px 8.0 \
  --min-conf 0.3
```

The output `aligned_keypoints3d_world` is `T(SAM3D_3D_skeleton)`. Matched
point-cloud points are used as anchors for estimating `T`; they do not replace
individual joints. When `--manifest` and `--alignment-json` are provided, anchors
are transformed into the same rough/refined point-cloud coordinates used by the
merged VIPE reconstruction.

## File Map

- `src/person_centered_filter.py`: PLY IO, pose loading, trajectory ROI, skeleton capsule filtering, voxel/statistical outlier filtering.
- `src/pose_pointcloud_overlay.py`: append pose markers/bone samples to scene PLYs and save previews.
- `src/image_anchor_alignment.py`: SAM3D 2D/3D keypoint to COLMAP 2D observation matching, similarity alignment, and JSON output.
- `scripts/filter_person_centered_cloud.py`: CLI for person-centered scene/human filtering.
- `scripts/overlay_pose_pointcloud.py`: CLI for pose and point-cloud overlay.
- `scripts/align_pose_to_pointcloud_by_image.py`: CLI for image-anchored SAM3D skeleton alignment to COLMAP/VIPE point clouds.
- `tests/`: unit tests for filtering and overlay helpers.

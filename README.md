# pseudo-multiview-360-pose

Pseudo-multiview 3D human pose estimation from monocular 360 videos using CoTracker, virtual perspective views, and SAM3D Body fusion.

## Overview

This repository is for a research project on pseudo-multiview 3D human pose estimation from monocular 360-degree videos.

The method:

1. uses YOLO and CoTracker to stably track a person's bounding box in an equirectangular video,
2. generates multiple virtual perspective views around the tracked human direction,
3. runs SAM3D Body independently on each virtual view to obtain camera-space 3D keypoints,
4. transforms the predicted keypoints into a shared world coordinate system using the known virtual camera yaw and pitch, and
5. fuses the multiview results to produce a more stable 3D human pose estimate.

## Main Scripts

- `cotracker_person_tracking.py`: YOLO detection, pose/grid query generation, CoTracker propagation, bbox reconstruction, and track ID association.
- `sam3d_body_multiview_fusion.py`: equirectangular-to-perspective rendering, projected bbox generation, SAM3D Body execution, camera-to-world transform, and multiview 3D keypoint fusion.
- `framewise_person_detection.py`: frame-by-frame baseline detection.
- `test_360_detection.py`: 360 cubemap detection utilities and experiments.
- `vlm_video_analyze.py`: VLM-based video frame analysis.

## Example Commands

```bash
python cotracker_person_tracking.py
python sam3d_body_multiview_fusion.py --max-frames 1
python framewise_person_detection.py
```

Most scripts expect local video, model, checkpoint, or SAM3D Body paths to be configured in each script's `CONFIG` dictionary or passed by CLI flags where available.

## Short Description

> A research project for pseudo-multiview 3D human pose estimation from monocular 360-degree videos, using CoTracker-stabilized person tracking, spherical-to-perspective virtual view generation, SAM3D Body inference, and camera-to-world keypoint fusion.

## Suggested GitHub Description

> Pseudo-multiview 3D human pose estimation from monocular 360 videos using CoTracker, virtual perspective views, and SAM3D Body fusion.

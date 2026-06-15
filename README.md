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

- `cotracker_person_tracking_yolo.py`: YOLO detection, pose/grid query generation, CoTracker propagation, bbox reconstruction, and track ID association.
- `sam3d_body_multiview_fusion.py`: equirectangular-to-perspective rendering, projected bbox generation, SAM3D Body execution, camera-to-world transform, and multiview 3D keypoint fusion.
- `framewise_person_detection.py`: frame-by-frame baseline detection.
- `test_360_detection.py`: 360 cubemap detection utilities and experiments.
- `vlm_video_analyze.py`: VLM-based video frame analysis.

## Project Layout

The project is organized around two stages: tracking people in 360 videos, then
creating pseudo-multiview views for SAM3D Body and fused 3D pose output. The
original root-level scripts remain compatible entry points while the package
layout grows around those stages.

```text
360PoseFusion/
|-- configs/
|   |-- tracking.yaml
|   |-- multiview_fusion.yaml
|   `-- paths.example.yaml
|-- scripts/
|   |-- run_tracking.py
|   |-- run_multiview_fusion.py
|   |-- run_direct_360_compare.py
|   `-- run_full_pipeline.py
|-- src/posefusion360/
|   |-- io/               # video, tracking JSON, and output layout helpers
|   |-- geometry/         # spherical, perspective, and camera/world math
|   |-- tracking/         # stage 1: YOLO/CoTracker person tracking
|   |-- multiview/        # stage 2: virtual views and world-space fusion
|   |-- sam3d/            # SAM3D Body runner and result payload handling
|   |-- visualization/    # frame/world/summary visualizations
|   `-- pipelines/        # tracking, multiview, and full-pipeline entry points
|-- tests/
|-- third_party/
|-- sam3d_body_multiview_fusion.py
|-- sam3d_body_360_direct_compare.py
`-- cotracker_person_tracking_yolo.py
```

The current package wrappers call the legacy scripts internally, so existing
commands keep working while newer code can import from `posefusion360`.

## Example Commands

```bash
python cotracker_person_tracking_yolo.py
python sam3d_body_multiview_fusion.py --max-frames 1
python framewise_person_detection.py
```

Package-backed wrappers can be run from the repository root:

```bash
python scripts/run_yolo_tracking.py
python scripts/run_multiview_fusion.py --max-frames 1
python scripts/run_direct_360_compare.py --frame-number 41
python scripts/run_full_pipeline.py
```

For editable package use:

```bash
python -m pip install -e .
posefusion360-tracking
posefusion360-multiview --max-frames 1
posefusion360-direct360 --frame-number 41
posefusion360-full-pipeline
```

Quick structure check:

```bash
python tests/test_project_structure.py
```

Most scripts expect local video, model, checkpoint, or SAM3D Body paths to be configured in each script's `CONFIG` dictionary or passed by CLI flags where available.

## Short Description

> A research project for pseudo-multiview 3D human pose estimation from monocular 360-degree videos, using CoTracker-stabilized person tracking, spherical-to-perspective virtual view generation, SAM3D Body inference, and camera-to-world keypoint fusion.

## Suggested GitHub Description

> Pseudo-multiview 3D human pose estimation from monocular 360 videos using CoTracker, virtual perspective views, and SAM3D Body fusion.

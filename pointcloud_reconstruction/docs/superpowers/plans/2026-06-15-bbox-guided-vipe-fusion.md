# BBox-Guided VIPE Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone wrapper project that uses selfie bbox tracks to cut stable perspective views from a 360 video, run VIPE per view, and prepare rough merged geometry.

**Architecture:** Keep reusable geometry in `src/`, command-line tools in `scripts/`, and docs/config at project root. The wrapper calls the existing VIPE checkout as an external dependency and does not vendor or edit VIPE.

**Tech Stack:** Python stdlib, NumPy, OpenCV for video projection, VIPE CLI, COLMAP text format.

---

### Task 1: BBox and View Geometry

**Files:**
- Create: `pointcloud_reconstruction/src/bbox_views.py`
- Create: `pointcloud_reconstruction/src/projection.py`
- Test: `pointcloud_reconstruction/tests/test_bbox_views.py`
- Test: `pointcloud_reconstruction/tests/test_projection.py`

- [x] Write tests for bbox center, yaw conversion, circular mean, and view offsets.
- [x] Implement bbox JSON helpers and stable anchor yaw inference.
- [x] Implement projection and camera-to-world matrix helpers.
- [x] Run `python3 -m unittest discover -s pointcloud_reconstruction/tests -v`.

### Task 2: View Extraction CLI

**Files:**
- Create: `pointcloud_reconstruction/src/extract_views.py`
- Create: `pointcloud_reconstruction/scripts/extract_dynamic_views.py`

- [x] Implement equirectangular video reading.
- [x] Write one mp4 per fixed virtual view.
- [x] Write `view_manifest.json` with view transforms and output video paths.

### Task 3: VIPE and Merge Wrappers

**Files:**
- Create: `pointcloud_reconstruction/scripts/run_vipe_views.py`
- Create: `pointcloud_reconstruction/src/merge_colmap.py`
- Create: `pointcloud_reconstruction/scripts/merge_vipe_views.py`
- Test: `pointcloud_reconstruction/tests/test_merge_colmap.py`

- [x] Add a dry-runnable VIPE command wrapper.
- [x] Parse COLMAP `points3D.txt`.
- [x] Write rough merged ASCII PLY.
- [x] Save a JSON summary next to the merged PLY.

### Task 4: Documentation and Config

**Files:**
- Create: `pointcloud_reconstruction/README.md`
- Create: `pointcloud_reconstruction/configs/kimura2_360.json`
- Create: `pointcloud_reconstruction/docs/superpowers/specs/2026-06-15-bbox-guided-vipe-fusion-design.md`

- [x] Document the selfie-bbox anchor design.
- [x] Document example commands.
- [x] Call out the rough-merge limitation.

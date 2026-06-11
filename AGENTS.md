# Repository Guidelines

## Project Structure & Module Organization

This repository is a flat Python workflow for 360-degree skiing video analysis.
Core scripts live at the repository root:

- `test_person_detection.py` and `test_360_detection.py`: YOLO-based detection experiments.
- `cotracker_person_tracking.py`: clip-based person tracking and bbox JSON export.
- `framewise_person_detection.py`: frame-by-frame baseline detection and pose extraction.
- `sam3d_body_multiview_fusion.py`: perspective-view generation, SAM3D Body execution, and fused 3D keypoint visualization.
- `vlm_video_analyze.py` and `test_360_vlm_person_detection.py`: VLM-based frame/video analysis.

Local assets include `kimura2_360_half.mp4`, `kimura2_360_half_detected.mp4`, `yolov8n.pt`, and `vlm_analysis_result.txt`. Generated caches such as `__pycache__/` should not be committed.

## Build, Test, and Development Commands

There is no package build step. Run scripts directly from the repository root.

```bash
python test_person_detection.py
python test_360_detection.py
python cotracker_person_tracking.py
python framewise_person_detection.py
python sam3d_body_multiview_fusion.py --max-frames 1
conda run -n vlminference python vlm_video_analyze.py --frames 8
```

Use `python -m py_compile *.py` for a quick syntax check. Most workflows require GPU-capable dependencies such as `torch`, `opencv-python`, `ultralytics`, `numpy`, `matplotlib`, `Pillow`, `decord`, and `transformers`.

## Coding Style & Naming Conventions

Use Python 3 with 4-space indentation. Keep configuration near the top of each script in a `CONFIG` dictionary, as existing files do. Prefer descriptive snake_case names for functions, variables, CLI flags, and output files. Preserve explicit path arguments for reproducible experiments, but avoid hard-coding new machine-specific paths when a CLI argument can be added.

## Testing Guidelines

Tests are currently script-style experiments rather than a formal `pytest` suite. Name new experimental checks `test_<topic>.py` and make them runnable as standalone scripts. For changes to tracking, detection, or fusion logic, verify with a small frame count or `--max-frames 1` before running full videos. Include expected output paths in the script output.

## Commit & Pull Request Guidelines

This checkout does not include local Git history, so no repository-specific commit convention can be inferred. Use concise imperative commit messages, for example `Add SAM3D fusion visualization` or `Tune cubemap detection thresholds`.

Pull requests should describe the affected script, input data used, command run, and generated outputs. Include screenshots or short video samples when visual results change. Note GPU/model requirements and any external checkpoints, Hugging Face models, or third-party repositories needed to reproduce the result.

## Security & Configuration Tips

Do not commit large generated videos, private datasets, downloaded checkpoints, API tokens, or machine-local absolute paths unless they are intentional sample assets. Prefer documenting required paths and keeping environment-specific settings in ignored local files.

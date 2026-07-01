from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ViewSpec:
    name: str
    yaw_deg: float
    pitch_deg: float
    fov_deg: float
    size: int

    def to_json(self) -> dict:
        return asdict(self)


DEFAULT_VIEW_NAMES = {
    0: "selfie",
    60: "front_right",
    120: "back_right",
    180: "back",
    -120: "back_left",
    -60: "front_left",
    90: "right",
    -90: "left",
}


def normalize_yaw(yaw_deg: float) -> float:
    """Normalize yaw to [-180, 180)."""
    return ((float(yaw_deg) + 180.0) % 360.0) - 180.0


def equirect_x_to_yaw(x: float, width: int) -> float:
    """Convert equirectangular x coordinate into the yaw convention used by 360PoseFusion."""
    if width <= 0:
        raise ValueError("width must be positive")
    yaw = (float(x) / float(width) - 0.5) * 360.0
    return normalize_yaw(yaw)


def box_center_x(box_record: dict) -> float:
    """Return the horizontal center of one bbox record, preferring tracked center metadata."""
    if "center_xy" in box_record and box_record["center_xy"]:
        return float(box_record["center_xy"][0])
    x1, _y1, x2, _y2 = box_record["box"]
    return (float(x1) + float(x2)) * 0.5


def select_selfie_box(frame_record: dict, target_id: int = 1) -> dict | None:
    """Select the selfie bbox for one frame."""
    boxes = list(frame_record.get("boxes", []))
    if not boxes:
        return None
    for box in boxes:
        if int(box.get("id", -1)) == int(target_id):
            return box
    return max(boxes, key=lambda item: float(item.get("score", 0.0)))


def circular_mean_degrees(angles: Iterable[float]) -> float:
    """Return a circular mean angle in degrees."""
    values = list(angles)
    if not values:
        raise ValueError("at least one angle is required")
    sin_sum = sum(math.sin(math.radians(value)) for value in values)
    cos_sum = sum(math.cos(math.radians(value)) for value in values)
    if abs(sin_sum) < 1e-12 and abs(cos_sum) < 1e-12:
        return 0.0
    return normalize_yaw(math.degrees(math.atan2(sin_sum, cos_sum)))


def load_selfie_yaws(bbox_json_path: Path | str, target_id: int = 1) -> tuple[list[float], dict]:
    """Load per-frame selfie yaw estimates from a 360PoseFusion bbox JSON file."""
    path = Path(bbox_json_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    width = int(payload["width"])
    yaws: list[float] = []
    for frame in payload.get("frames", []):
        box = select_selfie_box(frame, target_id=target_id)
        if box is None:
            continue
        yaws.append(equirect_x_to_yaw(box_center_x(box), width))
    return yaws, payload


def infer_anchor_yaw(bbox_json_path: Path | str, target_id: int = 1) -> float:
    """Infer one stable sequence-level anchor yaw from selfie bbox tracks."""
    yaws, _payload = load_selfie_yaws(bbox_json_path, target_id=target_id)
    return circular_mean_degrees(yaws)


def build_anchor_views(
    anchor_yaw: float,
    yaw_offsets: Sequence[float] = (0.0, 60.0, 120.0, 180.0, -120.0, -60.0),
    fov_deg: float = 100.0,
    view_size: int = 1024,
    pitch_deg: float = 0.0,
) -> list[ViewSpec]:
    """Build fixed virtual camera views around the selfie-derived anchor yaw."""
    views: list[ViewSpec] = []
    for offset in yaw_offsets:
        rounded_offset = int(round(float(offset)))
        name = DEFAULT_VIEW_NAMES.get(rounded_offset, f"yaw_{rounded_offset:+d}")
        views.append(
            ViewSpec(
                name=name,
                yaw_deg=normalize_yaw(float(anchor_yaw) + float(offset)),
                pitch_deg=float(pitch_deg),
                fov_deg=float(fov_deg),
                size=int(view_size),
            )
        )
    return views


def save_view_manifest(
    output_path: Path | str,
    *,
    source_video: Path | str,
    bbox_json: Path | str,
    anchor_yaw: float,
    views: Sequence[ViewSpec],
) -> None:
    """Write a JSON manifest describing the extracted virtual camera views."""
    payload = {
        "source_video": str(source_video),
        "bbox_json": str(bbox_json),
        "anchor_yaw_deg": float(anchor_yaw),
        "views": [view.to_json() for view in views],
    }
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

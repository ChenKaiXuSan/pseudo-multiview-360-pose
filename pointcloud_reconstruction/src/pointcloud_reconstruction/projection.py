from __future__ import annotations

import numpy as np


def xyz_to_equirectangular(x, y, z, width: int, height: int):
    """Convert world direction vectors to equirectangular pixel coordinates."""
    lon = np.arctan2(x, z)
    lat = np.arctan2(y, np.sqrt(x**2 + z**2))
    x_e = (lon / (2 * np.pi) + 0.5) * width
    y_e = (0.5 - lat / np.pi) * height
    return lon, lat, np.mod(x_e, width), np.clip(y_e, 0, height - 1)


def rotation_axes_yaw_pitch(yaw_deg: float, pitch_deg: float):
    """Return right, up, forward axes for a virtual perspective camera."""
    yaw = np.radians(float(yaw_deg))
    pitch = np.radians(float(pitch_deg))
    forward = np.array(
        [
            np.sin(yaw) * np.cos(pitch),
            np.sin(pitch),
            np.cos(yaw) * np.cos(pitch),
        ],
        dtype=np.float32,
    )
    forward /= max(float(np.linalg.norm(forward)), 1e-6)

    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(world_up, forward)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    right /= max(float(np.linalg.norm(right)), 1e-6)
    up = np.cross(forward, right)
    up /= max(float(np.linalg.norm(up)), 1e-6)
    return right, up, forward


def camera_to_world_matrix(yaw_deg: float, pitch_deg: float):
    """Build a 4x4 camera-to-world rotation matrix for a virtual view."""
    right, up, forward = rotation_axes_yaw_pitch(yaw_deg, pitch_deg)
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, 0] = right
    matrix[:3, 1] = up
    matrix[:3, 2] = forward
    return matrix


def perspective_view_xyz(u, v, yaw_deg: float, pitch_deg: float, fov_deg: float):
    """Map normalized perspective coordinates to world direction vectors."""
    right, up, forward = rotation_axes_yaw_pitch(yaw_deg, pitch_deg)
    half_fov = np.tan(np.radians(float(fov_deg)) * 0.5)
    x_cam = u * half_fov
    y_cam = v * half_fov
    x = forward[0] + x_cam * right[0] + y_cam * up[0]
    y = forward[1] + x_cam * right[1] + y_cam * up[1]
    z = forward[2] + x_cam * right[2] + y_cam * up[2]
    norm = np.sqrt(x**2 + y**2 + z**2)
    return x / norm, y / norm, z / norm


def equirectangular_to_perspective(frame, *, view_size: int, yaw_deg: float, pitch_deg: float, fov_deg: float):
    """Project one equirectangular frame into a square perspective view."""
    import cv2

    height, width = frame.shape[:2]
    coords = np.zeros((view_size, view_size, 2), dtype=np.float32)
    u = (np.arange(view_size) / view_size - 0.5) * 2
    v = -(np.arange(view_size) / view_size - 0.5) * 2
    u, v = np.meshgrid(u, v)
    x, y, z = perspective_view_xyz(u, v, yaw_deg, pitch_deg, fov_deg)
    _lon, _lat, x_e, y_e = xyz_to_equirectangular(x, y, z, width, height)
    coords[..., 0] = x_e
    coords[..., 1] = y_e
    return cv2.remap(frame, coords, None, cv2.INTER_LINEAR)

"""Drawing primitives shared by the per-stage visualizations.

Every function draws in place on a BGR uint8 image and takes coordinates that are
already in pixels, except :func:`gripper`, which needs the camera frame to lay out
the jaw line and the axes at a metric length.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from pipeline import config, geometry

HAND_COLORS = {"left": (80, 200, 255), "right": (120, 255, 120)}  # BGR
RAW_COLOR = (150, 150, 150)
AXIS_COLORS = ((0, 0, 255), (0, 255, 0), (255, 0, 0))  # x approach, y normal, z grasp
AXIS_LENGTH_M = 0.03
FONT = cv2.FONT_HERSHEY_SIMPLEX
TIP_JOINTS = (config.THUMB_TIP, config.INDEX_TIP, config.MIDDLE_TIP)


def _pt(xy) -> tuple[int, int]:
    return int(round(float(xy[0]))), int(round(float(xy[1])))


def skeleton(img: np.ndarray, kp2d: np.ndarray, color, thickness: int = 2,
             radius: int = 3) -> None:
    """Hand skeleton over the 21 keypoints; the tips used by eq. (1)-(2) are larger."""
    for a, b in config.HAND_EDGES:
        cv2.line(img, _pt(kp2d[a]), _pt(kp2d[b]), color, thickness, cv2.LINE_AA)
    for j, xy in enumerate(kp2d):
        cv2.circle(img, _pt(xy), radius + 2 if j in TIP_JOINTS else radius,
                   color, -1, cv2.LINE_AA)


def gripper(img: np.ndarray, position_cam: np.ndarray, quat_cam: np.ndarray,
            width: float, intrins: np.ndarray) -> None:
    """Jaw line and the three axes of a retargeted gripper pose (eq. 2-3)."""
    rot = Rotation.from_quat(quat_cam).as_matrix()
    jaw = 0.5 * width * rot[:, 2]
    ends = geometry.project(np.stack([position_cam - jaw, position_cam + jaw]), intrins)
    cv2.line(img, _pt(ends[0]), _pt(ends[1]), (255, 255, 255), 2, cv2.LINE_AA)
    for end in ends:
        cv2.circle(img, _pt(end), 5, (255, 255, 255), -1, cv2.LINE_AA)
    for axis, color in enumerate(AXIS_COLORS):
        seg = geometry.project(
            np.stack([position_cam, position_cam + AXIS_LENGTH_M * rot[:, axis]]), intrins)
        cv2.arrowedLine(img, _pt(seg[0]), _pt(seg[1]), color, 2, cv2.LINE_AA, tipLength=0.3)


def hud(img: np.ndarray, lines: list[str], scale: float = 0.7) -> None:
    """Top-left status text."""
    for k, text in enumerate(lines):
        cv2.putText(img, text, (10, 26 + int(26 * scale / 0.7) * k), FONT, scale,
                    (255, 255, 255), 2, cv2.LINE_AA)

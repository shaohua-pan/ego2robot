"""Hand-to-gripper retargeting (paper section 3.1, equations 1-3, appendix A.3).

The 21 MANO keypoints are reduced to a parallel-jaw gripper pose:

    p_vf  = 0.7 * p_index_tip + 0.3 * p_middle_tip           (1)
    p_tcp = (p_thumb_tip + p_vf) / 2,   w = |p_thumb_tip - p_vf|   (2)
    z = s (p_thumb_tip - p_vf) / w,  y = z x d / |z x d|,  x = y x z   (3)

with d = p_vf - p_wrist and s = +1 for the right hand, -1 for the left, so both
hands land in the same gripper frame. R = [x y z] as columns.

Frames where the grasp frame is undefined (paper A.3 "Degenerate Orientation":
w < 1 cm, or z nearly parallel to d) keep the previous valid orientation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from pipeline import config


@dataclass
class GripperTraj:
    """Per-frame gripper trajectory; invalid frames hold arbitrary values."""
    position: np.ndarray      # (T, 3) TCP in the input frame, metres
    quat: np.ndarray          # (T, 4) xyzw
    width: np.ndarray         # (T,) jaw opening, metres
    valid: np.ndarray         # (T,) bool
    frozen: np.ndarray        # (T,) bool, orientation held from a previous frame


def virtual_fingertip(kp3d: np.ndarray) -> np.ndarray:
    """Equation (1) on (..., 21, 3) keypoints."""
    return (config.VF_INDEX_WEIGHT * kp3d[..., config.INDEX_TIP, :]
            + config.VF_MIDDLE_WEIGHT * kp3d[..., config.MIDDLE_TIP, :])


def grasp_frame(jaw: np.ndarray, width: float, d: np.ndarray,
                sign: float) -> np.ndarray | None:
    """Equation (3) for one frame, or None where the frame is degenerate.

    :param jaw p_thumb - p_vf
    :param width |jaw|
    :param d p_vf - p_wrist
    :param sign +1 for the right hand, -1 for the left
    :returns R = [x y z] as columns
    """
    if width < config.GRIPPER_WIDTH_MIN_M:
        return None
    z = sign * jaw / width
    normal = np.cross(z, d)
    norm = np.linalg.norm(normal)
    if norm <= config.DEGENERATE_CROSS_EPS:
        return None
    y = normal / norm
    return np.stack([np.cross(y, z), y, z], axis=1)


def retarget(kp3d: np.ndarray, valid: np.ndarray, is_right: bool) -> GripperTraj:
    """Equations (1)-(3) for one hand track.

    :param kp3d (T, 21, 3) keypoints in a single frame of reference, metres
    :param valid (T,) bool
    :param is_right sets the sign s of the grasp axis
    """
    p_thumb = kp3d[:, config.THUMB_TIP, :]
    p_wrist = kp3d[:, config.WRIST, :]
    p_vf = virtual_fingertip(kp3d)

    jaw = p_thumb - p_vf                       # thumb tip -> virtual fingertip
    d = p_vf - p_wrist                         # wrist -> virtual fingertip
    width = np.linalg.norm(jaw, axis=-1)
    position = 0.5 * (p_thumb + p_vf)
    sign = 1.0 if is_right else -1.0

    quat = np.zeros((len(kp3d), 4))
    frozen = np.zeros(len(kp3d), dtype=bool)
    out_valid = np.array(valid, dtype=bool, copy=True)

    last = None
    for t in np.flatnonzero(out_valid):
        rot = grasp_frame(jaw[t], float(width[t]), d[t], sign)
        if rot is not None:
            quat[t] = Rotation.from_matrix(rot).as_quat()
            last = quat[t]
        elif last is None:
            out_valid[t] = False               # degenerate with nothing to hold yet
        else:
            quat[t] = last                     # paper A.3: hold the last valid estimate
            frozen[t] = True

    return GripperTraj(position=position, quat=quat, width=width,
                       valid=out_valid, frozen=frozen)

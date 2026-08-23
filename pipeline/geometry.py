"""Camera geometry shared across stages.

Dyn-HaMR (and therefore this pipeline) stores ``cam_R``/``cam_t`` as the
world-to-camera transform and intrinsics as ``[fx, fy, cx, cy]``, so all
conversions live here rather than being re-derived per stage.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def project(points_cam: np.ndarray, intrins: np.ndarray) -> np.ndarray:
    """Pinhole projection of camera-frame points; ``(..., 3) -> (..., 2)``."""
    fx, fy, cx, cy = intrins
    z = np.clip(points_cam[..., 2:3], 1e-6, None)
    return points_cam[..., :2] / z * np.array([fx, fy]) + np.array([cx, cy])


def points_to_camera(points_world: np.ndarray, cam_R: np.ndarray,
                     cam_t: np.ndarray) -> np.ndarray:
    """World -> camera for keypoint arrays ``(B, T, J, 3)`` with ``(B, T, ...)`` cameras."""
    return np.einsum("btij,btkj->btki", cam_R, points_world) + cam_t[:, :, None, :]


def pose_to_camera(position: np.ndarray, quat: np.ndarray, cam_R: np.ndarray,
                   cam_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World -> camera for a pose trajectory ``(T, 3)`` + ``(T, 4)`` xyzw."""
    position_cam = np.einsum("tij,tj->ti", cam_R, position) + cam_t
    rot_cam = np.einsum("tij,tjk->tik", cam_R, Rotation.from_quat(quat).as_matrix())
    return position_cam, Rotation.from_matrix(rot_cam).as_quat()

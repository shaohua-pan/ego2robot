"""Velocity filtering and temporal smoothing (paper appendix A.3).

Velocity filter, iterated ``VEL_FILTER_ROUNDS`` times:

    tau = max(5 * median(v), floor),  floor = 0.9/fps m/frame (position)
                                            10.0/fps rad/frame (rotation)

Frames whose velocity exceeds tau are replaced by interpolation from their
neighbours. Smoothing then applies a Savitzky-Golay filter to positions and
widths (window min(21, n), order min(3, window-1)) and a Gaussian-weighted
quaternion average to orientations (sigma = 10 frames, kernel 21), with adjacent
quaternions hemisphere-corrected first.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp

from pipeline import config


def hemisphere_correct(quat: np.ndarray) -> np.ndarray:
    """Flip q -> -q where it takes the short way round from its predecessor."""
    out = np.array(quat, dtype=np.float64, copy=True)
    for t in range(1, len(out)):
        if np.dot(out[t], out[t - 1]) < 0:
            out[t] = -out[t]
    return out


def _interp_positions(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Linear interpolation of the invalid entries from the valid ones."""
    out = np.array(values, dtype=np.float64, copy=True)
    idx, target = np.flatnonzero(valid), np.flatnonzero(~valid)
    if idx.size == 0 or target.size == 0:
        return out
    flat = out.reshape(len(out), -1)          # a view, so writes reach `out`
    for c in range(flat.shape[1]):
        flat[target, c] = np.interp(target, idx, flat[idx, c])
    return out


def _interp_quats(quat: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """SLERP the invalid rows from the surrounding valid ones; ends are held."""
    out = np.array(quat, dtype=np.float64, copy=True)
    idx = np.flatnonzero(valid)
    if idx.size < 2:
        return out
    slerp = Slerp(idx, Rotation.from_quat(out[idx]))
    target = np.flatnonzero(~valid)
    inside = target[(target > idx[0]) & (target < idx[-1])]
    if inside.size:
        out[inside] = slerp(inside).as_quat()
    out[target[target < idx[0]]] = out[idx[0]]
    out[target[target > idx[-1]]] = out[idx[-1]]
    return out


def velocity_filter(position: np.ndarray, quat: np.ndarray, valid: np.ndarray,
                    fps: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Paper eq. (7); returns filtered copies plus the per-round removal counts."""
    position = np.array(position, dtype=np.float64, copy=True)
    quat = hemisphere_correct(quat)
    stats = {"position": [], "rotation": []}

    for _ in range(config.VEL_FILTER_ROUNDS):
        for kind in ("position", "rotation"):
            keep = np.array(valid, dtype=bool, copy=True)
            idx = np.flatnonzero(keep)
            if idx.size < 3:
                stats[kind].append(0)
                continue
            if kind == "position":
                vel = np.linalg.norm(np.diff(position[idx], axis=0), axis=-1)
                floor = config.VEL_POS_FLOOR_M_PER_S / fps
            else:
                rel = Rotation.from_quat(quat[idx][1:]) * Rotation.from_quat(quat[idx][:-1]).inv()
                vel = np.linalg.norm(rel.as_rotvec(), axis=-1)
                floor = config.VEL_ROT_FLOOR_RAD_PER_S / fps
            tau = max(config.VEL_MEDIAN_FACTOR * float(np.median(vel)), floor)
            # vel[k] is the step into idx[k + 1], so an outlier marks that frame.
            bad = idx[1:][vel > tau]
            stats[kind].append(int(bad.size))
            if bad.size == 0:
                continue
            keep[bad] = False
            if kind == "position":
                position = _interp_positions(position, keep)
            else:
                quat = _interp_quats(quat, keep)
    return position, quat, stats


def savgol_window(n: int) -> tuple[int, int]:
    """Window min(21, n) forced odd (``savgol_filter`` requires it) and its order."""
    window = min(config.SAVGOL_MAX_WINDOW, n)
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return 0, 0
    return window, min(config.SAVGOL_MAX_ORDER, window - 1)


def smooth_series(values: np.ndarray) -> np.ndarray:
    """Savitzky-Golay along time; ``values`` is (T,) or (T, D)."""
    window, order = savgol_window(len(values))
    if window == 0:
        return np.array(values, dtype=np.float64, copy=True)
    return savgol_filter(np.asarray(values, dtype=np.float64), window, order, axis=0)


def gaussian_slerp(quat: np.ndarray) -> np.ndarray:
    """Gaussian-weighted quaternion average over a kernel of SLERP_KERNEL frames.

    The paper calls this "Gaussian-weighted SLERP" without giving a formula. A
    weighted average of more than two rotations has no closed-form SLERP, so this
    uses the standard weighted quaternion mean: the dominant eigenvector of
    sum_i w_i q_i q_i^T, which reduces to SLERP for two samples and does not
    depend on the order the neighbours are visited.
    """
    quat = hemisphere_correct(quat)
    n = len(quat)
    half = config.SLERP_KERNEL // 2
    offsets = np.arange(-half, half + 1)
    weights = np.exp(-0.5 * (offsets / config.SLERP_SIGMA_FRAMES) ** 2)

    out = np.empty_like(quat)
    for t in range(n):
        lo, hi = max(0, t - half), min(n, t + half + 1)
        w = weights[(lo - t) + half: (hi - t) + half]
        q = quat[lo:hi]
        # Align the window to the centre frame before averaging.
        q = q * np.where(q @ quat[t] < 0, -1.0, 1.0)[:, None]
        m = (q * w[:, None]).T @ q
        vals, vecs = np.linalg.eigh(m)
        out[t] = vecs[:, int(np.argmax(vals))]
    norms = np.linalg.norm(out, axis=-1, keepdims=True)
    return out / np.where(norms > 0, norms, 1.0)

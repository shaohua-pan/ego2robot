"""Keyframe selection for the base pose search (paper eq. 4).

The paper defines the keyframe set only by intent: "representative keyframes
selected to cover the spatial extremes of the trajectory (positions with maximum
displacement or orientation change)", capped at 20 in A.4. Farthest-point sampling
in a combined position/orientation metric is the standard way to realise that: it
starts from the two endpoints and repeatedly adds whichever frame is farthest from
everything picked so far, so extremes are taken first and the set stays spread out.
"""
from __future__ import annotations

import numpy as np

from pipeline import config

# 1 rad of wrist rotation is treated as 0.1 m of travel, so a full flip counts about
# as much as a 30 cm reach - enough for orientation extremes to be picked, not enough
# to crowd out position extremes.
ROT_WEIGHT_M_PER_RAD = 0.1


def _geodesic(quats: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """Angle between each quaternion and ``quat``, in radians."""
    return 2.0 * np.arccos(np.clip(np.abs(quats @ quat), 0.0, 1.0))


def select(positions: np.ndarray, quats: np.ndarray, valid: np.ndarray,
           limit: int = config.IK_MAX_KEYFRAMES) -> np.ndarray:
    """Frame indices of at most ``limit`` representative keyframes."""
    frames = np.flatnonzero(valid)
    if len(frames) <= limit:
        return frames

    pos, quat = positions[frames], quats[frames]
    # Seed with the two frames farthest apart in position: those are the extremes of
    # the reach envelope, which is what the base placement has to cover.
    gaps = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    first, second = np.unravel_index(np.argmax(gaps), gaps.shape)
    picked = [int(first), int(second)]

    def distance(idx: int) -> np.ndarray:
        return (np.linalg.norm(pos - pos[idx], axis=-1)
                + ROT_WEIGHT_M_PER_RAD * _geodesic(quat, quat[idx]))

    spread = np.minimum(distance(picked[0]), distance(picked[1]))
    while len(picked) < limit:
        spread[picked] = -1.0
        nxt = int(np.argmax(spread))
        picked.append(nxt)
        spread = np.minimum(spread, distance(nxt))
    return np.sort(frames[np.array(picked)])

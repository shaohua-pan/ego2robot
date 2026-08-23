"""Gap handling for hand-pose tracks (paper appendix A.1, last paragraph).

When hand detections are missing for consecutive frames the paper applies
gap-specific interpolation:

- small gaps (<= 10 frames): linear position interpolation + SLERP orientation;
- large gaps (> 10 frames): fill with the robot's home configuration and blend
  smoothly at the gap boundaries. The blend length is

      n = max(5, min(90, ceil(0.6 n_pos + 0.4 n_rot)))
      n_pos = dp / 3.25 mm       n_rot = dtheta / 1.08 deg

  i.e. the number of frames needed to cover the displacement at a fixed blend
  speed.

The home configuration is an embodiment-specific quantity, so it is passed in by
the caller (step 5 supplies it once the robot model is loaded). Without it, large
gaps are left unfilled and stay invalid; a caller can then drop those frames.

Poses are (position, quaternion) pairs with quaternions in scipy's xyzw order.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from pipeline import config


@dataclass
class GapStats:
    small_filled: int = 0          # frames filled by interpolation
    large_filled: int = 0          # frames filled with the home configuration
    unfilled: int = 0              # frames left invalid
    gaps: list[dict] = field(default_factory=list)


def blend_length(delta_pos_m: float, delta_rot_rad: float) -> int:
    """Paper A.1 blend length in frames for a given boundary-to-home offset."""
    n_pos = delta_pos_m / config.BLEND_POS_STEP_M
    n_rot = math.degrees(delta_rot_rad) / config.BLEND_ROT_STEP_DEG
    n = math.ceil(config.BLEND_POS_WEIGHT * n_pos + config.BLEND_ROT_WEIGHT * n_rot)
    return max(config.BLEND_MIN_FRAMES, min(config.BLEND_MAX_FRAMES, n))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [start, end) index ranges of consecutive True entries."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate([[idx[0]], idx[breaks + 1]])
    ends = np.concatenate([idx[breaks], [idx[-1]]]) + 1
    return list(zip(starts.tolist(), ends.tolist()))


def _rot_angle(q_a: np.ndarray, q_b: np.ndarray) -> float:
    """Geodesic angle between two quaternions, in radians."""
    rel = Rotation.from_quat(q_b) * Rotation.from_quat(q_a).inv()
    return float(np.linalg.norm(rel.as_rotvec()))


def _slerp_pair(q_a: np.ndarray, q_b: np.ndarray, ratios: np.ndarray) -> np.ndarray:
    slerp = Slerp([0.0, 1.0], Rotation.from_quat(np.stack([q_a, q_b])))
    return slerp(ratios).as_quat()


def _fill_small_gap(positions, quats, widths, start, end) -> None:
    """Linear position interpolation + SLERP between the two valid boundaries."""
    left, right = start - 1, end
    ratios = (np.arange(start, end) - left) / float(right - left)
    positions[start:end] = (positions[left][None, :]
                            + ratios[:, None] * (positions[right] - positions[left])[None, :])
    quats[start:end] = _slerp_pair(quats[left], quats[right], ratios)
    if widths is not None:
        widths[start:end] = widths[left] + ratios * (widths[right] - widths[left])


def _blend_to(positions, quats, widths, span, src_idx, dst_pos, dst_quat, dst_width,
              reverse: bool) -> None:
    """Blend the frames in ``span`` from the pose at ``src_idx`` towards ``dst_*``.

    ``reverse`` blends the other way round (home -> first valid pose after a gap),
    which is the mirror image of the entry transition.
    """
    n = len(span)
    if n == 0:
        return
    steps = np.arange(1, n + 1) / float(n + 1)
    if reverse:
        steps = steps[::-1]
    src_pos, src_quat = positions[src_idx], quats[src_idx]
    positions[span] = src_pos[None, :] + steps[:, None] * (dst_pos - src_pos)[None, :]
    quats[span] = _slerp_pair(src_quat, dst_quat, steps)
    if widths is not None and dst_width is not None:
        widths[span] = widths[src_idx] + steps * (dst_width - widths[src_idx])


def _fill_large_gap(positions, quats, widths, start, end,
                    home_position, home_quat, home_width) -> None:
    """Hold the home configuration through the gap, blending at both boundaries."""
    left = start - 1 if start > 0 else None
    right = end if end < len(positions) else None

    n_in = 0 if left is None else blend_length(
        float(np.linalg.norm(home_position - positions[left])),
        _rot_angle(quats[left], home_quat))
    n_out = 0 if right is None else blend_length(
        float(np.linalg.norm(positions[right] - home_position)),
        _rot_angle(home_quat, quats[right]))

    # The blends have to fit inside the gap; shrink them proportionally if not.
    gap_len = end - start
    if n_in + n_out > gap_len:
        scale = gap_len / float(n_in + n_out)
        n_in, n_out = int(n_in * scale), int(n_out * scale)

    positions[start:end] = home_position[None, :]
    quats[start:end] = home_quat[None, :]
    if widths is not None and home_width is not None:
        widths[start:end] = home_width

    if left is not None and n_in > 0:
        span = np.arange(start, start + n_in)
        _blend_to(positions, quats, widths, span, left,
                  home_position, home_quat, home_width, reverse=False)
    if right is not None and n_out > 0:
        span = np.arange(end - n_out, end)
        _blend_to(positions, quats, widths, span, right,
                  home_position, home_quat, home_width, reverse=True)


def fill_gaps(positions: np.ndarray,
              quats: np.ndarray,
              valid: np.ndarray,
              widths: np.ndarray | None = None,
              home_position: np.ndarray | None = None,
              home_quat: np.ndarray | None = None,
              home_width: float | None = None):
    """Fill missing-detection gaps in a pose trajectory (paper A.1).

    :param positions (T, 3) metres
    :param quats (T, 4) xyzw
    :param valid (T,) bool, True where a hand was detected
    :param widths (T,) optional gripper width, interpolated alongside the pose
    :param home_* the robot home configuration used for gaps > 10 frames
    :returns (positions, quats, widths, valid, stats) with filled frames marked
        valid; frames that could not be filled stay invalid.
    """
    positions = np.array(positions, dtype=np.float64, copy=True)
    quats = np.array(quats, dtype=np.float64, copy=True)
    valid = np.array(valid, dtype=bool, copy=True)
    widths = None if widths is None else np.array(widths, dtype=np.float64, copy=True)
    stats = GapStats()

    has_home = home_position is not None and home_quat is not None
    if has_home:
        home_position = np.asarray(home_position, dtype=np.float64)
        home_quat = np.asarray(home_quat, dtype=np.float64)

    for start, end in _runs(~valid):
        gap_len = end - start
        interior = start > 0 and end < len(valid)
        if gap_len <= config.GAP_LARGE_FRAMES and interior:
            _fill_small_gap(positions, quats, widths, start, end)
            valid[start:end] = True
            stats.small_filled += gap_len
            kind = "small"
        elif has_home:
            _fill_large_gap(positions, quats, widths, start, end,
                            home_position, home_quat, home_width)
            valid[start:end] = True
            stats.large_filled += gap_len
            kind = "large"
        else:
            stats.unfilled += gap_len
            kind = "unfilled"
        stats.gaps.append({"start": start, "end": end, "length": gap_len, "kind": kind})

    return positions, quats, widths, valid, stats


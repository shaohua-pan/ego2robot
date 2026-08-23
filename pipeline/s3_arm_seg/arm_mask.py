"""Arm-mask post-processing (paper appendix A.4).

SAM 3 gives one mask per frame; the paper then applies three fixes, in this order:

    (i)   gaps of at most 3 frames are filled by interpolating neighbouring masks;
    (ii)  frames whose mask area falls below 50% of the local median (window 11)
          are replaced by the nearest valid mask;
    (iii) a morphological close with a 5x5 elliptical kernel.

"Interpolating neighbouring masks" is not defined further. Blending two binary
masks and thresholding just switches from one to the other half-way through, so
this interpolates their signed distance transforms instead, which is the usual way
to morph between binary shapes and degenerates to the neighbours at the ends.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from pipeline import config


@dataclass
class MaskStats:
    interpolated: list[int] = field(default_factory=list)   # frames filled by (i)
    replaced: list[int] = field(default_factory=list)       # frames replaced by (ii)
    missing: list[int] = field(default_factory=list)        # still without a mask
    area_ratio: list[float] = field(default_factory=list)   # mask area / frame area


def _signed_distance(mask: np.ndarray) -> np.ndarray:
    """Positive inside the mask, negative outside, in pixels."""
    inside = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    outside = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
    return inside - outside


def interpolate_masks(left: np.ndarray, right: np.ndarray, alpha: float) -> np.ndarray:
    """Shape-interpolate two binary masks; ``alpha`` 0 gives ``left``, 1 ``right``.

    Distance-transform interpolation collapses to nothing when the two masks do not
    overlap at all, which for a mask that step 4 inpaints would silently leave the
    arm in the frame, so that case falls back to the union.
    """
    blended = (1.0 - alpha) * _signed_distance(left) + alpha * _signed_distance(right)
    out = blended >= 0.0
    return out if out.any() else (left | right)


def fill_short_gaps(masks: list[np.ndarray | None], stats: MaskStats) -> None:
    """Paper A.4 (i): gaps of <= 3 frames between two masks are interpolated."""
    present = [i for i, m in enumerate(masks) if m is not None]
    for left, right in zip(present, present[1:]):
        gap = right - left - 1
        if not 0 < gap <= config.MASK_GAP_MAX_FRAMES:
            continue
        for i in range(left + 1, right):
            masks[i] = interpolate_masks(masks[left], masks[right],
                                         (i - left) / float(right - left))
            stats.interpolated.append(i)


def replace_small_masks(masks: list[np.ndarray | None], stats: MaskStats) -> None:
    """Paper A.4 (ii): drop masks below half the local median area, then backfill.

    The median runs over a centred window of 11 frames, counting only frames that
    have a mask, and each dropped frame takes the nearest surviving mask.
    """
    areas = np.array([0.0 if m is None else float(m.sum()) for m in masks])
    have = np.array([m is not None for m in masks])
    half = config.MASK_AREA_MEDIAN_WINDOW // 2

    too_small = []
    for i in np.flatnonzero(have):
        lo, hi = max(0, i - half), min(len(masks), i + half + 1)
        window = areas[lo:hi][have[lo:hi]]
        if window.size and areas[i] < config.MASK_AREA_MIN_RATIO * float(np.median(window)):
            too_small.append(int(i))

    keep = np.flatnonzero(have & ~np.isin(np.arange(len(masks)), too_small))
    for i in too_small:
        if keep.size == 0:
            masks[i] = None
            continue
        masks[i] = masks[int(keep[np.argmin(np.abs(keep - i))])].copy()
        stats.replaced.append(i)


def close_mask(mask: np.ndarray) -> np.ndarray:
    """Paper A.4 (iii): morphological close with a 5x5 elliptical kernel."""
    size = config.MASK_CLOSE_KERNEL
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return closed.astype(bool)


def postprocess(masks: list[np.ndarray | None]) -> tuple[list[np.ndarray | None], MaskStats]:
    """Run A.4 (i)-(iii) in order; frames without a mask stay ``None``."""
    masks = list(masks)
    stats = MaskStats()
    fill_short_gaps(masks, stats)
    replace_small_masks(masks, stats)
    frame_area = None
    for i, mask in enumerate(masks):
        if mask is None:
            stats.missing.append(i)
            stats.area_ratio.append(0.0)
            continue
        masks[i] = close_mask(mask)
        frame_area = frame_area or float(mask.size)
        stats.area_ratio.append(round(float(masks[i].sum()) / frame_area, 5))
    return masks, stats

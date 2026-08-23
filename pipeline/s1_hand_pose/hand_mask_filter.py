"""Filter WiLoR detections against a SAM 3 hand mask (paper appendix A.1).

"Detections are filtered by a SAM 3-generated hand mask: frames where more than
80% of projected keypoints fall outside the mask are discarded."

This runs on the raw per-frame detections, before cross-frame association, which
is the order the paper describes. The mask is the union of every instance SAM 3
matches to the "hand" prompt, so one mask sequence serves both hands. Masks are
consumed as SAM 3 propagates, so peak memory does not grow with clip length.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pipeline import config
from pipeline.s1_hand_pose.wilor_runner import FrameResult


def outside_ratio(kp2d: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of the projected keypoints that fall outside ``mask``.

    Keypoints projected outside the image count as outside the mask.
    """
    h, w = mask.shape
    xy = np.round(kp2d).astype(int)
    inside_img = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    hit = np.zeros(len(xy), dtype=bool)
    if inside_img.any():
        px = xy[inside_img]
        hit[inside_img] = mask[px[:, 1], px[:, 0]]
    return float(1.0 - hit.mean())


def filter_detections(frames: Sequence[FrameResult], frames_rgb: Sequence[np.ndarray],
                      segmenter, prompt: str | None = None) -> dict:
    """Drop detections that miss the SAM 3 hand mask; modifies ``frames`` in place.

    :param frames per-frame WiLoR results
    :param frames_rgb the same frames as RGB uint8 images
    :param segmenter a :class:`pipeline.sam3_runner.Sam3VideoSegmenter`
    :returns counts of kept/dropped detections
    """
    prompt = prompt or config.SAM3_HAND_PROMPT
    n_before = sum(len(fr.detections) for fr in frames)
    frames_without_mask = 0

    for idx, mask in segmenter.segment_iter(frames_rgb, prompt):
        if idx >= len(frames):
            break
        frame = frames[idx]
        if mask is None:
            frames_without_mask += 1
            frame.detections = []
            continue
        frame.detections = [
            det for det in frame.detections
            if outside_ratio(det.kp2d, mask) <= config.HAND_MASK_OUTSIDE_RATIO
        ]

    n_after = sum(len(fr.detections) for fr in frames)
    return {"detections_before": n_before, "detections_after": n_after,
            "dropped": n_before - n_after, "frames_without_mask": frames_without_mask}

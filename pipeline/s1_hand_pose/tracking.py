"""Cross-frame hand association and jump filtering (paper appendix A.1).

Cross-frame association:
- Pick the frame with the highest combined left+right detection score as the seed
  and initialise the two tracks from the detector's handedness labels.
- Propagate bidirectionally from the seed: each new detection is assigned to the
  hand whose previous wrist position is nearest in image space (L2).
- When several detections compete for the same hand, keep the highest-scoring one.

Jump filter:
- After association, drop frames whose 3D wrist velocity exceeds
  max(4 x median velocity, 0.003 m/frame), which indicates a misdetection.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pipeline import config
from pipeline.s1_hand_pose.wilor_runner import FrameResult, HandDetection

LEFT, RIGHT = 0, 1
HAND_NAMES = {LEFT: "left", RIGHT: "right"}


@dataclass
class HandTrack:
    """Full-video trajectory for one hand; invalid frames stay zero-filled."""
    hand: int
    valid: np.ndarray          # (N,) bool
    score: np.ndarray          # (N,)
    bbox: np.ndarray           # (N, 4)
    kp3d_cam: np.ndarray       # (N, 21, 3)
    kp2d: np.ndarray           # (N, 21, 2)
    cam_t: np.ndarray          # (N, 3)
    global_orient: np.ndarray  # (N, 3, 3)
    hand_pose: np.ndarray      # (N, 15, 3, 3)
    betas: np.ndarray          # (N, 10)

    @classmethod
    def empty(cls, hand: int, n: int) -> "HandTrack":
        return cls(
            hand=hand,
            valid=np.zeros(n, dtype=bool),
            score=np.zeros(n, dtype=np.float32),
            bbox=np.zeros((n, 4), dtype=np.float32),
            kp3d_cam=np.zeros((n, 21, 3), dtype=np.float32),
            kp2d=np.zeros((n, 21, 2), dtype=np.float32),
            cam_t=np.zeros((n, 3), dtype=np.float32),
            global_orient=np.zeros((n, 3, 3), dtype=np.float32),
            hand_pose=np.zeros((n, 15, 3, 3), dtype=np.float32),
            betas=np.zeros((n, 10), dtype=np.float32),
        )

    def set(self, idx: int, det: HandDetection) -> None:
        self.valid[idx] = True
        self.score[idx] = det.score
        self.bbox[idx] = det.bbox
        self.kp3d_cam[idx] = det.kp3d_cam
        self.kp2d[idx] = det.kp2d
        self.cam_t[idx] = det.cam_t
        self.global_orient[idx] = det.global_orient.reshape(3, 3)
        self.hand_pose[idx] = det.hand_pose
        self.betas[idx] = det.betas

    @property
    def wrist3d(self) -> np.ndarray:
        return self.kp3d_cam[:, config.WRIST, :]

    @property
    def wrist2d(self) -> np.ndarray:
        return self.kp2d[:, config.WRIST, :]
def _seed_frame(frames: list[FrameResult]) -> int:
    """Return the frame index with the highest combined left+right score."""
    best_idx, best_score = -1, -1.0
    for i, fr in enumerate(frames):
        if not fr.detections:
            continue
        left = max((d.score for d in fr.detections if d.is_right == LEFT), default=0.0)
        right = max((d.score for d in fr.detections if d.is_right == RIGHT), default=0.0)
        total = left + right
        if total > best_score:
            best_idx, best_score = i, total
    return best_idx


def _assign_by_nearest(
    dets: list[HandDetection],
    last_wrist2d: dict[int, np.ndarray | None],
) -> dict[int, HandDetection]:
    """Assign this frame's detections to the left/right tracks by nearest wrist.

    Hands without a tracking history fall back to the detector's handedness label.
    If several detections claim the same hand, the highest-scoring one wins.
    """
    assigned: dict[int, HandDetection] = {}
    for det in dets:
        wrist = det.kp2d[config.WRIST]
        dists = {hand: float(np.linalg.norm(wrist - prev))
                 for hand, prev in last_wrist2d.items() if prev is not None}
        hand = min(dists, key=dists.get) if dists else int(det.is_right)
        prev_det = assigned.get(hand)
        if prev_det is None or det.score > prev_det.score:
            assigned[hand] = det
    return assigned


def associate(frames: list[FrameResult]) -> dict[int, HandTrack]:
    """Build temporally consistent left/right tracks by seeded bidirectional propagation."""
    n = len(frames)
    tracks = {LEFT: HandTrack.empty(LEFT, n), RIGHT: HandTrack.empty(RIGHT, n)}
    seed = _seed_frame(frames)
    if seed < 0:
        return tracks

    # Seed frame: trust the detector's handedness, keep the best score per hand.
    for hand in (LEFT, RIGHT):
        cands = [d for d in frames[seed].detections if d.is_right == hand]
        if cands:
            tracks[hand].set(seed, max(cands, key=lambda d: d.score))

    for direction in (1, -1):
        last = {hand: (tracks[hand].wrist2d[seed].copy()
                       if tracks[hand].valid[seed] else None)
                for hand in (LEFT, RIGHT)}
        idx = seed + direction
        while 0 <= idx < n:
            dets = frames[idx].detections
            if dets:
                for hand, det in _assign_by_nearest(dets, last).items():
                    tracks[hand].set(idx, det)
                    last[hand] = det.kp2d[config.WRIST].copy()
            idx += direction
    return tracks


def jump_filter(track: HandTrack) -> int:
    """Invalidate frames whose 3D wrist velocity is implausible; return the count.

    Paper A.1: drop frames where the 3D wrist velocity exceeds
    max(4 x median velocity, 0.003 m/frame). Velocities are only defined between
    consecutive frames that both carry a detection, so pairs separated by a
    dropout are skipped rather than rescaled.
    """
    idxs = np.flatnonzero(track.valid)
    if idxs.size < 3:
        return 0
    pairs = idxs[1:][np.diff(idxs) == 1]
    if pairs.size == 0:
        return 0
    wrist = track.wrist3d
    vel = np.linalg.norm(wrist[pairs] - wrist[pairs - 1], axis=1)
    thresh = max(config.JUMP_VEL_MEDIAN_FACTOR * float(np.median(vel)),
                 config.JUMP_VEL_FLOOR_M_PER_FRAME)
    bad = pairs[vel > thresh]
    track.valid[bad] = False
    return int(bad.size)

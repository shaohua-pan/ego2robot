"""Base pose candidate generation, pruning and scoring (paper eq. 4, 8 and A.4).

Everything here lives in the **world** frame, which Dyn-HaMR anchors to the first
frame's camera: its ``+x`` is right, ``+y`` down and ``+z`` forward as seen from frame
0, so "vertical" offsets run along ``-y``. The robot is bolted to the scene, not to the
wearer's head, so eq.(4) has to be solved once in this frame rather than in the moving
camera frame - a base held still in camera coordinates slides through the scene by as
much as the head moves (95 mm and 77 degrees over the sample clip's third take).

The paper does not say how its pitch/yaw/roll triples compose, or what they are
measured from. The nominal orientation used here puts the robot's own ``+x``
(forward) along the wearer's initial viewing direction and its ``+z`` (up) along the
initial camera up, i.e. a base standing behind the workspace looking the same way as
the wearer; yaw then turns it about its up axis, pitch tips its nose up about its right
axis and roll spins it about its own forward axis.

The pitch sign is not arbitrary: every candidate that survives A.4's reach pruning
on the sample clip sits *below* the trajectory, so a nose-down tilt aims the arm
away from the targets and IK feasibility collapses to zero, while nose-up gives
0.7-1.0. All of A.4's pitches are positive, so nose-up is what they must mean.
"""
from __future__ import annotations

import numpy as np

from pipeline import config

LATERAL = np.array([1.0, 0.0, 0.0])
UP = np.array([0.0, -1.0, 0.0])
FORWARD = np.array([0.0, 0.0, 1.0])
# World-frame columns of the nominal base orientation: robot x = initial view direction,
# robot y = robot up x robot forward (its left), robot z = initial camera up.
NOMINAL = np.stack([FORWARD, np.cross(UP, FORWARD), UP], axis=1)
# Axes of the base's own frame, in that frame: forward, right (= -left), up.
BASE_ROLL_AXIS = np.array([1.0, 0.0, 0.0])
BASE_PITCH_AXIS = np.array([0.0, -1.0, 0.0])
BASE_YAW_AXIS = np.array([0.0, 0.0, 1.0])


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation about a unit axis."""
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def orientations() -> np.ndarray:
    """The 3 x 5 x 3 pitch/yaw/roll grid, as world-frame rotation matrices."""
    out = []
    for pitch in np.radians(config.BASE_PITCH_DEG):
        for yaw in np.radians(config.BASE_YAW_DEG):
            for roll in np.radians(config.BASE_ROLL_DEG):
                out.append(NOMINAL
                           @ _axis_rotation(BASE_YAW_AXIS, yaw)
                           @ _axis_rotation(BASE_PITCH_AXIS, pitch)
                           @ _axis_rotation(BASE_ROLL_AXIS, roll))
    return np.stack(out)



def positions(centroid: np.ndarray, reach: float, sign: float) -> np.ndarray:
    """Candidate base origins around the trajectory centroid, scaled by reach."""
    lat, fwd, vert = np.meshgrid(np.asarray(config.BASE_LATERAL_FACTORS) * sign,
                                 config.BASE_FORWARD_FACTORS,
                                 config.BASE_VERTICAL_FACTORS, indexing="ij")
    offsets = (lat.reshape(-1, 1) * LATERAL + fwd.reshape(-1, 1) * FORWARD
               + vert.reshape(-1, 1) * UP)
    return centroid + reach * offsets


def prune(origins: np.ndarray, traj: np.ndarray, reach: float,
          cameras: np.ndarray) -> tuple[np.ndarray, dict]:
    """A.4's hard constraints. Returns the keep mask and a per-rule rejection count.

    Distances do not depend on the base orientation, so this runs on the 245 origins
    once instead of on all 11,025 pose candidates. ``cameras`` are the world-frame
    camera centres of every frame: A.4's minimum camera distance has to hold for the
    whole clip, so the closest approach is what counts.
    """
    dist = np.linalg.norm(traj[None, :, :] - origins[:, None, :], axis=-1)
    camera_dist = np.linalg.norm(cameras[None, :, :] - origins[:, None, :], axis=-1)
    too_close_to_camera = camera_dist.min(axis=1) < config.BASE_MIN_CAMERA_DIST_M
    out_of_reach = (dist > config.BASE_MAX_REACH_RATIO * reach).any(axis=1)
    too_close_to_base = (dist < config.BASE_MIN_TRAJ_DIST_M).any(axis=1)
    keep = ~(too_close_to_camera | out_of_reach | too_close_to_base)
    return keep, {"candidates": int(len(origins)),
                  "rejected_camera_distance": int(too_close_to_camera.sum()),
                  "rejected_out_of_reach": int(out_of_reach.sum()),
                  "rejected_too_close": int(too_close_to_base.sum()),
                  "kept": int(keep.sum())}


def reach_ratio(origins: np.ndarray, key_positions: np.ndarray, reach: float) -> np.ndarray:
    """rho_bar of eq.(8): mean keyframe distance from the base, over the reach."""
    dist = np.linalg.norm(key_positions[None, :, :] - origins[:, None, :], axis=-1)
    return dist.mean(axis=1) / reach


def score(feasibility: np.ndarray | float, rho: np.ndarray | float) -> np.ndarray | float:
    """Equation (8): S = FR - 5.0 |rho_bar - 0.65|."""
    return feasibility - config.BASE_REACH_PENALTY * np.abs(
        rho - config.BASE_TARGET_REACH_RATIO)


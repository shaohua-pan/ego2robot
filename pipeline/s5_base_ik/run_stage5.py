"""Stage 5 entry point: robot base pose search + IK (paper section 3.2, A.4).

    python -m pipeline.s5_base_ik.run_stage5 \
        --h5 <stage2>/hand_pose.h5 --robot panda --out-dir <dir>

The search follows eq.(4) and eq.(8) exactly but does not evaluate the full grid.
Because the reach penalty of eq.(8) depends only on the base *position*, a position
whose penalty makes ``1 - penalty`` (its score with a perfect feasibility rate) worse
than the fifth best score found so far cannot enter the top five, so it can be
skipped without changing the result. The same bound prunes keyframes inside a
candidate. Positions are therefore visited in order of increasing penalty and the
loop stops as soon as the bound closes - the outcome is the same argmax the paper
describes, at a fraction of the 11,025 candidates per arm.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from pipeline import config, robots
from pipeline.s5_base_ik import candidates, keyframes
from pipeline.s5_base_ik.ik import IKSolver

HANDS = ("left", "right")
# A.4 generates lateral offsets "sign-flipped per arm"; the wearer's left hand is on
# the camera's -x side, so that arm's base goes there too.
LATERAL_SIGN = {"left": -1.0, "right": 1.0}# Eq.(2) sets the grasp axis to s (p_thumb - p_vf) / w with s = -1 for the left hand,
# so the two hands hand over grasp frames that differ by a 180 degree turn about the
# approach axis. A parallel jaw is symmetric about that axis, i.e. both describe the
# same physical grasp, but only one of them is reachable by a wrist with less than a
# full turn of travel (Panda's joint7 stops at +/-166 deg). Undoing the flip for the
# left arm - equivalent to evaluating eq.(2) with s = +1 - puts both arms in the same
# wrist convention and is what makes the left arm solvable at all.
MIRROR = np.diag([1.0, -1.0, -1.0])



@dataclasses.dataclass
class Arm:
    """One hand's retargeted trajectory, in the camera frame."""

    hand: str
    position: np.ndarray
    rotation: np.ndarray
    width: np.ndarray
    valid: np.ndarray
    keys: np.ndarray


@dataclasses.dataclass
class BasePose:
    position: np.ndarray
    rotation: np.ndarray
    feasibility: float
    rho: float
    score: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ego2Robot stage 5: base pose search + IK")
    p.add_argument("--h5", type=Path, required=True, help="stage-2 hand_pose.h5")
    p.add_argument("--robot", default="panda", choices=robots.available())
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--hands", nargs="+", default=list(HANDS), choices=HANDS)
    p.add_argument("--max-frames", type=int, default=0,
                   help="keep only the first N frames; one base pose is only meaningful "
                        "over a continuous shot, so a clip with a cut has to be split")
    return p.parse_args()


def load_arm(h5: h5py.File, hand: str, limit: int = 0) -> Arm:
    """One hand's retargeted trajectory, in the world frame.

    Eq.(4) places one base for the whole clip, so it has to be solved where the base
    actually stands still: the world. Stage ② stores both frames; the camera-frame copy
    is only used for rendering, per frame, in stage ⑥.
    """
    group = h5[hand]["gripper"]
    end = limit or len(group["valid"])
    valid = group["valid"][:end]
    position = group["position"][:end]
    quat = group["quat"][:end]
    rotation = Rotation.from_quat(quat).as_matrix()
    if hand == "left":
        rotation = rotation @ MIRROR
    keys = keyframes.select(position, quat, valid)
    return Arm(hand, position, rotation, group["width"][:end], valid, keys)


def camera_centres(h5: h5py.File, hand: str, limit: int = 0) -> np.ndarray:
    """World-frame camera positions, for A.4's minimum camera distance."""
    group = h5[hand]["refined"]
    end = limit or len(group["cam_t"])
    rot, trans = group["cam_R"][:end], group["cam_t"][:end]
    return np.array([-rot[i].T @ trans[i] for i in range(len(trans))])


def to_base(base: np.ndarray, base_rot: np.ndarray, position: np.ndarray,
            rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Camera-frame target -> base frame, i.e. the T_base^-1 T^ee of eq.(4)."""
    return base_rot.T @ (position - base), base_rot.T @ rotation


def feasibility(solver: IKSolver, arm: Arm, base: np.ndarray, base_rot: np.ndarray,
                cutoff: float | None = None) -> tuple[float, list[float]]:
    """IK feasibility rate over the keyframes, with an optional early exit.

    ``cutoff`` is the feasibility rate this candidate must still be able to beat;
    once it cannot, the remaining keyframes are skipped. Every solve starts from the
    home configuration so that the rate does not depend on keyframe order.
    """
    n = len(arm.keys)
    ok, errors = 0, []
    for done, frame in enumerate(arm.keys):
        position, rotation = to_base(base, base_rot, arm.position[frame], arm.rotation[frame])
        result = solver.solve(position, rotation, arm.width[frame])
        ok += result.feasible
        errors.append(result.position_error_m)
        if cutoff is not None and (ok + n - done - 1) / n <= cutoff:
            break
    return ok / n, errors


def search_arm(solver: IKSolver, arm: Arm, reach: float, centroid: np.ndarray,
               cameras: np.ndarray) -> tuple[list[BasePose], dict]:
    """Eq.(4) + eq.(8) search over the A.4 grid, bounded as described in the module docstring."""
    origins = candidates.positions(centroid, reach, LATERAL_SIGN[arm.hand])
    keep, stats = candidates.prune(origins, arm.position[arm.valid], reach, cameras)
    rho = candidates.reach_ratio(origins, arm.position[arm.keys], reach)
    penalty = config.BASE_REACH_PENALTY * np.abs(rho - config.BASE_TARGET_REACH_RATIO)
    rotations = candidates.orientations()
    top: list[BasePose] = []
    evaluated = visited = 0
    for idx in sorted(np.flatnonzero(keep), key=lambda i: penalty[i]):
        full = len(top) >= config.BASE_TOPK_PER_ARM
        if full and 1.0 - penalty[idx] <= top[-1].score:
            break
        visited += 1
        for rotation in rotations:
            cutoff = top[-1].score + penalty[idx] if full else None
            rate, _ = feasibility(solver, arm, origins[idx], rotation, cutoff)
            evaluated += 1
            value = float(candidates.score(rate, rho[idx]))
            if not full or value > top[-1].score:
                top.append(BasePose(origins[idx], rotation, rate, float(rho[idx]), value))
                top.sort(key=lambda pose: -pose.score)
                del top[config.BASE_TOPK_PER_ARM:]
                full = len(top) >= config.BASE_TOPK_PER_ARM
    stats.update({"orientations": len(rotations), "positions_visited": visited,
                  "candidates_scored": evaluated,
                  "grid_size": int(len(origins) * len(rotations)),
                  "keyframes": len(arm.keys)})
    return top, stats


def select_pair(top: dict[str, list[BasePose]], clearance: float) -> tuple[dict, dict]:
    """A.4's joint verification of all 25 left-right combinations.

    The paper does not say what the joint check is; the only interaction two
    independently placed arms have is physical overlap, so pairs whose bases are
    closer than the sum of their base radii are rejected and the remaining pair with
    the best combined score wins. Arm-versus-arm collision along the trajectory is
    not checked here - that is what the L1 self-collision curation is for.
    """
    hands = list(top)
    if len(hands) == 1:
        return {hands[0]: top[hands[0]][0]}, {"pairs_checked": 0, "pairs_rejected": 0}
    left, right = top[hands[0]], top[hands[1]]
    rejected, best, chosen = 0, -np.inf, None
    for first in left:
        for second in right:
            if np.linalg.norm(first.position - second.position) < clearance:
                rejected += 1
                continue
            if first.score + second.score > best:
                best = first.score + second.score
                chosen = {hands[0]: first, hands[1]: second}
    if chosen is None:
        raise SystemExit("no left/right base pair clears the minimum separation")
    return chosen, {"pairs_checked": len(left) * len(right), "pairs_rejected": rejected,
                    "combined_score": round(float(best), 4),
                    "base_clearance_m": round(clearance, 4)}


def solve_trajectory(solver: IKSolver, robot: robots.RobotModel, arm: Arm,
                     base: BasePose) -> dict:
    """Frame-by-frame IK for the chosen base pose, warm-started from the last frame."""
    n = len(arm.position)
    qpos = np.tile(robot.home_qpos, (n, 1))
    feasible = np.zeros(n, dtype=bool)
    pos_err = np.full(n, np.nan)
    ori_err = np.full(n, np.nan)
    warm = None
    for frame in np.flatnonzero(arm.valid):
        position, rotation = to_base(base.position, base.rotation,
                                     arm.position[frame], arm.rotation[frame])
        result = solver.solve(position, rotation, arm.width[frame], warm)
        qpos[frame] = result.qpos
        feasible[frame] = result.feasible
        pos_err[frame] = result.position_error_m
        ori_err[frame] = result.orientation_error_rad
        warm = result.qpos if result.feasible else None
    return {"qpos": qpos, "feasible": feasible,
            "position_error_m": pos_err, "orientation_error_rad": ori_err}


def save(path: Path, robot: robots.RobotModel, arms: dict[str, Arm],
         bases: dict[str, BasePose], solved: dict[str, dict],
         intrins: np.ndarray, cam_rot: np.ndarray, cam_trans: np.ndarray) -> None:
    with h5py.File(path, "w") as out:
        out.attrs["robot"] = robot.spec.name
        out.attrs["mjcf"] = robot.spec.mjcf
        out.attrs["reach_m"] = robot.spec.reach_m
        out.attrs["tcp_offset"] = robot.tcp_offset
        out.attrs["tcp_quat"] = robot.tcp_quat
        # Stage 6 renders through the same pinhole the trajectory was estimated with,
        # and needs the camera trajectory to place a world-fixed base per frame.
        out.attrs["intrins"] = intrins
        out.create_dataset("cam_R", data=cam_rot, compression="gzip")
        out.create_dataset("cam_t", data=cam_trans, compression="gzip")
        for hand, arm in arms.items():
            group = out.create_group(hand)
            group.attrs["base_position_world"] = bases[hand].position
            group.attrs["base_rotation_world"] = bases[hand].rotation
            group.attrs["feasibility_rate"] = bases[hand].feasibility
            group.attrs["reach_ratio"] = bases[hand].rho
            group.attrs["score"] = bases[hand].score
            group.attrs["keyframes"] = arm.keys
            group.attrs["mirrored"] = hand == "left"
            for key, value in solved[hand].items():
                group.create_dataset(key, data=value, compression="gzip")
            group.create_dataset("width", data=arm.width, compression="gzip")


def main() -> None:
    args = parse_args()
    robot = robots.load(args.robot)
    solver = IKSolver(robot)
    reach = robot.spec.reach_m

    with h5py.File(args.h5, "r") as h5:
        arms = {hand: load_arm(h5, hand, args.max_frames) for hand in args.hands}
        cameras = camera_centres(h5, args.hands[0], args.max_frames)
        source = dict(h5.attrs)
        refined = h5[args.hands[0]]["refined"]
        intrins = refined["intrins"][:]
        end = args.max_frames or len(refined["cam_t"])
        cam_rot, cam_trans = refined["cam_R"][:end], refined["cam_t"][:end]
    centroid = np.concatenate([arm.position[arm.valid] for arm in arms.values()]).mean(axis=0)

    stats = {"robot": args.robot, "reach_m": reach,
             "measured_reach_m": round(robot.measure_reach(), 4),
             "n_frames": len(next(iter(arms.values())).position),
             "frame": "world",
             "centroid_world": centroid.round(4).tolist(), "arms": {}}
    top, timing = {}, {}
    for hand, arm in arms.items():
        start = time.time()
        top[hand], search_stats = search_arm(solver, arm, reach, centroid, cameras)
        timing[hand] = time.time() - start
        best = top[hand][0]
        print(f"[stage5] {hand}: FR {best.feasibility:.2f} rho {best.rho:.2f} "
              f"score {best.score:.3f} after {search_stats['candidates_scored']} "
              f"candidates in {timing[hand]:.0f}s")
        stats["arms"][hand] = {"search": search_stats, "search_seconds": round(timing[hand], 1),
                              "top": [{"score": round(pose.score, 4),
                                       "feasibility": round(pose.feasibility, 4),
                                       "reach_ratio": round(pose.rho, 4),
                                       "position_world": pose.position.round(4).tolist()}
                                      for pose in top[hand]]}

    bases, pair_stats = select_pair(top, 2.0 * robot.base_radius())
    stats["pair"] = pair_stats
    solved = {hand: solve_trajectory(solver, robot, arm, bases[hand])
              for hand, arm in arms.items()}
    for hand, result in solved.items():
        valid = arms[hand].valid
        ok = int(result["feasible"][valid].sum())
        stats["arms"][hand]["trajectory"] = {
            "frames": int(valid.sum()), "feasible": ok,
            "feasible_ratio": round(ok / max(int(valid.sum()), 1), 4),
            "median_position_error_mm": round(
                float(np.nanmedian(result["position_error_m"][valid]) * 1000), 4),
            "p90_position_error_mm": round(
                float(np.nanpercentile(result["position_error_m"][valid], 90) * 1000), 4)}
        print(f"[stage5] {hand}: {ok}/{int(valid.sum())} frames solved")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_h5 = args.out_dir / f"robot_{args.robot}.h5"
    save(out_h5, robot, arms, bases, solved, intrins, cam_rot, cam_trans)
    stats["source"] = {"h5": str(args.h5), "fps": float(source.get("fps", 30.0)),
                       "width": int(source.get("width", 0)),
                       "height": int(source.get("height", 0)),
                       "video": str(source.get("video", ""))}
    (args.out_dir / f"stage5_{args.robot}_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[save] {out_h5}, {args.out_dir / f'stage5_{args.robot}_stats.json'}")


if __name__ == "__main__":
    main()





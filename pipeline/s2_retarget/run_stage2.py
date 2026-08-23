"""Stage 2 entry point: refined hand poses -> gripper end-effector trajectories.

Order of operations (paper section 3.1 + appendices A.1, A.3):

1. retarget each hand track to a parallel-jaw gripper pose (eq. 1-3)
2. gap handling (A.1): gaps <= 10 frames are interpolated; longer gaps need the
   robot home configuration and stay invalid until step 5 supplies it
3. velocity filter (eq. 7), 2 rounds
4. Savitzky-Golay on position and width, Gaussian-weighted SLERP on orientation

Retargeting and smoothing run in the world frame: the robot base is static there,
whereas the egocentric camera moves with the head, so smoothing in the camera
frame would mix head motion into the hand trajectory. The camera-frame trajectory
the paper uses downstream (A.4 base search, camera-frame EEF actions) is written
alongside it, obtained with the per-frame extrinsics from VIPE.

    python -m pipeline.s2_retarget.run_stage2 --h5 <hand_pose.h5>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from pipeline import geometry
from pipeline.s1_hand_pose import gap_handling
from pipeline.s1_hand_pose.tracking import HAND_NAMES
from pipeline.s2_retarget import retarget as rt
from pipeline.s2_retarget import smoothing


def process_hand(group: h5py.Group, hand: int, fps: float) -> tuple[dict, dict]:
    refined = group["refined"]
    kp3d = refined["kp3d_world"][:]
    traj = rt.retarget(kp3d, refined["valid"][:].astype(bool), is_right=bool(hand))

    position, quat, width, valid, gaps = gap_handling.fill_gaps(
        traj.position, traj.quat, traj.valid, widths=traj.width)

    position, quat, vel_stats = smoothing.velocity_filter(position, quat, valid, fps)
    position = smoothing.smooth_series(position)
    width = smoothing.smooth_series(width)
    quat = smoothing.gaussian_slerp(quat)

    pos_cam, quat_cam = geometry.pose_to_camera(
        position, quat, refined["cam_R"][:], refined["cam_t"][:])
    data = {"position": position, "quat": quat, "width": width,
            "position_cam": pos_cam, "quat_cam": quat_cam,
            "valid": valid, "frozen": traj.frozen}
    stats = {
        "valid_frames": int(valid.sum()),
        "orientation_frozen": int(traj.frozen.sum()),
        "gaps": {"small_filled": gaps.small_filled, "large_unfilled": gaps.unfilled,
                 "runs": gaps.gaps},
        "velocity_filter": vel_stats,
        "width_m": [round(float(width[valid].min()), 4),
                    round(float(width[valid].max()), 4),
                    round(float(np.median(width[valid])), 4)] if valid.any() else None,
    }
    return data, stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ego2Robot stage 2: hand -> gripper")
    p.add_argument("--h5", type=Path, required=True, help="stage-1 hand_pose.h5 with refined groups")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summary = {}
    with h5py.File(args.h5, "a") as f:
        fps = float(dict(f.attrs)["fps"])
        for hand, name in HAND_NAMES.items():
            if name not in f or "refined" not in f[name]:
                continue
            data, stats = process_hand(f[name], hand, fps)
            if "gripper" in f[name]:
                del f[name]["gripper"]
            g = f[name].create_group("gripper")
            for key, value in data.items():
                g.create_dataset(key, data=value, compression="gzip")
            summary[name] = stats
            lo, hi, med = stats["width_m"] or (0, 0, 0)
            print(f"[stage2] {name}: {stats['valid_frames']} valid frames, "
                  f"width {lo * 100:.1f}-{hi * 100:.1f}cm (median {med * 100:.1f}), "
                  f"gaps filled {stats['gaps']['small_filled']}, "
                  f"unfilled {stats['gaps']['large_unfilled']}, "
                  f"orientation frozen {stats['orientation_frozen']}, "
                  f"velocity outliers {stats['velocity_filter']}")
    out = args.h5.parent / "stage2_stats.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"[save] {args.h5} (gripper groups), {out}")


if __name__ == "__main__":
    main()

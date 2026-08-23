"""Visualize the stage-⑤ product: the retargeted arms as solved by IK.

Nothing is rendered with OpenGL here - the robot is drawn as a projected skeleton,
one segment per body-to-parent link, so the placement can be checked before the
depth-aware compositing of step ⑥ exists. What to look for: the base marker sitting
where a real arm could be mounted, the chain staying inside the frame, and the TCP
marker landing on the human hand it was retargeted from.

    python -m pipeline.viz.stage5 --stats <stage5_panda_stats.json> --out <out.mp4>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np

from pipeline import geometry, robots
from pipeline.viz import draw, render

BASE_COLOR = (200, 200, 200)
ERROR_MAX_MM = 20.0
LINK_THICKNESS = 4


def load(stats_path: Path) -> dict:
    stats = json.loads(stats_path.read_text())
    h5_path = stats_path.parent / f"robot_{stats['robot']}.h5"
    robot = robots.load(stats["robot"])
    arms = {}
    with h5py.File(h5_path, "r") as h5:
        intrins = h5.attrs["intrins"]
        cam_rot, cam_trans = h5["cam_R"][:], h5["cam_t"][:]
        for hand in h5:
            if hand in ("cam_R", "cam_t"):
                continue
            group = h5[hand]
            arms[hand] = {
                "qpos": group["qpos"][:],
                "feasible": group["feasible"][:],
                "error_mm": group["position_error_m"][:] * 1000.0,
                "base": group.attrs["base_position_world"],
                "base_rotation": group.attrs["base_rotation_world"],
                "feasibility": float(group.attrs["feasibility_rate"]),
            }
    return {"stats": stats, "robot": robot, "arms": arms, "intrins": intrins,
            "cam_R": cam_rot, "cam_t": cam_trans,
            "video": Path(stats["source"]["video"]), "fps": stats["source"]["fps"],
            "n_frames": len(next(iter(arms.values()))["qpos"])}


def annotate(data: dict, img: np.ndarray, i: int) -> None:
    robot = data["robot"]
    parents = robot.model.body_parentid
    lines = [f"{robot.spec.name}  frame {i}"]
    # The base is fixed in the world, so it has to be brought into this frame's camera.
    cam_rot, cam_trans = data["cam_R"][i], data["cam_t"][i]
    for hand, arm in data["arms"].items():
        color = draw.HAND_COLORS[hand]
        qpos = arm["qpos"][i]
        base = cam_rot @ arm["base"] + cam_trans
        rotation = cam_rot @ arm["base_rotation"]
        bodies = base + robot.body_positions(qpos) @ rotation.T
        tcp = base + rotation @ robot.tcp_pose(qpos)[0]
        uv = geometry.project(bodies, data["intrins"])
        for body in range(1, robot.model.nbody):
            parent = parents[body]
            # Bodies parented to the world have no segment to draw, and anything
            # behind the pinhole projects to nonsense.
            if parent == 0 or min(bodies[body, 2], bodies[parent, 2]) <= 0.05:
                continue
            cv2.line(img, draw._pt(uv[parent]), draw._pt(uv[body]), color,
                     LINK_THICKNESS, cv2.LINE_AA)
        cv2.circle(img, draw._pt(uv[robot.base]), 10, BASE_COLOR, 2, cv2.LINE_AA)
        cv2.circle(img, draw._pt(geometry.project(tcp, data["intrins"])), 6, color, -1,
                   cv2.LINE_AA)
        state = "ok" if arm["feasible"][i] else f"IK {arm['error_mm'][i]:.0f}mm off"
        lines.append(f"{hand}: FR {arm['feasibility']:.2f}, {state}")
    draw.hud(img, lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize stage-5 base search and IK")
    p.add_argument("--stats", type=Path, required=True, help="stage5_<robot>_stats.json")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--video", type=Path, default=None, help="override the source video")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = load(args.stats)
    series = [render.Series(arm["error_mm"], np.isfinite(arm["error_mm"]),
                            draw.HAND_COLORS[hand])
              for hand, arm in data["arms"].items()]
    caption = ("IK position error per frame; a solve counts as feasible below "
               "1e-5 m (paper A.4)")
    render.render(args.video or data["video"], args.out, data["n_frames"], data["fps"],
                  draw=lambda img, i: annotate(data, img, i),
                  strip=lambda w, _h: render.Strip(series, w, ERROR_MAX_MM,
                                                   guides=[(1.0, "1mm"), (10.0, "10mm")],
                                                   caption=caption))


if __name__ == "__main__":
    main()


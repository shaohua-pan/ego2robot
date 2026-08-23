"""Visualize the stage-② product: the retargeted parallel-jaw gripper trajectory.

Drawn per frame: the jaw line between the thumb tip and the virtual fingertip
(eq. 2), the three axes of the grasp frame (eq. 3, red = x approach,
green = y normal, blue = z grasp axis), and the smoothed opening width. The strip
plots the width against the 1 cm degeneracy threshold from appendix A.3.

    python -m pipeline.viz.stage2 --h5 <hand_pose.h5> --out <stage2.mp4>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from pipeline import config
from pipeline.viz import draw, render

WIDTH_MAX_M = 0.10


def load(h5_path: Path) -> dict:
    out = {"hands": {}}
    with h5py.File(h5_path, "r") as f:
        meta = dict(f.attrs)
        out.update(n_frames=int(meta["n_frames"]), fps=float(meta["fps"]),
                   video=Path(meta["video"]))
        for name in draw.HAND_COLORS:
            if name not in f or "gripper" not in f[name]:
                continue
            grip = f[name]["gripper"]
            out["hands"][name] = {
                "position_cam": grip["position_cam"][:],
                "quat_cam": grip["quat_cam"][:],
                "width": grip["width"][:],
                "valid": grip["valid"][:].astype(bool),
                "frozen": grip["frozen"][:].astype(bool),
                "intrins": f[name]["refined"]["intrins"][:],
            }
    return out


def build_strip(data: dict, width: int, _height: int) -> render.Strip:
    series = [render.Series(hand["width"], hand["valid"], draw.HAND_COLORS[name])
              for name, hand in data["hands"].items()]
    return render.Strip(
        series, width, WIDTH_MAX_M,
        guides=[(config.GRIPPER_WIDTH_MIN_M, "1cm (degenerate)"), (0.08, "8cm")],
        caption="gripper opening width after Savitzky-Golay smoothing")


def annotate(data: dict, img: np.ndarray, i: int) -> None:
    lines = [f"frame {i}"]
    for name, hand in data["hands"].items():
        if not hand["valid"][i]:
            lines.append(f"{name}: no pose")
            continue
        draw.gripper(img, hand["position_cam"][i], hand["quat_cam"][i],
                     float(hand["width"][i]), hand["intrins"])
        note = " (orientation held)" if hand["frozen"][i] else ""
        lines.append(f"{name} gripper: {hand['width'][i] * 100:.1f}cm open, "
                     f"{hand['position_cam'][i][2] * 100:.1f}cm deep{note}")
    draw.hud(img, lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize stage-2 gripper trajectories")
    p.add_argument("--h5", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--video", type=Path, default=None, help="override the source video")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = load(args.h5)
    if not data["hands"]:
        raise SystemExit(f"{args.h5} has no gripper groups; run stage 2 first")
    render.render(args.video or data["video"], args.out, data["n_frames"], data["fps"],
                  draw=lambda img, i: annotate(data, img, i),
                  strip=lambda w, h: build_strip(data, w, h))


if __name__ == "__main__":
    main()

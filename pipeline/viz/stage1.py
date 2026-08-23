"""Visualize the stage-① product: per-frame WiLoR vs Dyn-HaMR-refined hand poses.

The refined poses only exist in 3D, so they are projected back with the intrinsics
Dyn-HaMR optimized against and drawn over the raw keypoints. In 2D the two nearly
coincide; the difference is depth, which the bottom strip plots against the paper's
[0.05, 0.4] m constraint band.

    python -m pipeline.viz.stage1 --h5 <hand_pose.h5> --out <stage1.mp4>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from pipeline import config, geometry
from pipeline.viz import draw, render

DEPTH_MAX_M = 0.5


def load(h5_path: Path) -> dict:
    """Per-hand raw 2D keypoints, projected refined keypoints and both depth traces."""
    out = {"hands": {}}
    with h5py.File(h5_path, "r") as f:
        meta = dict(f.attrs)
        out.update(n_frames=int(meta["n_frames"]), fps=float(meta["fps"]),
                   video=Path(meta["video"]))
        for name in draw.HAND_COLORS:
            if name not in f or "refined" not in f[name]:
                continue
            group, refined = f[name], f[name]["refined"]
            intrins = refined["intrins"][:]
            kp3d_cam = refined["kp3d_cam"][:]
            # WiLoR's tz is only comparable after the focal-length correction the
            # bridge applies when exporting to Dyn-HaMR.
            scale = float(intrins[0]) / float(meta["focal_length"])
            out["hands"][name] = {
                "raw_kp2d": group["kp2d"][:],
                "raw_valid": group["valid"][:].astype(bool),
                "raw_depth": group["cam_t"][:, 2] * scale,
                "kp2d": geometry.project(kp3d_cam, intrins),
                "depth": kp3d_cam[:, config.WRIST, 2],
                "valid": refined["valid"][:].astype(bool),
            }
    return out


def build_strip(data: dict, width: int, _height: int) -> render.Strip:
    series = []
    for name, hand in data["hands"].items():
        color = draw.HAND_COLORS[name]
        dim = tuple(int(c * 0.45) for c in color)
        series.append(render.Series(hand["raw_depth"], hand["raw_valid"], dim, 1))
        series.append(render.Series(hand["depth"], hand["valid"], color, 2))
    return render.Strip(
        series, width, DEPTH_MAX_M,
        guides=[(config.HAND_DEPTH_MIN_M, "0.05m"), (config.HAND_DEPTH_MAX_M, "0.4m")],
        caption="wrist depth: dim = per-frame WiLoR (focal-corrected), bright = refined")


def annotate(data: dict, img: np.ndarray, i: int) -> None:
    for hand in data["hands"].values():
        if hand["raw_valid"][i]:
            draw.skeleton(img, hand["raw_kp2d"][i], draw.RAW_COLOR, 1, 2)
    lines = [f"frame {i}"]
    for name, hand in data["hands"].items():
        if not hand["valid"][i]:
            lines.append(f"{name}: no detection")
            continue
        draw.skeleton(img, hand["kp2d"][i], draw.HAND_COLORS[name])
        lines.append(f"{name} wrist: {hand['depth'][i] * 100:.1f}cm"
                     f" (raw {hand['raw_depth'][i] * 100:.1f})")
    draw.hud(img, lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize stage-1 hand poses")
    p.add_argument("--h5", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--video", type=Path, default=None, help="override the source video")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = load(args.h5)
    if not data["hands"]:
        raise SystemExit(f"{args.h5} has no refined groups; run dynhamr_import first")
    render.render(args.video or data["video"], args.out, data["n_frames"], data["fps"],
                  draw=lambda img, i: annotate(data, img, i),
                  strip=lambda w, h: build_strip(data, w, h))


if __name__ == "__main__":
    main()

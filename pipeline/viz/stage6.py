"""Visualize the stage-⑥ product: the finished robot video.

This is the pipeline's output, so the default is a clean video with no overlays at
all - what a policy would actually be trained on. ``--annotate`` adds the frame
number, the robot coverage and a strip of how many gripper pixels the depth test
hid, for inspection.

    python -m pipeline.viz.stage6 --stats <stage6_panda_stats.json> --out <out.mp4>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from pipeline.viz import draw, render

OCCLUDED_COLOR = (60, 200, 255)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize stage-6 composited frames")
    p.add_argument("--stats", type=Path, required=True, help="stage6_<robot>_stats.json")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--annotate", action="store_true", help="add a HUD and a strip")
    p.add_argument("--fps", type=float, default=30.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    stats = json.loads(args.stats.read_text())
    frames = sorted((args.stats.parent / "composited").glob("*.png"))
    if not frames:
        raise SystemExit(f"no composited frames next to {args.stats}")
    occluded = np.asarray(stats["per_frame"]["gripper_occluded_px"], dtype=float)
    pixels = stats["source"]["width"] * stats["source"]["height"]
    coverage = np.asarray(stats["per_frame"]["robot_px"], dtype=float) / pixels * 100.0

    # The source here is a PNG directory rather than a video, so the frames are
    # encoded directly instead of going through render.render's decode loop.
    strip = None
    if args.annotate:
        strip = render.Strip(
            [render.Series(occluded, np.ones(len(occluded), bool), OCCLUDED_COLOR)],
            stats["source"]["width"], max(float(occluded.max()), 1.0),
            caption="gripper pixels hidden by the eq.(9) depth test")
    writer = None
    for i, path in enumerate(frames):
        img = cv2.imread(str(path))
        if args.annotate:
            draw.hud(img, [f"{stats['robot']}  frame {i}",
                           f"robot {coverage[i]:.1f}% of frame",
                           f"gripper pixels hidden by depth: {int(occluded[i])}"])
        if strip is not None:
            img = np.vstack([img, strip.frame(i)])
        if writer is None:
            writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"),
                                     args.fps, (img.shape[1], img.shape[0]))
        writer.write(img)
    writer.release()
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()

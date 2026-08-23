"""Visualize the stage-③ product: the human arm masks that step 4 will inpaint.

Each frame gets a translucent fill plus a bright contour, and the strip plots mask
area as a fraction of the frame so dropouts and the A.4 area filter are visible.

    python -m pipeline.viz.stage3 --stats <stage3_stats.json> --out <stage3.mp4>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from pipeline.viz import draw, render

MASK_COLOR = (60, 120, 255)   # BGR
FILL_ALPHA = 0.45
AREA_MAX = 0.25


def load(stats_path: Path, mask_dir: Path | None) -> dict:
    payload = json.loads(stats_path.read_text())
    meta, post = payload["meta"], payload["postprocess"]
    mask_dir = mask_dir or stats_path.parent / "arm_mask"
    return {
        "video": Path(meta["video"]),
        "n_frames": int(meta["n_frames"]),
        "fps": float(meta["fps"]),
        "prompt": meta["prompt"],
        "mask_dir": mask_dir,
        "area": np.asarray(post["area_ratio"], dtype=float),
        "interpolated": set(post["interpolated"]),
        "replaced": set(post["replaced"]),
        "missing": set(post["missing"]),
    }


def read_mask(data: dict, i: int, shape: tuple[int, int]) -> np.ndarray | None:
    path = data["mask_dir"] / f"{i:06d}.png"
    if not path.exists():
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 127


def build_strip(data: dict, width: int, _height: int) -> render.Strip:
    area = data["area"]
    series = [render.Series(area, area > 0, MASK_COLOR)]
    return render.Strip(series, width, AREA_MAX,
                        guides=[(0.05, "5%"), (0.15, "15%")],
                        caption=f"arm mask area / frame area, SAM 3 prompt '{data['prompt']}'")


def annotate(data: dict, img: np.ndarray, i: int) -> None:
    mask = read_mask(data, i, img.shape[:2])
    lines = [f"frame {i}"]
    if mask is None:
        lines.append("no arm mask")
    else:
        overlay = np.zeros_like(img)
        overlay[mask] = MASK_COLOR
        cv2.addWeighted(overlay, FILL_ALPHA, img, 1.0, 0.0, dst=img)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (255, 255, 255), 2, cv2.LINE_AA)
        note = (" (interpolated)" if i in data["interpolated"]
                else " (replaced)" if i in data["replaced"] else "")
        lines.append(f"arm mask: {data['area'][i]:.2%} of frame{note}")
    draw.hud(img, lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize stage-3 arm masks")
    p.add_argument("--stats", type=Path, required=True, help="stage3_stats.json")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mask-dir", type=Path, default=None)
    p.add_argument("--video", type=Path, default=None, help="override the source video")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = load(args.stats, args.mask_dir)
    render.render(args.video or data["video"], args.out, data["n_frames"], data["fps"],
                  draw=lambda img, i: annotate(data, img, i),
                  strip=lambda w, h: build_strip(data, w, h))


if __name__ == "__main__":
    main()

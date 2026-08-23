"""Visualize the stage-④ product: the video with the human arms inpainted away.

The inpainted frame fills the canvas; the original is inset in the corner and the
arm-mask contour is drawn faintly, so it is easy to check that the background was
reconstructed where the arms used to be and left alone everywhere else. The strip
plots the mean absolute difference from the original inside the mask.

    python -m pipeline.viz.stage4 --stats <stage4_stats.json> --out <stage4.mp4>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from pipeline.viz import draw, render

CONTOUR_COLOR = (60, 120, 255)
INSET_SCALE = 0.25
DIFF_MAX = 60.0


def load(stats_path: Path) -> dict:
    payload = json.loads(stats_path.read_text())
    meta = payload["meta"]
    root = stats_path.parent
    return {
        "video": Path(meta["video"]),
        "n_frames": int(meta["n_frames"]),
        "fps": float(meta["fps"]),
        "inpainted_dir": root / "inpainted",
        "mask_dir": Path(meta["mask_dir"]),
        "settings": meta["settings"],
        "diff": np.asarray(payload["comparison"]["diff_inside_per_frame"], dtype=float),
    }


def _read(path: Path, shape: tuple[int, int], gray: bool = False) -> np.ndarray | None:
    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), flag)
    if img is None:
        return None
    if img.shape[:2] != shape:
        img = cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return img


def annotate(data: dict, img: np.ndarray, i: int) -> None:
    shape = img.shape[:2]
    original = img.copy()
    inpainted = _read(data["inpainted_dir"] / f"{i:06d}.png", shape)
    lines = [f"frame {i}"]
    if inpainted is None:
        lines.append("no inpainted frame")
    else:
        mask = _read(data["mask_dir"] / f"{i:06d}.png", shape, gray=True)
        img[:] = inpainted
        if mask is not None:
            binary = (mask > 127).astype(np.uint8)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, CONTOUR_COLOR, 1, cv2.LINE_AA)
            lines.append(f"mean |diff| in mask: {data['diff'][i]:.1f}/255")

    inset = cv2.resize(original, None, fx=INSET_SCALE, fy=INSET_SCALE,
                       interpolation=cv2.INTER_AREA)
    h, w = inset.shape[:2]
    y0 = shape[0] - h - 12
    img[y0:y0 + h, 12:12 + w] = inset
    cv2.rectangle(img, (12, y0), (12 + w, y0 + h), (255, 255, 255), 2)
    cv2.putText(img, "original", (20, y0 + 26), draw.FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    draw.hud(img, lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize stage-4 hand removal")
    p.add_argument("--stats", type=Path, required=True, help="stage4_stats.json")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--video", type=Path, default=None, help="override the source video")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = load(args.stats)
    caption = ("mean |diff| from the original inside the arm mask, ProPainter "
               f"fp16={data['settings']['fp16']} dilation={data['settings']['mask_dilation']}")
    render.render(args.video or data["video"], args.out, data["n_frames"], data["fps"],
                  draw=lambda img, i: annotate(data, img, i),
                  strip=lambda w, _h: render.Strip(
                      [render.Series(data["diff"], data["diff"] > 0, CONTOUR_COLOR)],
                      w, DIFF_MAX, guides=[(20.0, "20"), (40.0, "40")], caption=caption))


if __name__ == "__main__":
    main()

"""Compare the stage-⑥ product across robot morphologies.

The pipeline is morphology-agnostic, so the same clip run through several entries of
:mod:`pipeline.robots` is the cheapest way to tell an implementation bug (wrong in
every column) from a morphology/clip mismatch (wrong in one column only).

Contact sheet - rows are frames, columns are the source, the stage-④ inpainting and
one column per robot::

    python -m pipeline.viz.compare --stats <s6_*/stage6_*_stats.json> --out sheet.png

Tiled video - the same columns, every frame::

    python -m pipeline.viz.compare --stats <...> --out tiles.mp4 --video
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

LABEL_HEIGHT = 34
FONT = cv2.FONT_HERSHEY_SIMPLEX


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare stage-6 output across robots")
    p.add_argument("--stats", type=Path, nargs="+", required=True,
                   help="stage6_<robot>_stats.json, one per column")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--video", action="store_true", help="write every frame as a video")
    p.add_argument("--frames", type=int, nargs="+", help="contact sheet rows")
    p.add_argument("--rows", type=int, default=4, help="contact sheet rows when --frames is absent")
    p.add_argument("--tile-width", type=int, default=480)
    p.add_argument("--fps", type=float, default=30.0)
    return p.parse_args()


def columns(stats_paths: list[Path]) -> list[tuple[str, list[Path]]]:
    """``(label, frames)`` per column: source, inpainted, then one robot each."""
    first = json.loads(stats_paths[0].read_text())
    out = [("source (human)", sorted(Path(first["source"]["inpainted"]).parent
                                    .joinpath("source").glob("*.png"))),
           ("stage 4 inpainted", sorted(Path(first["source"]["inpainted"]).glob("*.png")))]
    for path in stats_paths:
        stats = json.loads(path.read_text())
        frames = sorted((path.parent / "composited").glob("*.png"))
        reach = stats["coverage"]["robot_percent_mean"]
        out.append((f"{stats['robot']}  ({reach:.1f}% of frame)", frames))
    shortest = min(len(frames) for _, frames in out)
    return [(label, frames[:shortest]) for label, frames in out]


def tile(img: np.ndarray, label: str, width: int) -> np.ndarray:
    """Downscale one frame and caption it."""
    height = int(round(img.shape[0] * width / img.shape[1]))
    cell = np.zeros((height + LABEL_HEIGHT, width, 3), np.uint8)
    cell[LABEL_HEIGHT:] = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    cv2.putText(cell, label, (6, 23), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return cell


def row(cols: list[tuple[str, list[Path]]], index: int, width: int,
        with_labels: bool) -> np.ndarray:
    cells = []
    for label, frames in cols:
        caption = f"{label}" if with_labels else ""
        cells.append(tile(cv2.imread(str(frames[index])),
                          f"{caption}   #{index}" if with_labels else f"#{index}", width))
    return np.hstack(cells)


def main() -> None:
    args = parse_args()
    cols = columns(args.stats)
    count = len(cols[0][1])

    if args.video:
        writer = None
        for i in range(count):
            frame = row(cols, i, args.tile_width, True)
            if writer is None:
                writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"),
                                         args.fps, (frame.shape[1], frame.shape[0]))
            writer.write(frame)
        writer.release()
    else:
        picks = args.frames or np.linspace(0, count - 1, args.rows).round().astype(int).tolist()
        sheet = np.vstack([row(cols, i, args.tile_width, k == 0)
                           for k, i in enumerate(picks)])
        cv2.imwrite(str(args.out), sheet)
    print(f"[save] {args.out}  ({len(cols)} columns x {count} frames)")


if __name__ == "__main__":
    main()

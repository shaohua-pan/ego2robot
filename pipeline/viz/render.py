"""Video rendering helpers: a time-series strip and the decode/encode loop.

The strip is rendered once as a static background and only the frame cursor moves,
which keeps the per-frame cost to a copy.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import av
import cv2
import numpy as np

from pipeline.viz.draw import FONT

STRIP_HEIGHT = 150
BACKGROUND = 32


@dataclass
class Series:
    """One curve on a strip. ``valid`` gaps are left unconnected."""
    values: np.ndarray
    valid: np.ndarray
    color: tuple[int, int, int]
    thickness: int = 2


class Strip:
    """Static plot of one or more series against frame index."""

    def __init__(self, series: Sequence[Series], width: int, y_max: float,
                 guides: Sequence[tuple[float, str]] = (), caption: str = "",
                 height: int = STRIP_HEIGHT):
        self.width, self.height, self.y_max = width, height, y_max
        self.n_frames = max((len(s.values) for s in series), default=1)
        self._panel = np.full((height, width, 3), BACKGROUND, dtype=np.uint8)
        for value, label in guides:
            y = self._y(value)
            cv2.line(self._panel, (0, y), (width - 1, y), (90, 90, 90), 1, cv2.LINE_AA)
            cv2.putText(self._panel, label, (4, max(y - 4, 10)), FONT, 0.4,
                        (150, 150, 150), 1, cv2.LINE_AA)
        for s in series:
            self._plot(s)
        if caption:
            cv2.putText(self._panel, caption, (4, height - 6), FONT, 0.42,
                        (200, 200, 200), 1, cv2.LINE_AA)

    def _x(self, i: int) -> int:
        return int(round(i / max(self.n_frames - 1, 1) * (self.width - 1)))

    def _y(self, value: float) -> int:
        frac = min(max(float(value), 0.0), self.y_max) / self.y_max
        return int(round((self.height - 1) * (1.0 - frac)))

    def _plot(self, s: Series) -> None:
        for i in range(len(s.values) - 1):
            if not (s.valid[i] and s.valid[i + 1]):
                continue
            cv2.line(self._panel, (self._x(i), self._y(s.values[i])),
                     (self._x(i + 1), self._y(s.values[i + 1])), s.color,
                     s.thickness, cv2.LINE_AA)

    def frame(self, i: int) -> np.ndarray:
        panel = self._panel.copy()
        x = self._x(i)
        cv2.line(panel, (x, 0), (x, self.height - 1), (255, 255, 255), 1)
        return panel


def render(video: Path, out_path: Path, n_frames: int, fps: float,
           draw: Callable[[np.ndarray, int], None],
           strip: Callable[[int, int], Strip] | None = None) -> None:
    """Decode ``video``, let ``draw`` annotate each frame, write ``out_path``.

    :param strip called once with (width, height) to build the bottom panel
    """
    writer, panel = None, None
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for i, frame in enumerate(container.decode(stream)):
            if i >= n_frames:
                break
            img = frame.to_ndarray(format="bgr24")
            height, width = img.shape[:2]
            if writer is None:
                panel = strip(width, height) if strip is not None else None
                out_h = height + (panel.height if panel is not None else 0)
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                         max(fps, 1.0), (width, out_h))
            draw(img, i)
            writer.write(img if panel is None else np.vstack([img, panel.frame(i)]))
    if writer is not None:
        writer.release()
    print(f"[save] {out_path}")

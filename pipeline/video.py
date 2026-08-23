"""Video decoding shared by the stages.

PyAV is used instead of ``cv2.VideoCapture`` because the OpenCV wheel's ffmpeg
build cannot decode AV1, which is what the EgoDex clips use; PyAV ships libdav1d.
Every frame is decoded: the paper's per-source frame-rate matching ("Action Speed
Alignment") happens later, during training-data assembly.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import av
import numpy as np


def probe(path: Path) -> dict:
    """Container metadata without decoding any frames."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        ctx = stream.codec_context
        return {
            "codec": ctx.name,
            "width": ctx.width,
            "height": ctx.height,
            "fps": float(stream.average_rate) if stream.average_rate else 30.0,
            "n_frames": stream.frames or 0,
        }


def decode(path: Path, max_frames: int = 0) -> Iterator[np.ndarray]:
    """Yield BGR uint8 frames; ``max_frames = 0`` decodes the whole video."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for kept, frame in enumerate(container.decode(stream), start=1):
            yield frame.to_ndarray(format="bgr24")
            if max_frames and kept >= max_frames:
                break


class BgrToRgb:
    """Lazy BGR->RGB view over decoded frames.

    SAM 3 wants RGB while OpenCV/PyAV give us BGR. Converting a whole 1080p clip up
    front would double peak memory, and the segmenter only materializes one chunk at
    a time, so convert on access instead.
    """

    def __init__(self, frames: list[np.ndarray]):
        self._frames = frames

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return [f[:, :, ::-1] for f in self._frames[item]]
        return self._frames[item][:, :, ::-1]

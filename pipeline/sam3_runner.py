"""Text-prompted video segmentation with SAM 3 (paper appendix A.4).

Both places the paper uses SAM 3 go through this runner:

- A.1 filters WiLoR detections against a hand mask;
- A.4 segments the human arm ("person" prompt) for inpainting in step 4.

Paper A.4 settings reproduced here: long videos are processed in 400-frame chunks
with 50-frame overlap, masks in the overlap regions are merged with a bitwise OR,
and propagation is anchored on each chunk's middle frame (the iterator is run
forwards and backwards from it). The mask returned per frame is the union of all
instances SAM 3 matched to the prompt.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import torch

from pipeline import config


class Sam3VideoSegmenter:
    """Wraps ``Sam3VideoModel`` for chunked, text-prompted video segmentation."""

    def __init__(self, model_dir=None, device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16):
        from transformers import Sam3VideoModel, Sam3VideoProcessor

        self.model_dir = str(model_dir or config.SAM3_DIR)
        self.device = torch.device(device)
        self.dtype = dtype
        self.model = Sam3VideoModel.from_pretrained(self.model_dir).to(self.device, dtype=dtype)
        self.model.eval()
        self.processor = Sam3VideoProcessor.from_pretrained(self.model_dir)

    @torch.no_grad()
    def _segment_chunk(self, frames_rgb: Sequence[np.ndarray], prompt: str,
                       score_threshold: float) -> dict[int, np.ndarray]:
        session = self.processor.init_video_session(
            video=list(frames_rgb),
            inference_device=self.device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=self.dtype,
        )
        session = self.processor.add_text_prompt(inference_session=session, text=prompt)

        # Paper A.4 anchors propagation on the middle frame, so run the iterator
        # forwards and backwards from there.
        anchor = len(frames_rgb) // 2
        masks: dict[int, np.ndarray] = {}
        for reverse in (False, True):
            for model_outputs in self.model.propagate_in_video_iterator(
                    inference_session=session, start_frame_idx=anchor, reverse=reverse):
                out = self.processor.postprocess_outputs(session, model_outputs)
                mask = _union_masks(out.get("masks"), out.get("scores"), score_threshold)
                idx = model_outputs.frame_idx
                if mask is None:
                    masks.setdefault(idx, None)
                elif masks.get(idx) is None:
                    masks[idx] = mask
                else:
                    masks[idx] |= mask
        return {k: v for k, v in masks.items() if v is not None}

    def segment_iter(self, frames_rgb: Sequence[np.ndarray], prompt: str,
                     score_threshold: float = 0.0,
                     ) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ``(frame_idx, mask)`` in order, one frame at a time.

        Overlapping frames are yielded once, with the two chunks' masks OR'ed.
        ``score_threshold`` optionally drops low-scoring instances; the paper does
        not specify one, so by default every instance SAM 3 returns is kept.
        """
        n = len(frames_rgb)
        step = config.SAM3_CHUNK_FRAMES - config.SAM3_CHUNK_OVERLAP
        pending: dict[int, np.ndarray] = {}
        emitted = 0

        for start in range(0, max(n, 1), step):
            end = min(start + config.SAM3_CHUNK_FRAMES, n)
            chunk = self._segment_chunk(frames_rgb[start:end], prompt, score_threshold)
            for local_idx, mask in chunk.items():
                idx = start + local_idx
                if idx in pending:
                    pending[idx] |= mask
                elif idx >= emitted:
                    pending[idx] = mask
            # Frames before the next chunk's overlap window are final.
            final_before = n if end >= n else start + step
            while emitted < final_before:
                yield emitted, pending.pop(emitted, None)
                emitted += 1
            if end >= n:
                break

    def segment(self, frames_rgb: Sequence[np.ndarray], prompt: str,
                score_threshold: float = 0.0) -> np.ndarray:
        """Materialize all masks as a single ``(T, H, W)`` bool array."""
        h, w = frames_rgb[0].shape[:2]
        out = np.zeros((len(frames_rgb), h, w), dtype=bool)
        for idx, mask in self.segment_iter(frames_rgb, prompt, score_threshold):
            if mask is not None:
                out[idx] = mask
        return out


def _union_masks(masks, scores=None, score_threshold: float = 0.0) -> np.ndarray | None:
    """Reduce SAM 3's per-instance masks to one binary mask per frame."""
    if masks is None or len(masks) == 0:
        return None
    if isinstance(masks, torch.Tensor):
        masks = masks.detach().to("cpu")
        keep = masks > 0
        if score_threshold > 0.0 and scores is not None:
            sel = scores.detach().to("cpu") >= score_threshold
            if not bool(sel.any()):
                return None
            keep = keep[sel]
        return torch.any(keep, dim=0).numpy().astype(bool)
    arr = np.asarray(masks)
    if score_threshold > 0.0 and scores is not None:
        sel = np.asarray(scores) >= score_threshold
        if not sel.any():
            return None
        arr = arr[sel]
    return np.any(arr > 0, axis=0).astype(bool)

"""Stage 3 entry point: ego video -> human arm masks (paper section 3.2, A.4).

SAM 3 is prompted with the text "person" and propagated from each chunk's middle
frame; :mod:`pipeline.s3_arm_seg.arm_mask` then applies the paper's three
post-processing steps. Masks are written as one PNG per frame, named by the
zero-based frame index, which is what ProPainter consumes in step 4.

    python -m pipeline.s3_arm_seg.run_stage3 --video <clip.mp4> --out-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from pipeline import config, video
from pipeline.s3_arm_seg import arm_mask


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ego2Robot stage 3: arm segmentation")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--prompt", type=str, default=config.SAM3_ARM_PROMPT)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max-frames", type=int, default=0, help="debug: 0 means the whole video")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mask_dir = args.out_dir / "arm_mask"
    mask_dir.mkdir(parents=True, exist_ok=True)
    vmeta = video.probe(args.video)
    print(f"[video] {args.video.name} {vmeta['width']}x{vmeta['height']} "
          f"{vmeta['fps']:.2f}fps codec={vmeta['codec']}")

    frames = list(video.decode(args.video, args.max_frames))
    print(f"[video] {len(frames)} frames decoded")

    from pipeline.sam3_runner import Sam3VideoSegmenter
    segmenter = Sam3VideoSegmenter(device=args.device)
    t0 = time.time()
    masks: list[np.ndarray | None] = [None] * len(frames)
    for idx, mask in segmenter.segment_iter(video.BgrToRgb(frames), args.prompt):
        masks[idx] = mask
    found = sum(m is not None for m in masks)
    print(f"[sam3] prompt '{args.prompt}': {found}/{len(frames)} frames segmented, "
          f"{time.time() - t0:.1f}s")
    del segmenter

    masks, stats = arm_mask.postprocess(masks)
    for idx, mask in enumerate(masks):
        if mask is None:
            continue
        cv2.imwrite(str(mask_dir / f"{idx:06d}.png"), mask.astype(np.uint8) * 255)

    ratios = [r for r in stats.area_ratio if r > 0]
    print(f"[postproc] interpolated {len(stats.interpolated)} frames, "
          f"replaced {len(stats.replaced)}, still missing {len(stats.missing)}")
    if ratios:
        print(f"[postproc] mask area {min(ratios):.2%}-{max(ratios):.2%} of the frame "
              f"(median {float(np.median(ratios)):.2%})")

    meta = {"video": str(args.video), "n_frames": len(frames), "fps": vmeta["fps"],
            "width": vmeta["width"], "height": vmeta["height"], "prompt": args.prompt}
    (args.out_dir / "stage3_stats.json").write_text(
        json.dumps({"meta": meta, "postprocess": asdict(stats)}, indent=2))
    print(f"[save] {mask_dir} ({sum(m is not None for m in masks)} PNGs), "
          f"{args.out_dir / 'stage3_stats.json'}")


if __name__ == "__main__":
    main()

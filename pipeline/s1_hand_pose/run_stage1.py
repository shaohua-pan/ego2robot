"""Stage 1 entry point: ego video -> hand pose tracks (HDF5) + overlay video.

Example:
    python -m pipeline.s1_hand_pose.run_stage1 \
        --video /root/paddlejob/ego/data/test_videos/egodex_sample.mp4 \
        --out-dir /root/paddlejob/ego/outputs/s1_egodex_sample \
        --max-frames 120 --viz
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import h5py
import numpy as np

from pipeline import video
from pipeline.s1_hand_pose import hand_mask_filter, tracking
from pipeline.s1_hand_pose.tracking import HAND_NAMES, HandTrack
from pipeline.s1_hand_pose.wilor_runner import WiLoRRunner
from pipeline.viz import draw


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ego2Robot stage 1: WiLoR hand pose estimation")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", type=str, default="cuda")
    # Debug-only switches; they do not change the method itself.
    p.add_argument("--max-frames", type=int, default=0, help="debug: 0 means the whole video")
    p.add_argument("--viz", action="store_true", help="debug: write a keypoint overlay video")
    p.add_argument("--skip-hand-mask", action="store_true",
                   help="debug: skip the paper's SAM 3 hand-mask detection filter")
    return p.parse_args()


def draw_overlay(img: np.ndarray, tracks: dict[int, HandTrack], idx: int) -> np.ndarray:
    """Per-frame WiLoR detections, before any temporal refinement.

    ``pipeline.viz.stage1`` renders the refined result; this one only needs the raw
    tracks, so it stays here and runs straight after inference.
    """
    out = img.copy()
    lines = [f"frame {idx}"]
    for hand, track in tracks.items():
        name = HAND_NAMES[hand]
        if not track.valid[idx]:
            lines.append(f"{name}: no detection")
            continue
        draw.skeleton(out, track.kp2d[idx], draw.HAND_COLORS[name])
        lines.append(f"{name}: score {track.score[idx]:.2f}")
    draw.hud(out, lines)
    return out


def save_h5(path: Path, tracks: dict[int, HandTrack], meta: dict) -> None:
    with h5py.File(path, "w") as f:
        for k, v in meta.items():
            f.attrs[k] = v
        for hand, track in tracks.items():
            g = f.create_group(HAND_NAMES[hand])
            for name in ("valid", "score", "bbox", "kp3d_cam", "kp2d", "cam_t",
                         "global_orient", "hand_pose", "betas"):
                g.create_dataset(name, data=getattr(track, name), compression="gzip")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    vmeta = video.probe(args.video)
    print(f"[video] {args.video.name} {vmeta['width']}x{vmeta['height']} "
          f"{vmeta['fps']:.2f}fps codec={vmeta['codec']} total_frames={vmeta['n_frames']}")

    runner = WiLoRRunner(device=args.device)
    focal = runner.scaled_focal_length(vmeta["width"], vmeta["height"])
    print(f"[model] WiLoR ready, focal={focal:.1f}px (WiLoR camera model)")

    keep_images = args.viz or not args.skip_hand_mask
    frames, images = [], []
    t0 = time.time()
    for i, img in enumerate(video.decode(args.video, args.max_frames)):
        frames.append(runner.process_frame(img, frame_idx=i))
        if keep_images:
            images.append(img)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1} frames done, {(time.time() - t0) / (i + 1):.3f}s/frame")
    n = len(frames)
    if n == 0:
        raise RuntimeError("no frames decoded")
    n_det = sum(len(fr.detections) for fr in frames)
    print(f"[infer] {n} frames, {n_det} hand detections, {time.time() - t0:.1f}s")

    # Paper A.1: discard detections whose projected keypoints miss the SAM 3 hand mask.
    mask_stats = None
    if not args.skip_hand_mask:
        from pipeline.sam3_runner import Sam3VideoSegmenter
        segmenter = Sam3VideoSegmenter(device=args.device)
        mask_stats = hand_mask_filter.filter_detections(
            frames, video.BgrToRgb(images), segmenter)
        print(f"[sam3] hand-mask filter dropped {mask_stats['dropped']}/"
              f"{mask_stats['detections_before']} detections")
        del segmenter

    tracks = tracking.associate(frames)
    stats = {}
    for hand, track in tracks.items():
        before = int(track.valid.sum())
        removed = tracking.jump_filter(track)
        after = int(track.valid.sum())
        stats[HAND_NAMES[hand]] = {"associated": before, "jump_removed": removed,
                                   "valid": after, "coverage": round(after / n, 3)}
        print(f"[track] {HAND_NAMES[hand]}: associated {before}, jump-removed {removed}, "
              f"valid {after}/{n} ({after / n:.1%})")

    meta = {"video": str(args.video), "n_frames": n, "fps": vmeta["fps"],
            "width": vmeta["width"], "height": vmeta["height"],
            "focal_length": focal}
    h5_path = args.out_dir / "hand_pose.h5"
    save_h5(h5_path, tracks, meta)
    (args.out_dir / "stage1_stats.json").write_text(
        json.dumps({"meta": meta, "hand_mask_filter": mask_stats, "tracks": stats}, indent=2))
    print(f"[save] {h5_path}")

    if args.viz:
        viz_path = args.out_dir / "hand_pose_viz.mp4"
        writer = cv2.VideoWriter(str(viz_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 max(vmeta["fps"], 1.0),
                                 (vmeta["width"], vmeta["height"]))
        for i, img in enumerate(images):
            writer.write(draw_overlay(img, tracks, i))
        writer.release()
        print(f"[save] {viz_path}")


if __name__ == "__main__":
    main()

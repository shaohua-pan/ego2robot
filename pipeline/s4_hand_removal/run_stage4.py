"""Stage 4 entry point: remove the human arms with ProPainter (paper section 3.2, A.4).

The paper's settings - fp16, ``neighbor_length=10``, ``ref_stride=10``,
``subvideo_length=80``, ``mask_dilation=4``, 20 RAFT iterations - are ProPainter's
own defaults plus fp16; they are passed explicitly anyway so the run is
self-documenting and independent of upstream default changes.

ProPainter is driven through its own ``inference_propainter.py`` in a subprocess:
the script resolves its weights relative to the working directory and writes its
results next to them, so nothing about its internals has to be replicated here.
Frames are handed over as a PNG directory rather than a video file, because
ProPainter decodes video with ``torchvision.io.read_video``, which cannot read the
AV1 clips, and because a directory guarantees frame-for-frame alignment with the
stage-③ masks.

    python -m pipeline.s4_hand_removal.run_stage4 \
        --video <clip.mp4> --mask-dir <stage3>/arm_mask --out-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from pipeline import config, video

WEIGHT_FILES = ("ProPainter.pth", "raft-things.pth", "recurrent_flow_completion.pth")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ego2Robot stage 4: hand removal")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--mask-dir", type=Path, required=True, help="stage-3 arm_mask directory")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--max-frames", type=int, default=0, help="debug: 0 means the whole video")
    p.add_argument("--resize-ratio", type=float, default=1.0,
                   help="ProPainter processing scale; below 1.0 trades fidelity for VRAM")
    p.add_argument("--repo", type=Path, default=config.PROPAINTER_ROOT)
    p.add_argument("--weights", type=Path, default=config.PROPAINTER_CKPT_DIR)
    return p.parse_args()


def link_weights(repo: Path, weights: Path) -> None:
    """ProPainter downloads into ``<cwd>/weights``; point that at our copies."""
    target = repo / "weights"
    target.mkdir(parents=True, exist_ok=True)
    for name in WEIGHT_FILES:
        src, dst = weights / name, target / name
        if not src.exists():
            raise SystemExit(f"missing ProPainter weight: {src}")
        if not dst.exists():
            dst.symlink_to(src)


def export_frames(clip: Path, out_dir: Path, max_frames: int) -> list[Path]:
    """Decode with PyAV into the 6-digit PNG layout the masks already use."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for idx, frame in enumerate(video.decode(clip, max_frames)):
        path = out_dir / f"{idx:06d}.png"
        if not path.exists():
            cv2.imwrite(str(path), frame)
        paths.append(path)
    return paths


def run_propainter(repo: Path, source: Path, mask_dir: Path, results: Path,
                   resize_ratio: float) -> Path:
    """Invoke ProPainter's CLI; returns the directory holding its output frames."""
    cmd = [sys.executable, "inference_propainter.py",
           "--video", str(source), "--mask", str(mask_dir), "--output", str(results),
           "--mask_dilation", str(config.PROPAINTER_MASK_DILATION),
           "--ref_stride", str(config.PROPAINTER_REF_STRIDE),
           "--neighbor_length", str(config.PROPAINTER_NEIGHBOR_LENGTH),
           "--subvideo_length", str(config.PROPAINTER_SUBVIDEO_LENGTH),
           "--raft_iter", str(config.PROPAINTER_RAFT_ITER),
           "--resize_ratio", str(resize_ratio), "--save_frames"]
    if config.PROPAINTER_FP16:
        cmd.append("--fp16")
    print("[propainter] " + " ".join(cmd[1:]))
    subprocess.run(cmd, cwd=repo, check=True)
    return results / source.name / "frames"


def dilated_mask(mask_path: Path, shape: tuple[int, int]) -> np.ndarray:
    """The mask ProPainter actually inpaints: ours dilated by ``mask_dilation``.

    Matches ``inference_propainter.read_mask``, which dilates with
    ``scipy.ndimage.binary_dilation(iterations=mask_dilation)``.
    """
    from scipy import ndimage

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros(shape, dtype=bool)
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return ndimage.binary_dilation(mask > 127, iterations=config.PROPAINTER_MASK_DILATION)


def collect(frames_dir: Path, source: list[Path], mask_dir: Path, out_dir: Path) -> list[Path]:
    """Take ProPainter's pixels inside the mask and the original ones outside.

    ProPainter processes (and returns) the whole frame at ``resize_ratio`` scale, so
    at 1080p its output is an upsampled 540p image everywhere, not just where it
    inpainted. Compositing against the original keeps the untouched background at
    full resolution, which is what step ⑥ renders the robot into. It also makes the
    "outside the mask nothing changed" check below exact rather than approximate.
    ProPainter writes 4-digit names; the pipeline's convention is 6 digits.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    produced = sorted(frames_dir.glob("*.png"))
    if len(produced) != len(source):
        raise SystemExit(f"ProPainter produced {len(produced)} frames, expected {len(source)}")
    paths = []
    for idx, (src_path, painted_path) in enumerate(zip(source, produced)):
        original = cv2.imread(str(src_path))
        painted = cv2.imread(str(painted_path))
        if painted.shape != original.shape:
            painted = cv2.resize(painted, (original.shape[1], original.shape[0]),
                                 interpolation=cv2.INTER_CUBIC)
        mask = dilated_mask(mask_dir / f"{idx:06d}.png", original.shape[:2])
        composite = original.copy()
        composite[mask] = painted[mask]
        dst = out_dir / f"{idx:06d}.png"
        cv2.imwrite(str(dst), composite)
        paths.append(dst)
    return paths


def compare(source: list[Path], inpainted: list[Path], mask_dir: Path) -> dict:
    """Measure both sides of the mask.

    ``inside`` is taken over the stage-③ mask (where the arm was) and shows how much
    the frame actually changed. ``outside`` is taken over the complement of the
    *dilated* mask - the only pixels ``collect`` is allowed to overwrite - so it must
    come out exactly 0 unless the composite logic or the mask indexing is wrong.
    """
    inside, outside, untouched = [], [], 0
    for idx, (src_path, out_path) in enumerate(zip(source, inpainted)):
        mask_path = mask_dir / f"{idx:06d}.png"
        if not mask_path.exists():
            continue
        src = cv2.imread(str(src_path)).astype(np.int16)
        out = cv2.imread(str(out_path)).astype(np.int16)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127
        diff = np.abs(src - out).mean(axis=2)
        inside.append(float(diff[mask].mean()) if mask.any() else 0.0)
        outside.append(float(diff[~dilated_mask(mask_path, src.shape[:2])].max()))
        untouched += int(mask.any() and diff[mask].max() == 0)
    return {
        "frames_compared": len(inside),
        "mean_abs_diff_inside_mask": round(float(np.mean(inside)), 3),
        "max_abs_diff_outside_dilated_mask": round(float(np.max(outside)), 5),
        "frames_unchanged_inside_mask": untouched,
        "diff_inside_per_frame": [round(v, 3) for v in inside],
    }


def main() -> None:
    args = parse_args()
    link_weights(args.repo, args.weights)
    vmeta = video.probe(args.video)

    source_dir = args.out_dir / "source"
    t0 = time.time()
    source = export_frames(args.video, source_dir, args.max_frames)
    print(f"[stage4] {len(source)} source frames -> {source_dir}")

    frames_dir = run_propainter(args.repo, source_dir, args.mask_dir,
                                args.out_dir / "propainter", args.resize_ratio)
    inpainted = collect(frames_dir, source, args.mask_dir, args.out_dir / "inpainted")
    stats = compare(source, inpainted, args.mask_dir)
    print(f"[stage4] {len(inpainted)} frames inpainted in {time.time() - t0:.1f}s; "
          f"mean |diff| inside mask {stats['mean_abs_diff_inside_mask']}, "
          f"max outside dilated mask {stats['max_abs_diff_outside_dilated_mask']}")

    meta = {"video": str(args.video), "n_frames": len(source), "fps": vmeta["fps"],
            "width": vmeta["width"], "height": vmeta["height"],
            "mask_dir": str(args.mask_dir), "resize_ratio": args.resize_ratio,
            "settings": {"fp16": config.PROPAINTER_FP16,
                         "neighbor_length": config.PROPAINTER_NEIGHBOR_LENGTH,
                         "ref_stride": config.PROPAINTER_REF_STRIDE,
                         "subvideo_length": config.PROPAINTER_SUBVIDEO_LENGTH,
                         "mask_dilation": config.PROPAINTER_MASK_DILATION,
                         "raft_iter": config.PROPAINTER_RAFT_ITER}}
    (args.out_dir / "stage4_stats.json").write_text(
        json.dumps({"meta": meta, "comparison": stats}, indent=2))
    print(f"[save] {args.out_dir / 'inpainted'}, {args.out_dir / 'stage4_stats.json'}")


if __name__ == "__main__":
    main()

"""Stage 6 entry point: depth-aware compositing (paper section 3.2, eq. 5 / A.4 eq. 9).

    MUJOCO_GL=osmesa python -m pipeline.s6_composite.run_stage6 \
        --stage5 <dir>/robot_panda.h5 --inpainted <stage4>/inpainted \
        --mask-dir <stage3>/arm_mask --depth <vipe>/depth/<seq>.zip --out-dir <dir>

A.4 splits the robot in two: the arm body is always drawn, because it is between the
camera and the workspace and nothing in the scene can be in front of it, while every
gripper pixel is depth-tested against the scene and hidden where the scene is nearer -
unless it falls inside the dilated hand mask, which is where the inpainting removed
the real arm and there is nothing valid to occlude with.

``D_scene`` comes from the VIPE run that stage ① already needed for the camera, not
from Depth Anything V3 as A.4 says: VIPE's depth is metric and shares the scale the
hand trajectory was solved in, so it can be compared with MuJoCo's depth directly.
"""
from __future__ import annotations

import argparse
import json
import time
import zipfile
from pathlib import Path

import cv2
import h5py
import numpy as np

from pipeline import config, robots
from pipeline.s6_composite.scene import Render, Scene


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ego2Robot stage 6: depth-aware compositing")
    p.add_argument("--stage5", type=Path, required=True, help="stage-5 robot_<name>.h5")
    p.add_argument("--inpainted", type=Path, required=True, help="stage-4 inpainted/ frames")
    p.add_argument("--mask-dir", type=Path, required=True, help="stage-3 arm_mask/ frames")
    p.add_argument("--depth", type=Path, required=True, help="VIPE metric depth zip (EXR)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--max-frames", type=int, default=0)
    return p.parse_args()


def read_depth(archive: Path) -> list[np.ndarray]:
    """VIPE writes metric depth as a zip of single-channel half-float EXR files."""
    import OpenEXR

    frames = []
    with zipfile.ZipFile(archive) as zf:
        for name in sorted(zf.namelist()):
            with zf.open(name) as handle:
                exr = OpenEXR.InputFile(handle)
                window = exr.header()["dataWindow"]
                width = window.max.x - window.min.x + 1
                height = window.max.y - window.min.y + 1
                data = np.frombuffer(exr.channels(["Z"])[0], dtype=np.float16)
            frames.append(data.reshape(height, width).astype(np.float32))
    return frames


def hand_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    """Stage-③ arm mask, dilated by A.4's 5x5 kernel for one iteration."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros(shape, dtype=bool)
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    kernel = np.ones((config.COMPOSITE_HAND_DILATE_KERNEL,) * 2, np.uint8)
    return cv2.dilate((mask > 127).astype(np.uint8), kernel,
                      iterations=config.COMPOSITE_HAND_DILATE_ITERATIONS).astype(bool)


def composite(render: Render, background: np.ndarray, scene_depth: np.ndarray,
              dilated_hand: np.ndarray) -> tuple[np.ndarray, dict]:
    """Equation (9). Returns the frame and the per-frame pixel counts."""
    if scene_depth.shape != background.shape[:2]:
        scene_depth = cv2.resize(scene_depth, (background.shape[1], background.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
    occluded = render.gripper & (scene_depth < render.depth) & ~dilated_hand
    visible = render.arm | (render.gripper & ~occluded)
    out = background.copy()
    out[visible] = render.rgb[visible]
    return out, {"arm_px": int(render.arm.sum()), "gripper_px": int(render.gripper.sum()),
                 "gripper_occluded_px": int(occluded.sum()),
                 "robot_px": int(visible.sum())}


def camera_check(scene: Scene, render: Render, hand: str) -> tuple[float, float]:
    """Camera check: the rendered depth must equal the geometry's depth on the same ray.

    Measured at the gripper pixel closest to the projected TCP, so the pixel is known to
    be gripper geometry, and the ray query is independent of the rasteriser. If the field
    of view, the principal point or the camera pose disagreed with the intrinsics the
    trajectory was estimated with, the arm would still render, just in the wrong place.
    The distance to that pixel is returned as well: the TCP is a free-space point between
    the pads, so it usually projects into the gap between them rather than onto metal, but
    it has to land within a few pixels of the jaws.
    """
    tcp = scene.tcp_camera(hand)
    u, v = scene.project(tcp)
    ys, xs = np.nonzero(render.gripper)
    if len(xs) == 0:
        return float("nan"), float("nan")
    squared = (xs - u) ** 2 + (ys - v) ** 2
    nearest = int(np.argmin(squared))
    x, y = int(xs[nearest]), int(ys[nearest])
    error = abs(float(render.depth[y, x]) - scene.ray_depth(x, y)) * 1000.0
    return error, float(np.sqrt(squared[nearest]))


def main() -> None:
    args = parse_args()
    with h5py.File(args.stage5, "r") as h5:
        name = h5.attrs["robot"]
        intrins = np.asarray(h5.attrs["intrins"], dtype=float)
        cam_rot, cam_trans = h5["cam_R"][:], h5["cam_t"][:]
        hands = [key for key in h5 if key not in ("cam_R", "cam_t")]
        bases = {hand: (h5[hand].attrs["base_position_world"],
                        h5[hand].attrs["base_rotation_world"]) for hand in hands}
        qpos = {hand: h5[hand]["qpos"][:] for hand in hands}
        feasible = {hand: h5[hand]["feasible"][:] for hand in hands}

    frames = sorted(args.inpainted.glob("*.png"))
    depths = read_depth(args.depth)
    n = min(len(frames), len(depths), len(cam_rot), *(len(q) for q in qpos.values()))
    if args.max_frames:
        n = min(n, args.max_frames)
    first = cv2.imread(str(frames[0]))
    height, width = first.shape[:2]

    robot = robots.load(name)
    scene = Scene(robot, hands, intrins, (width, height))
    out_dir = args.out_dir / "composited"
    out_dir.mkdir(parents=True, exist_ok=True)

    start, counts = time.time(), []
    depth_errors = {hand: [] for hand in qpos}
    tcp_gap = {hand: [] for hand in qpos}
    for i in range(n):
        # The base stands still in the world; the camera moves, so its pose in the
        # camera frame is x_cam = R_i x_world + t_i for this frame.
        scene.place({hand: (cam_rot[i] @ position + cam_trans[i], cam_rot[i] @ rotation)
                     for hand, (position, rotation) in bases.items()})
        scene.pose({hand: values[i] for hand, values in qpos.items()})
        render = scene.render()
        background = cv2.imread(str(frames[i]))
        frame, stats = composite(render, background, depths[i],
                                 hand_mask(args.mask_dir / f"{i:06d}.png", (height, width)))
        cv2.imwrite(str(out_dir / f"{i:06d}.png"), frame)
        counts.append(stats)
        for hand in qpos:
            error, gap = camera_check(scene, render, hand)
            depth_errors[hand].append(error)
            tcp_gap[hand].append(gap)
    elapsed = time.time() - start

    pixels = float(width * height)
    stats = {
        "robot": name, "n_frames": n, "seconds": round(elapsed, 1),
        "seconds_per_frame": round(elapsed / max(n, 1), 3),
        "intrins": intrins.round(3).tolist(),
        "hand_dilate": {"kernel": config.COMPOSITE_HAND_DILATE_KERNEL,
                        "iterations": config.COMPOSITE_HAND_DILATE_ITERATIONS},
        "coverage": {
            "robot_percent_mean": round(float(np.mean([c["robot_px"] for c in counts])) / pixels * 100, 2),
            "gripper_percent_mean": round(float(np.mean([c["gripper_px"] for c in counts])) / pixels * 100, 3),
            "gripper_occluded_percent_of_gripper": round(
                100.0 * sum(c["gripper_occluded_px"] for c in counts)
                / max(sum(c["gripper_px"] for c in counts), 1), 2)},
        "camera_check": {
            hand: {"depth_error_mm_median": round(float(np.nanmedian(values)), 4),
                   "depth_error_mm_max": round(float(np.nanmax(values)), 4),
                   "tcp_to_jaws_px_median": round(float(np.nanmedian(tcp_gap[hand])), 1),
                   "tcp_to_jaws_px_max": round(float(np.nanmax(tcp_gap[hand])), 1)}
            for hand, values in depth_errors.items()},
        "ik_feasible_frames": {hand: int(mask[:n].sum()) for hand, mask in feasible.items()},
        "per_frame": {"robot_px": [c["robot_px"] for c in counts],
                      "gripper_occluded_px": [c["gripper_occluded_px"] for c in counts]},
        "source": {"stage5": str(args.stage5), "inpainted": str(args.inpainted),
                   "mask_dir": str(args.mask_dir), "depth": str(args.depth),
                   "width": width, "height": height},
    }
    (args.out_dir / f"stage6_{name}_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[stage6] {n} frames in {elapsed:.0f}s ({stats['seconds_per_frame']}s/frame); "
          f"robot covers {stats['coverage']['robot_percent_mean']}% of the frame, "
          f"{stats['coverage']['gripper_occluded_percent_of_gripper']}% of gripper pixels occluded")
    print(f"[stage6] camera check: "
          + ", ".join(f"{h} {v['depth_error_mm_median']}mm at "
                      f"{v['tcp_to_jaws_px_median']}px from the TCP"
                      for h, v in stats["camera_check"].items()))
    print(f"[save] {out_dir}, {args.out_dir / f'stage6_{name}_stats.json'}")


if __name__ == "__main__":
    main()

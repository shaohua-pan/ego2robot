"""Feed WiLoR tracks into Dyn-HaMR's temporal optimization (paper appendix A.1).

The paper's Path B uses WiLoR for per-frame reconstruction and DynHaMR only for
temporal refinement, so Dyn-HaMR's own preprocessing (HaMeR + ViTPose) is
bypassed. Dyn-HaMR reads per-frame files whose presence also encodes track
visibility (``data/dataset.py`` builds its visibility mask from file existence),
so this writes exactly the files ``preproc/export_hamer.py`` would have produced:

    <root>/images/<seq>/000001.jpg                     1-indexed, as split_frame()
    <root>/dynhamr/track_preds/<seq>/00{0,1}/000001_keypoints.json
    <root>/dynhamr/track_preds/<seq>/00{0,1}/000001_mano.json
    <root>/dynhamr/shot_idcs/<seq>.json

Track ids are the hand ids: Dyn-HaMR asserts ``is_right == tid``, so left = 000
and right = 001. MANO parameters stay in WiLoR's right-hand space; the ``is_right``
flag tells Dyn-HaMR when to mirror, matching HaMeR's convention.

Camera translation: WiLoR's ``cam_crop_to_full`` derives tx/ty independently of
the focal length and tz proportional to it. Its default focal is the heuristic
``5000 / 256 * max(W, H)``, which for a 1920x1080 clip is 37500 px against a true
value of roughly 1400 px, so the root depth comes out ~25x too large. Passing
``--focal-length`` (e.g. fx from VIPE's intrinsics) rescales tz accordingly, which
is exactly what ``cam_crop_to_full`` would have returned with that focal.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from pipeline import config
from pipeline.s1_hand_pose.tracking import HAND_NAMES

JPEG_QUALITY = 95


def frame_name(idx0: int) -> str:
    """Dyn-HaMR extracts frames 1-indexed with 6 digits (``preproc/extract_frames.py``)."""
    return f"{idx0 + 1:06d}"


def write_images(video: Path, out_dir: Path, n_frames: int) -> list[str]:
    """Decode the clip with PyAV and write the frames Dyn-HaMR will read."""
    out_dir.mkdir(parents=True, exist_ok=True)
    names = []
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for i, frame in enumerate(container.decode(stream)):
            if i >= n_frames:
                break
            name = frame_name(i)
            path = out_dir / f"{name}.jpg"
            if not path.exists():
                cv2.imwrite(str(path), frame.to_ndarray(format="bgr24"),
                            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            names.append(name)
    return names


def write_h264_copy(src: Path, dst: Path, n_frames: int, fps: float) -> None:
    """Transcode the clip to H.264 for the tools that cannot decode it directly.

    VIPE (and Dyn-HaMR's own frame extraction) read videos through OpenCV/ffmpeg
    builds without an AV1 decoder, which is what the EgoDex clips use; PyAV ships
    libdav1d, so it decodes the source here and re-encodes with libx264. Frame
    count and order are preserved so VIPE's per-frame cameras stay aligned with
    the stage-1 frame indices.
    """
    with av.open(str(src)) as inp, av.open(str(dst), mode="w") as out:
        in_stream = inp.streams.video[0]
        in_stream.thread_type = "AUTO"
        out_stream = out.add_stream("libx264", rate=round(fps))
        out_stream.width = in_stream.codec_context.width
        out_stream.height = in_stream.codec_context.height
        out_stream.pix_fmt = "yuv420p"
        out_stream.options = {"crf": "18"}
        for i, frame in enumerate(inp.decode(in_stream)):
            if i >= n_frames:
                break
            out.mux(out_stream.encode(av.VideoFrame.from_ndarray(
                frame.to_ndarray(format="rgb24"), format="rgb24")))
        out.mux(out_stream.encode())


def write_track(track_dir: Path, names: list[str], group: h5py.Group,
                tid: int, focal_scale: float) -> int:
    """Write per-frame keypoint + MANO JSONs for one hand; returns frames written."""
    track_dir.mkdir(parents=True, exist_ok=True)
    valid = group["valid"][:]
    kp2d = group["kp2d"][:]
    cam_t = group["cam_t"][:]
    global_orient = group["global_orient"][:]
    hand_pose = group["hand_pose"][:]
    betas = group["betas"][:]

    written = 0
    for i, name in enumerate(names):
        if i >= len(valid) or not valid[i]:
            continue
        # OpenPose-style 2D keypoints: [x, y, conf] x 21, flattened.
        kp = np.concatenate([kp2d[i], np.ones((kp2d.shape[1], 1))], axis=1)
        (track_dir / f"{name}_keypoints.json").write_text(json.dumps(
            {"people": [{"pose_keypoints_2d": kp.reshape(-1).tolist()}]}))

        trans = cam_t[i].astype(np.float64).copy()
        trans[2] *= focal_scale
        (track_dir / f"{name}_mano.json").write_text(json.dumps({
            "betas": betas[i].astype(np.float64).tolist(),
            "body_pose": Rotation.from_matrix(hand_pose[i]).as_rotvec().tolist(),
            "global_orient": Rotation.from_matrix(
                global_orient[i].reshape(3, 3)).as_rotvec().tolist(),
            "cam_trans": trans.tolist(),
            "is_right": tid,
        }))
        written += 1
    return written


DATA_CONF_TEMPLATE = """# Generated by pipeline/s1_hand_pose/dynhamr_bridge.py - do not edit by hand.
type: video
split: custom
root: {root}
video_dir: videos
seq: {seq}
ext: mp4
src_path: ${{data.root}}/${{data.video_dir}}/${{data.seq}}.${{data.ext}}
frame_opts:
  ext: jpg
  fps: {fps}
  start_sec: 0
  end_sec: -1
use_cams: True
track_ids: "all"
shot_idx: 0
start_idx: 0
end_idx: -1
split_cameras: True
name: ${{data.seq}}-${{data.track_ids}}-shot-${{data.shot_idx}}-${{data.start_idx}}-${{data.end_idx}}
sources:
  images: ${{data.root}}/images/${{data.seq}}
  cameras: ${{data.root}}/dynhamr/cameras/${{data.seq}}/shot-${{data.shot_idx}}
  tracks: ${{data.root}}/dynhamr/track_preds/${{data.seq}}
  shots: ${{data.root}}/dynhamr/shot_idcs/${{data.seq}}.json

# Use VIPE cameras (intrinsics + poses) instead of DROID-SLAM.
use_vipe: True
vipe_dir: {vipe_dir}
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export stage-1 tracks for Dyn-HaMR")
    p.add_argument("--h5", type=Path, required=True, help="stage-1 hand_pose.h5")
    p.add_argument("--root", type=Path, required=True, help="Dyn-HaMR data root")
    p.add_argument("--seq", type=str, default="", help="sequence name (default: video stem)")
    p.add_argument("--focal-length", type=float, default=0.0,
                   help="true focal length in px; rescales cam_trans z")
    p.add_argument("--vipe-intrinsics", type=Path, default=None,
                   help="VIPE intrinsics npz; its median fx is used as focal length")
    p.add_argument("--conf-name", type=str, default="ego2robot",
                   help="name of the hydra data config written into Dyn-HaMR")
    return p.parse_args()


def resolve_focal(args: argparse.Namespace) -> float:
    """Focal length in pixels: explicit flag wins, otherwise VIPE's median fx."""
    if args.focal_length > 0:
        return args.focal_length
    if args.vipe_intrinsics is None:
        return 0.0
    data = np.load(args.vipe_intrinsics)["data"]  # (N, 4) fx, fy, cx, cy
    focal = float(np.median(data[:, 0]))
    print(f"[bridge] VIPE median fx = {focal:.1f}px from {args.vipe_intrinsics}")
    return focal


def main() -> None:
    args = parse_args()
    focal_true = resolve_focal(args)
    with h5py.File(args.h5, "r") as f:
        meta = dict(f.attrs)
        video = Path(meta["video"])
        seq = args.seq or video.stem
        n_frames = int(meta["n_frames"])
        wilor_focal = float(meta["focal_length"])
        focal_scale = focal_true / wilor_focal if focal_true > 0 else 1.0

        video_dir = args.root / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        copy = video_dir / f"{seq}.mp4"
        if not copy.exists():
            write_h264_copy(video, copy, n_frames, float(meta["fps"]))
            print(f"[bridge] H.264 copy -> {copy}")

        names = write_images(video, args.root / "images" / seq, n_frames)
        print(f"[bridge] {len(names)} frames -> {args.root / 'images' / seq}")

        track_root = args.root / "dynhamr" / "track_preds" / seq
        for hand, name in HAND_NAMES.items():
            if name not in f:
                continue
            written = write_track(track_root / f"{hand:03d}", names, f[name], hand, focal_scale)
            print(f"[bridge] {name} (tid {hand:03d}): {written} frames")

    shot_path = args.root / "dynhamr" / "shot_idcs" / f"{seq}.json"
    shot_path.parent.mkdir(parents=True, exist_ok=True)
    shot_path.write_text(json.dumps({f"{n}.jpg": 0 for n in names}, indent=1))

    conf = config.DYNHAMR_ROOT / "dyn-hamr" / "confs" / "data" / f"{args.conf_name}.yaml"
    conf.write_text(DATA_CONF_TEMPLATE.format(
        root=args.root, seq=seq, fps=int(round(float(meta["fps"]))),
        vipe_dir=config.VIPE_ROOT / "vipe_results"))
    print(f"[bridge] wrote {conf}")
    if focal_scale != 1.0:
        print(f"[bridge] cam_trans z scaled by {focal_scale:.4f} "
              f"({wilor_focal:.0f}px -> {focal_true:.0f}px)")
    print("\nnext steps:")
    print(f"  conda run -n vipe vipe infer {copy}")
    print(f"  cd {config.DYNHAMR_ROOT / 'dyn-hamr'} && conda run -n dynhamr "
          f"python run_opt.py data={args.conf_name} data.seq={seq} is_static=False")


if __name__ == "__main__":
    main()


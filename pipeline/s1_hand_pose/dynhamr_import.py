"""Read Dyn-HaMR's temporal optimization results back into stage-1 HDF5 (paper A.1).

``dynhamr_bridge.py`` writes the export side; this is the import side. Dyn-HaMR
saves a checkpoint per optimization step under its hydra run directory::

    <run_dir>/<stage>/<seq>_<step:06d>_world_results.npz
    <run_dir>/track_info.json                 per-track visibility used by the optimizer

The npz holds the optimized MANO parameters in the world frame (``trans``,
``root_orient``, ``pose_body``, ``betas``) plus the camera trajectory
(``cam_R``, ``cam_t``, already multiplied by the optimized ``world_scale``) and
the shared intrinsics ``intrins = [fx, fy, cx, cy]``.

Rather than re-deriving MANO forward kinematics, this module calls Dyn-HaMR's own
``body_model.run_mano`` so the joints match what the optimizer was scoring - in
particular the left hand is evaluated with the right-hand model and mirrored on
x afterwards, exactly as HaMeR/WiLoR do.

The refined poses are written back as a ``refined`` subgroup next to the raw
per-frame WiLoR arrays, keeping stage 2 a pure consumer of one HDF5 file. Run
this with the ``dynhamr`` interpreter, which owns MANO and torch:

    /root/anaconda3/envs/dynhamr/bin/python -m pipeline.s1_hand_pose.dynhamr_import \
        --run-dir <hydra run dir> --h5 <stage-1 hand_pose.h5>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

from pipeline import config, geometry
from pipeline.s1_hand_pose.tracking import HAND_NAMES

STAGES = ("smooth_fit", "root_fit", "init")


def latest_result(run_dir: Path, stage: str) -> Path:
    """Newest checkpoint of ``stage``; Dyn-HaMR numbers them by optimization step."""
    files = sorted(
        (run_dir / stage).glob("*_world_results.npz"),
        key=lambda p: int(p.stem.split("_")[-3]),
    )
    if not files:
        raise FileNotFoundError(f"no results in {run_dir / stage}")
    return files[-1]


def step_of(path: Path) -> int:
    return int(path.stem.split("_")[-3])


def world_joints(res: dict, mano_dir: Path, device: str):
    """MANO forward kinematics with Dyn-HaMR's own wrapper; returns (B, T, 21, 3)."""
    sys.path.insert(0, str(config.DYNHAMR_ROOT / "dyn-hamr"))
    import torch
    from body_model import MANO, run_mano

    def t(name):
        return torch.as_tensor(res[name], dtype=torch.float32, device=device)

    trans, root_orient = t("trans"), t("root_orient")
    pose_body, betas, is_right = t("pose_body"), t("betas"), t("is_right")
    B, T = trans.shape[:2]
    model = MANO(model_path=str(mano_dir), gender="neutral", num_hand_joints=15,
                 mean_params=str(mano_dir.parent / "mano_mean_params.npz"),
                 create_body_pose=False, batch_size=B * T, pose2rot=True).to(device)
    with torch.no_grad():
        out = run_mano(model, trans, root_orient,
                       pose_body.reshape(B, T, -1), is_right, betas=betas)
    return out["joints"].cpu().numpy()


def to_camera(joints_world: np.ndarray, cam_R: np.ndarray, cam_t: np.ndarray) -> np.ndarray:
    """World -> camera. Dyn-HaMR stores cam_R/cam_t as the world-to-camera transform."""
    return geometry.points_to_camera(joints_world, cam_R, cam_t)


def vis_masks(run_dir: Path, n_tracks: int, n_frames: int) -> np.ndarray:
    """Per-track visibility the optimizer used, from ``track_info.json``."""
    info = json.loads((run_dir / "track_info.json").read_text())["tracks"]
    mask = np.zeros((n_tracks, n_frames), dtype=bool)
    for track in info.values():
        mask[track["index"]] = np.asarray(track["vis_mask"], dtype=bool)[:n_frames]
    return mask


def write_refined(h5_path: Path, res: dict, joints_world: np.ndarray,
                  joints_cam: np.ndarray, valid: np.ndarray, provenance: dict) -> dict:
    """Store one ``refined`` subgroup per hand; returns a per-hand summary."""
    summary = {}
    with h5py.File(h5_path, "a") as f:
        for row, hand in enumerate(res["is_right"][:, 0].astype(int)):
            name = HAND_NAMES[hand]
            group = f.require_group(name)
            if "refined" in group:
                del group["refined"]
            g = group.create_group("refined")
            g.attrs.update(provenance)
            for key, value in (("trans", res["trans"][row]),
                               ("root_orient", res["root_orient"][row]),
                               ("hand_pose", res["pose_body"][row]),
                               ("betas", res["betas"][row]),
                               ("kp3d_world", joints_world[row]),
                               ("kp3d_cam", joints_cam[row]),
                               ("cam_R", res["cam_R"][row]),
                               ("cam_t", res["cam_t"][row]),
                               ("valid", valid[row])):
                g.create_dataset(key, data=value, compression="gzip")
            g.create_dataset("intrins", data=res["intrins"])
            g.attrs["world_scale"] = float(res["world_scale"].reshape(-1)[0])

            depth = joints_cam[row, valid[row], 0, 2]
            err = np.linalg.norm(
                geometry.project(joints_cam[row], res["intrins"])
                - f[name]["kp2d"][: joints_cam.shape[1]], axis=-1)[valid[row]]
            summary[name] = {
                "frames": int(valid[row].sum()),
                "wrist_depth_m": [round(float(depth.min()), 4),
                                  round(float(depth.max()), 4),
                                  round(float(np.median(depth)), 4)],
                "reproj_px_median": round(float(np.median(err)), 2),
                "reproj_px_p90": round(float(np.percentile(err, 90)), 2),
            }
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Import Dyn-HaMR results into stage-1 HDF5")
    p.add_argument("--run-dir", type=Path, required=True,
                   help="Dyn-HaMR hydra run directory (outputs/logs/<type>-<split>/<date>/<name>)")
    p.add_argument("--h5", type=Path, required=True, help="stage-1 hand_pose.h5 to update")
    p.add_argument("--stage", type=str, default="", choices=("",) + STAGES,
                   help="which optimization stage to import (default: latest available)")
    p.add_argument("--mano-dir", type=Path, default=None,
                   help="MANO model dir (default: Dyn-HaMR's _DATA/data/mano)")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    stages = [args.stage] if args.stage else [s for s in STAGES if (args.run_dir / s).is_dir()]
    if not stages:
        raise FileNotFoundError(f"no optimization stage directories under {args.run_dir}")
    npz_path = latest_result(args.run_dir, stages[0])
    res = dict(np.load(npz_path))
    print(f"[import] {npz_path.parent.name} step {step_of(npz_path)}: "
          f"{res['trans'].shape[0]} tracks x {res['trans'].shape[1]} frames")

    mano_dir = args.mano_dir or config.DYNHAMR_ROOT / "_DATA" / "data" / "mano"
    joints_world = world_joints(res, mano_dir, args.device)
    joints_cam = to_camera(joints_world, res["cam_R"], res["cam_t"])
    valid = vis_masks(args.run_dir, *res["trans"].shape[:2])

    provenance = {"source": str(npz_path), "stage": stages[0], "step": step_of(npz_path)}
    summary = write_refined(args.h5, res, joints_world, joints_cam, valid, provenance)
    for name, stat in summary.items():
        lo, hi, med = stat["wrist_depth_m"]
        print(f"[import] {name}: {stat['frames']} frames, wrist depth {lo}-{hi}m "
              f"(median {med}), reprojection {stat['reproj_px_median']}px median / "
              f"{stat['reproj_px_p90']}px p90")
    out = args.h5.parent / "dynhamr_import.json"
    out.write_text(json.dumps({"provenance": provenance, "hands": summary}, indent=2))
    print(f"[save] {args.h5} (refined groups), {out}")


if __name__ == "__main__":
    main()

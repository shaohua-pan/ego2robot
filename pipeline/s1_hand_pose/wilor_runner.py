"""Per-frame WiLoR hand reconstruction (Path B, step 1; paper appendix A.1).

Wraps the official third_party/WiLoR model and detector to produce, for each
frame: left/right label, detection score, bbox, camera-frame 3D keypoints
(21, 3), full-image 2D projections (21, 2) and MANO parameters.

Implementation notes:
- ``load_wilor`` hardcodes the MANO path to ``./mano_data/`` relative to the CWD,
  so the model is loaded while chdir'ed into the WiLoR repo root and
  ``mano_data/MANO_RIGHT.pkl`` must exist as a symlink to our checkpoint store.
- WiLoR uses a heuristic focal length ``FOCAL_LENGTH / IMAGE_SIZE * max(W, H)``,
  which makes the absolute root depth unreliable. The paper resolves this later:
  DynHaMR re-optimizes translation under a [0.05, 0.4] m depth constraint using
  VIPE's estimated intrinsics (see ``dynhamr_bridge.py``).
- Left hands are processed as mirrored right hands internally: the x component
  of the 3D keypoints is flipped back here. MANO parameters are kept in the
  original "right-hand space" for the later DynHaMR refinement stage.
"""
from __future__ import annotations

import contextlib
import os
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch

from pipeline import config


@dataclass
class HandDetection:
    is_right: int              # 1 = right hand, 0 = left hand (YOLO class label)
    score: float               # detection confidence
    bbox: np.ndarray           # (4,) xyxy in full-image pixels
    kp3d_cam: np.ndarray       # (21, 3) camera frame, meters
    kp2d: np.ndarray           # (21, 2) full-image pixels
    cam_t: np.ndarray          # (3,) wrist-root translation in camera frame
    global_orient: np.ndarray  # (1, 3, 3) rotation matrix, right-hand space
    hand_pose: np.ndarray      # (15, 3, 3)
    betas: np.ndarray          # (10,)


@dataclass
class FrameResult:
    frame_idx: int
    detections: list[HandDetection] = field(default_factory=list)
def stub_pyrender() -> str:
    """Replace ``pyrender`` with a mock module before WiLoR is imported.

    ``WiLoR.__init__`` unconditionally builds ``MeshRenderer``, which constructs a
    ``pyrender.OffscreenRenderer``; combined with the ``PYOPENGL_PLATFORM=egl``
    that ``wilor.utils.renderer`` forces at import time, loading the checkpoint
    fails on any host without libEGL. Importing the real pyrender first is not
    enough, because the failure happens when the offscreen context is created.

    This pipeline never renders meshes through pyrender (robot rendering in later
    stages uses MuJoCo); it only needs keypoints and MANO parameters. So we install
    a stub whose attribute access yields mocks, which keeps WiLoR's renderer
    objects inert. Set ``EGO2ROBOT_ENABLE_PYRENDER=1`` to keep the real package
    on machines that do have a working GL stack.

    Returns "stub", "real" or "cached".
    """
    if os.environ.get("EGO2ROBOT_ENABLE_PYRENDER") == "1":
        return "real"
    if isinstance(sys.modules.get("pyrender"), types.ModuleType) and \
            getattr(sys.modules["pyrender"], "_ego2robot_stub", False):
        return "cached"
    stub = types.ModuleType("pyrender")
    stub._ego2robot_stub = True
    stub.__getattr__ = lambda name: MagicMock()  # PEP 562 module-level __getattr__
    sys.modules["pyrender"] = stub
    return "stub"


@contextlib.contextmanager
def _torch_load_full_pickle():
    """Temporarily restore ``torch.load(weights_only=False)`` semantics.

    PyTorch 2.6 flipped the ``weights_only`` default to True, while ultralytics
    8.1.34 (the version WiLoR pins) calls ``torch.load`` without the argument, so
    loading detector.pt fails on ``GLOBAL ultralytics.nn.tasks.PoseModel``.

    ``weights_only=False`` unpickles arbitrary objects and must only be used on
    checkpoints you trust. Scope it as narrowly as possible: here it wraps just the
    load of the official WiLoR detector weights, and the original ``torch.load``
    is restored immediately afterwards.
    """
    original = torch.load

    def patched(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = patched
    try:
        yield
    finally:
        torch.load = original


@contextlib.contextmanager
def _chdir(path: Path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def ensure_mano_symlink() -> None:
    """Symlink the downloaded MANO_RIGHT.pkl into the WiLoR repo's mano_data/."""
    link = config.WILOR_ROOT / "mano_data" / "MANO_RIGHT.pkl"
    if link.exists():
        return
    if not config.MANO_RIGHT_PKL.exists():
        raise FileNotFoundError(
            f"MANO_RIGHT.pkl not found at {config.MANO_RIGHT_PKL}; download it first")
    link.symlink_to(config.MANO_RIGHT_PKL)


class WiLoRRunner:
    # WiLoR's demo default detection threshold; the paper does not override it.
    DET_CONF_THRESH = 0.3
    RESCALE_FACTOR = 2.0

    def __init__(self, device: str = "cuda"):
        if str(config.WILOR_ROOT) not in sys.path:
            sys.path.insert(0, str(config.WILOR_ROOT))
        for f in (config.WILOR_CKPT, config.WILOR_DETECTOR):
            if not f.exists():
                raise FileNotFoundError(f"checkpoint not found: {f}")
        ensure_mano_symlink()
        print(f"[wilor] pyrender backend: {stub_pyrender()}")

        from ultralytics import YOLO
        from wilor.models import load_wilor

        # MANO paths inside load_wilor are CWD-relative, so load from the repo root.
        with _chdir(config.WILOR_ROOT):
            self.model, self.model_cfg = load_wilor(
                checkpoint_path=str(config.WILOR_CKPT),
                cfg_path=str(config.WILOR_ROOT / "pretrained_models" / "model_config.yaml"),
            )
        self.device = torch.device(device)
        self.model = self.model.to(self.device).eval()
        with _torch_load_full_pickle():
            self.detector = YOLO(str(config.WILOR_DETECTOR))

    def scaled_focal_length(self, width: int, height: int) -> float:
        """WiLoR's camera model: FOCAL_LENGTH / IMAGE_SIZE * max(W, H)."""
        cfg = self.model_cfg
        return cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * max(width, height)

    @torch.no_grad()
    def process_frame(self, img_bgr: np.ndarray, frame_idx: int = 0) -> FrameResult:
        from wilor.datasets.vitdet_dataset import ViTDetDataset
        from wilor.utils.renderer import cam_crop_to_full

        H, W = img_bgr.shape[:2]
        result = FrameResult(frame_idx=frame_idx)

        det_out = self.detector(img_bgr, conf=self.DET_CONF_THRESH, device=self.device,
                                verbose=False)[0]
        bboxes, is_right, scores = [], [], []
        for det in det_out.boxes:
            box = det.xyxy.squeeze().cpu().numpy()
            bboxes.append(box[:4])
            is_right.append(float(det.cls.squeeze().item()))
            scores.append(float(det.conf.squeeze().item()))
        if not bboxes:
            return result

        boxes = np.stack(bboxes)
        right = np.array(is_right, dtype=np.float32)

        dataset = ViTDetDataset(self.model_cfg, img_bgr, boxes, right,
                                rescale_factor=self.RESCALE_FACTOR)
        loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset),
                                             shuffle=False, num_workers=0)
        batch = next(iter(loader))
        batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}

        out = self.model(batch)

        multiplier = (2 * batch["right"] - 1)
        pred_cam = out["pred_cam"].float()
        pred_cam[:, 1] = multiplier * pred_cam[:, 1]

        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()
        focal = self.scaled_focal_length(W, H)
        cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size,
                                      focal).cpu().numpy()

        joints3d = out["pred_keypoints_3d"].float().cpu().numpy()  # (B, 21, 3), root-relative
        mano = out["pred_mano_params"]
        g_orient = mano["global_orient"].float().cpu().numpy()
        h_pose = mano["hand_pose"].float().cpu().numpy()
        betas = mano["betas"].float().cpu().numpy()

        for n in range(joints3d.shape[0]):
            r = float(batch["right"][n].item())
            j3d = joints3d[n].copy()
            j3d[:, 0] = (2 * r - 1) * j3d[:, 0]        # undo the left-hand mirroring
            j3d_cam = j3d + cam_t_full[n][None, :]      # absolute camera-frame coords
            # Pinhole projection to the full image; principal point at the image
            # center, matching WiLoR's project_full_img in demo.py.
            z = np.clip(j3d_cam[:, 2], 1e-6, None)
            kp2d = np.stack([
                j3d_cam[:, 0] / z * focal + W / 2.0,
                j3d_cam[:, 1] / z * focal + H / 2.0,
            ], axis=1)
            result.detections.append(HandDetection(
                is_right=int(r),
                score=scores[n],
                bbox=boxes[n].astype(np.float32),
                kp3d_cam=j3d_cam.astype(np.float32),
                kp2d=kp2d.astype(np.float32),
                cam_t=cam_t_full[n].astype(np.float32),
                global_orient=g_orient[n].astype(np.float32),
                hand_pose=h_pose[n].astype(np.float32),
                betas=betas[n].astype(np.float32),
            ))
        return result

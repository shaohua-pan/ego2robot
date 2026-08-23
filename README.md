# Ego2Robot — unofficial reproduction

Reproduction of **Ego2Robot: Scalable Robot Data Synthesis from Egocentric Human Data**
([arXiv:2608.02580](https://arxiv.org/abs/2608.02580)). The authors did not release code,
so every component here is written from the paper body and appendix; all constants are
kept in one place (`pipeline/config.py`) with the appendix section they come from.

Scope: a small-scale, faithful reproduction of the *pipeline*, not of the 18,561-hour
dataset or the 200K-step VLA pretraining runs.

## Status

Pipeline stages follow Figure 1. Path B (raw video in, no hand annotations) is the one
being implemented.

| Stage | Paper | State |
| --- | --- | --- |
| ① Hand pose estimation | §3.1, A.1 | working end to end, see below |
| ② Retargeting + temporal smoothing | §3.1, A.3 | working end to end, see below |
| ③ Arm segmentation | A.4 | working end to end, see below |
| ④ Hand removal (ProPainter) | A.4 | working end to end, see below |
| ⑤ Base pose search + IK | §3.2, A.4 | working for Panda and ARX-L5, see below |
| ⑥ Depth-aware compositing | §3.2, A.4 | working end to end, see below |
| Quality curation L1/L2/L3 | §3.3 | not started |
| Subtask segmentation (VLM) | A.2 | not started |

### Stage ① in detail

| A.1 component | Where |
| --- | --- |
| WiLoR per-frame MANO reconstruction | `pipeline/s1_hand_pose/wilor_runner.py` |
| SAM 3 hand-mask detection filter (>80% keypoints outside → drop) | `hand_mask_filter.py`, `pipeline/sam3_runner.py` |
| Cross-frame association (score seed, bidirectional nearest wrist) | `tracking.py:associate` |
| Jump filter `max(4·median v, 0.003 m/frame)` | `tracking.py:jump_filter` |
| Export to Dyn-HaMR, incl. focal correction of `cam_trans` | `dynhamr_bridge.py` |
| Dyn-HaMR eq. (6) with depth ∈ [0.05, 0.4] m | `scripts/patch_dynhamr.py`, `scripts/run_dynhamr.py` |
| Read refined poses back | `dynhamr_import.py` |
| Gap handling (small: lerp+SLERP, large: home config + blend) | `gap_handling.py` — applied in stage ②, which is where the gripper pose and width exist |

Not yet done in stage ①: SAM 3's optional `kernels` post-processing (NMS, hole fill,
sprinkle removal) stays disabled because it requires `trust_remote_code=True`.

### Stage ② in detail

| Component | Where |
| --- | --- |
| Virtual fingertip, TCP, width, grasp frame (eq. 1-3) | `pipeline/s2_retarget/retarget.py` |
| Degenerate orientation fallback (w < 1 cm, z ∥ d) | `retarget.py:retarget` |
| Gap handling (A.1) on the gripper trajectory | `s1_hand_pose/gap_handling.py` |
| Velocity filter, eq. (7), 2 rounds | `smoothing.py:velocity_filter` |
| Savitzky-Golay on position and width | `smoothing.py:smooth_series` |
| Gaussian-weighted SLERP on orientation | `smoothing.py:gaussian_slerp` |

Retargeting and smoothing run in the **world** frame: the robot base is static there,
while the egocentric camera moves with the head, so smoothing in the camera frame would mix
head motion into the hand trajectory. The paper does not state which frame these steps use;
it does fix the camera frame for base-pose candidates (A.4) and for the action labels the
policy consumes ("camera-frame relative EEF actions"), and it explicitly rejects world-frame
*actions* because ego camera placements are unknown and vary per source. Both frames are
therefore written to the HDF5, related by the per-frame VIPE extrinsics.

Large gaps (> 10 frames) are still left unfilled: the paper fills them with the robot's home
configuration, which only becomes a world-frame pose once step ⑤ has chosen a base pose.

### Stage ③ in detail

| Component | Where |
| --- | --- |
| SAM 3 text prompt "person", middle-frame anchor, 400/50 chunking, OR merge | `pipeline/sam3_runner.py` |
| (i) gaps ≤ 3 frames interpolated | `s3_arm_seg/arm_mask.py:fill_short_gaps` |
| (ii) area < 50% of the local median (window 11) replaced | `arm_mask.py:replace_small_masks` |
| (iii) morphological close, 5×5 ellipse | `arm_mask.py:close_mask` |

Masks are written as one PNG per frame named by zero-based frame index, which is the layout
ProPainter reads in step ④.

"Interpolating neighbouring masks" is not specified further. Blending two binary masks and
thresholding merely switches between them half-way, so `interpolate_masks` interpolates their
signed distance transforms, falling back to the union if the two masks are disjoint (an empty
mask would silently leave the arm in the frame for step ④ to keep).

### Stage ④ in detail

The paper's settings — fp16, `neighbor_length=10`, `ref_stride=10`, `subvideo_length=80`,
`mask_dilation=4`, 20 RAFT iterations — are ProPainter's own defaults plus fp16.
`pipeline/s4_hand_removal/run_stage4.py` passes them explicitly from `config.py` and drives
ProPainter's `inference_propainter.py` in a subprocess, so none of its internals are
duplicated here. Two details are forced by the environment:

- frames are handed over as a PNG directory, because ProPainter reads video files with
  `torchvision.io.read_video`, which cannot decode the AV1 clips, and a directory also
  guarantees frame-for-frame alignment with the stage-③ masks
- ProPainter resolves its weights relative to the working directory, so `link_weights`
  symlinks `$EGO2ROBOT_STORE/checkpoints/propainter/*.pth` into `<repo>/weights/`
- ProPainter returns the *whole* frame at its processing scale, so at `--resize-ratio 0.5` a
  1080p clip would come back as an upsampled 540p image everywhere, not just where it
  inpainted. `collect()` therefore takes ProPainter's pixels only inside the dilated mask
  and the original ones outside it, which keeps the background that step ⑥ renders the
  robot into at full resolution.

`compare()` then checks the invariant that matters: inside the stage-③ mask the frame must
have changed, and outside the *dilated* mask — the only pixels `collect()` may overwrite —
it must be bit-identical to the source.

### Stage ⑤ in detail

A morphology is eleven lines of YAML in `pipeline/robots/` naming bodies and joints; every
number is derived from the MuJoCo model so that adding one of the remaining thirteen needs
no manual measurement:

- the **TCP** is the midpoint of the two jaw pads (the most distal geoms of the jaw bodies,
  averaged so that a multi-box pad does not bias it sideways), with `z` along the gripper
  joint's travel and `x` from the wrist towards the pads — the same construction as eq.(3),
  so a retargeted human grasp frame can be used as an IK target unchanged
- that derivation is **checked against Table 3**: the largest TCP distance from the base over
  the joint limits comes out 1.281 m for Panda (paper 1.272) and 0.854 m for ARX-L5 (paper
  0.855). Leaving the TCP at the flange gives 1.189 m and 0.720 m, so the column really is
  measured to the fingertips, which is what confirms the convention
- gripper opening maps Table 3's stroke affinely onto each finger joint's own range, so the
  closed end of the stroke sits at the closed end of the joint. A plain `w / 2` is only right
  for jaws whose pads travel with the joint 1:1 (Panda, ARX-L5, Piper); ViperX's pads move
  21-57 mm for a 15-87 mm opening, so halving the width would open it 27 mm too far. The test
  checks the measured pad-to-pad *travel* over the stroke, not the absolute gap, because the
  pads are located by geom centres

The search itself follows eq.(4) and eq.(8), with one exact shortcut. The reach penalty
`5|rho_bar - 0.65|` depends only on the base *position*, so a position whose best possible
score `1 - penalty` is already worse than the fifth best score found so far cannot enter the
top five and can be skipped; the same bound stops the keyframe loop inside a candidate early.
Positions are therefore visited in order of increasing penalty and the loop exits when the
bound closes. The result is the same argmax the paper describes: on the sample clip that
means 135-495 candidates scored instead of 11,025, and 3-11 positions visited out of ~118
that survive A.4's pruning.

`pipeline/viz/stage5.py` draws the solved arms as a projected link skeleton rather than a
render, so the placement can be judged before step ⑥ exists — no OpenGL required.

### Stage ⑥ in detail

Stage ⑤ stores each base as a **world** pose, and the render camera stays at the origin of
MuJoCo's world looking down `+z`, so every frame the two arms are moved to
`R_i p_world + t_i` with that frame's VIPE extrinsics. That is what "the robot is rendered
from the original camera viewpoint" means once the head moves: the base is bolted to the
scene and the camera walks around it. MuJoCo's camera looks down its own `-z` with `+y` up
against OpenCV's `+z`/`+y`-down, so the camera carries a 180° turn about `x`, and its `fovy`
is `2 atan(h / 2 f_y)` from the same intrinsics the trajectory was estimated with.

Three rendering passes per frame give what eq.(9) needs: colour, depth, and a segmentation
image that is split into `robot_mask` and `gripper_mask`. The gripper is defined as the
subtree of the spec's `wrist_body`, which is `hand`+fingers on Panda and `link6`–`link8` on
ARX-L5 — that is the end-effector assembly, the part that reaches into the scene and can be
occluded by it. Arm-body pixels are always drawn; gripper pixels lose to nearer scene depth
except inside the dilated hand mask, where the inpainting removed the real arm and there is
nothing valid left to occlude with.

Two implementation notes. Each morphology is attached from a *freshly parsed* spec per hand:
attaching one spec object twice leaves MuJoCo with a contact-exclude list it cannot resolve
(`incompatible id in exclude array`). And the compile prints `Attach conflict … impratio /
integrator / cone`, which is MuJoCo saying the child's solver options are ignored — harmless
here, because the scene is only ever posed with `mj_forward`, never stepped.

Sites are hidden (`MjvOption.sitegroup[:] = 0`). MuJoCo draws them, so the injected TCP
site rendered as a 5 mm grey sphere floating between the jaws — 300 pixels per hand in the
colour frame — and, worse, wrote itself into the depth buffer and appeared in the
segmentation under a *site* id that collides with the geom ids the two masks of eq.(9) are
built from.

`camera_check` is the check that the camera is right: at the gripper pixel nearest the
projected TCP, the rendered depth must equal the depth `mujoco.mj_ray` reports for the same
pixel ray. Rasteriser against collision geometry, so it tests `fovy`, the principal point
and the camera pose together; it comes out at **1–2 µm**. The TCP itself lands 8–30 px away
because it is a free-space point in the jaw gap.

## Layout

```
pipeline/
  config.py              paths + every paper constant, annotated with its appendix section
  geometry.py            projection and world<->camera conversions
  video.py               PyAV decoding (AV1-capable) + a lazy BGR->RGB view
  sam3_runner.py         SAM 3 video segmentation (A.4 chunking, middle-frame anchor)
  s1_hand_pose/
    wilor_runner.py      per-frame WiLoR inference
    hand_mask_filter.py  A.1 detection filter against the SAM 3 hand mask
    tracking.py          A.1 association + jump filter
    run_stage1.py        stage ① entry point -> hand_pose.h5 (+ overlay video)
    dynhamr_bridge.py    hand_pose.h5 -> Dyn-HaMR inputs (export side)
    dynhamr_import.py    Dyn-HaMR results -> hand_pose.h5 `refined` groups (import side)
    gap_handling.py      A.1 gap interpolation / home-configuration blending
  s2_retarget/
    retarget.py          eq. (1)-(3) hand keypoints -> parallel-jaw gripper pose
    smoothing.py         eq. (7) velocity filter, Savitzky-Golay, Gaussian-weighted SLERP
    run_stage2.py        stage ② entry point -> hand_pose.h5 `gripper` groups
  s3_arm_seg/
    arm_mask.py          A.4 post-processing: gap interpolation, area filter, close
    run_stage3.py        stage ③ entry point -> arm_mask/%06d.png + stats
  s4_hand_removal/
    run_stage4.py        stage ④ entry point: ProPainter inpainting -> inpainted/%06d.png
  robots/
    __init__.py          morphology registry: spec loading, derived TCP frame, reach check
    panda.yaml           Franka Panda (Table 3: 7 DOF, 0-80 mm, 1.272 m)
    arx_l5.yaml          ARX L5 (Table 3: 6 DOF, 0-88 mm, 0.855 m)
  s5_base_ik/
    candidates.py        A.4 base grid, reach pruning, eq. (8) scoring
    keyframes.py         eq. (4) keyframe set: farthest-point sampling over pose extremes
    ik.py                mink IK (quadprog, 100 iterations, 1e-5 threshold)
    run_stage5.py        stage ⑤ entry point -> robot_<name>.h5 + stats
  s6_composite/
    scene.py             both arms in one MuJoCo scene + the ego camera; RGB/depth/masks
    run_stage6.py        stage ⑥ entry point: eq. (9) -> composited/%06d.png + stats
  viz/
    draw.py              drawing primitives (skeleton, gripper frame, HUD)
    render.py            time-series strip + the decode/annotate/encode loop
    stage1.py            one video per stage: raw vs refined hand poses + wrist depth
    stage2.py            gripper jaw line, axes and opening width
    stage3.py            arm mask overlay + mask-area trace
    stage4.py            inpainted video with the original inset + difference trace
    stage5.py            projected robot skeleton + per-frame IK error
    stage6.py            the finished video; --annotate adds a HUD and an occlusion trace
scripts/
  setup_envs.sh          the three conda environments
  prepare_bmc.py         generate Dyn-HaMR's BMC biomechanical priors headlessly
  patch_dynhamr.py       set depth range [0.05, 0.4] m and enable the bio loss
  run_dynhamr.py         headless launcher (stubs pyrender / VPoser)
tests/
  test_pipeline.py       closed-form checks (eq. 1-3, blend length, filters); no pytest needed
```

Code only. Checkpoints, datasets, third-party repos and outputs live under
`$EGO2ROBOT_STORE` (default `/root/paddlejob/ego`), which keeps the workspace small.

## Setup

```bash
bash scripts/setup_envs.sh          # creates conda envs: ego2robot, dynhamr, vipe
```

Three environments are required and cannot be merged: WiLoR + SAM 3 need torch 2.6 with
`transformers>=4.58`, Dyn-HaMR is pinned to torch 1.13 / numpy 1.23, and VIPE's vendored
GroundingDINO only works with `transformers` 4.x.

Repositories to place under `$EGO2ROBOT_STORE/third_party` (Dyn-HaMR) or `third_party/`
(WiLoR):

- [rolpotamias/WiLoR](https://github.com/rolpotamias/WiLoR) → `third_party/WiLoR`
- [ZhengdiYu/Dyn-HaMR](https://github.com/ZhengdiYu/Dyn-HaMR) → `$EGO2ROBOT_STORE/third_party/Dyn-HaMR`
  (VIPE ships inside it as `third-party/vipe`)
- [MengHao666/Hand-BMC-pytorch](https://github.com/MengHao666/Hand-BMC-pytorch) → only needed
  once, to generate the BMC priors with `scripts/prepare_bmc.py`

### Weights

| File | Goes to |
| --- | --- |
| WiLoR `wilor_final.ckpt`, `detector.pt` | `$EGO2ROBOT_STORE/checkpoints/wilor/` |
| `MANO_RIGHT.pkl` (from mano.is.tue.mpg.de) | `$EGO2ROBOT_STORE/checkpoints/mano/` |
| MANO + `mano_mean_params.npz` | `Dyn-HaMR/_DATA/data/` (symlink is fine) |
| BMC priors (8 `.npy`) | `Dyn-HaMR/_DATA/BMC/`, via `python scripts/prepare_bmc.py --repo <Hand-BMC-pytorch>` |
| SAM 3 (gated `facebook/sam3`) | anywhere; point `EGO2ROBOT_SAM3_DIR` at it |
| VIPE weights (DROID-SLAM, AoT, SAM ViT-B, GeoCalib, UniDepth-v2, bert-base) | `~/.cache/torch/hub/` and `~/.cache/huggingface/hub/` |

Dyn-HaMR's own HaMeR/ViTPose/HMP/VPoser checkpoints are **not** needed: this pipeline
replaces Dyn-HaMR's preprocessing with WiLoR, and eq. (6) has no motion or pose prior.

Then patch Dyn-HaMR once:

```bash
python scripts/patch_dynhamr.py    # depth range [0.05, 0.4] m, bio loss weights
```

### Robot models

Stage ⑤ reads MJCF models from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).
A sparse checkout of the arms in use is enough (151 MB for the seven below, against ~2 GB
for the whole repository):

```bash
git clone --filter=blob:none --no-checkout --depth 1 \
    https://github.com/google-deepmind/mujoco_menagerie.git $STORE/third_party/mujoco_menagerie
cd $STORE/third_party/mujoco_menagerie
git sparse-checkout init --cone
git sparse-checkout set franka_emika_panda arx_l5 robotiq_2f85 \
    trossen_vx300s trossen_wx250s agilex_piper i2rt_yam
git checkout main
```

Six morphologies are registered in `pipeline/robots/`. Each spec only names bodies and
joints; the TCP, the grasp frame and the gripper map are derived from the model, and both
derivations are checked against Table 3 — the reach column against the measured maximum
TCP distance from the base (`RobotModel.measure_reach`), the stroke column against the
measured pad travel (`RobotModel.measure_opening`):

- `panda` — 7 DOF, 0–80 mm, reach 1.272 m, measured 1.282 m (+0.8%), travel 80.0 mm
- `arx_l5` — 6 DOF, 0–88 mm, reach 0.855 m, measured 0.855 m (−0.0%), travel 88.0 mm
- `viperx` — 6 DOF, 15–87 mm, reach 0.911 m, measured 0.928 m (+1.9%), travel 72.0 mm
- `widowx` — 6 DOF, 11–55 mm, reach 0.787 m, measured 0.789 m (+0.2%), travel 44.0 mm
- `piper` — 6 DOF, 0–70 mm, reach 0.883 m, measured 0.868 m (−1.7%), travel 70.0 mm
- `yam` — 6 DOF, 4–75 mm, reach 0.866 m, measured 0.875 m (+1.0%), travel 72.5 mm

Menagerie covers 14 of Table 3's 15 morphologies: `franka_emika_panda`, `franka_fr3`,
`kuka_iiwa_14`, `kinova_gen3`, `rethink_robotics_sawyer`, `ufactory_xarm7`,
`universal_robots_ur5e`, `universal_robots_ur10e`, `trossen_vx300s` (ViperX),
`trossen_wx250s` (WidowX), `arx_l5`, `agilex_piper`, `i2rt_yam` and `aloha`. Jaco is not
there. Note that the UR arms, the IIWA, the Gen3 and the Sawyer ship without a gripper
while Table 3 quotes a stroke for them (0–85 mm is a Robotiq 2F-85), so those need a
gripper attached and a TCP re-derived; and Menagerie's `aloha` is the Trossen ALOHA 2, not
the AgileX Aloha-Agilex of Table 3, whose URDF comes with RoboTwin 2.0.

## Running stage ①

```bash
CONDA=$(conda info --base)/envs
STORE=${EGO2ROBOT_STORE:-/root/paddlejob/ego}
VIDEO=$STORE/data/test_videos/egodex_sample.mp4
OUT=$STORE/outputs/s1_egodex_sample
```

All six stages in order, for one clip and any number of morphologies:

```bash
bash scripts/run_clip.sh $VIDEO panda arx_l5 widowx yam piper viperx
```

**The clip must be one continuous shot.** VIPE's camera track, Dyn-HaMR's temporal
optimization (eq. 6), ProPainter's reference frames and the single base pose of eq.(4) are
all defined within a take, and none of them degrade gracefully across a cut. The bundled
`egodex_sample.mp4` is *not* one shot: it has hard cuts at frames 94 and 360 (frame
difference 6–10× the median; VIPE's camera pose jumps 39 mm / 3.2° at frame 94, against a
2 mm / 0.17° median step). Splitting it cut ①'s right-hand reprojection error by 2.5–4× and
brought VIPE down from 83 minutes to 86 s; the base search is the one step that survives a cut
without a measurable loss, because eq.(4) averages feasibility over keyframes.

The individual steps, if a stage needs to be re-run on its own:

```bash
# 1. WiLoR + SAM 3 filter + association + jump filter -> hand_pose.h5
$CONDA/ego2robot/bin/python -m pipeline.s1_hand_pose.run_stage1 \
    --video $VIDEO --out-dir $OUT --viz

# 2. Export for Dyn-HaMR (H.264 copy, frames, per-frame keypoint/MANO JSONs)
$CONDA/ego2robot/bin/python -m pipeline.s1_hand_pose.dynhamr_bridge \
    --h5 $OUT/hand_pose.h5 --root $STORE/outputs/dynhamr_data

# 3. Camera intrinsics + trajectory
$CONDA/vipe/bin/vipe infer $STORE/outputs/dynhamr_data/videos/egodex_sample.mp4

# 4. Re-export with the true focal length, then temporal optimization (eq. 6)
$CONDA/ego2robot/bin/python -m pipeline.s1_hand_pose.dynhamr_bridge \
    --h5 $OUT/hand_pose.h5 --root $STORE/outputs/dynhamr_data \
    --vipe-intrinsics <vipe intrinsics npz>
$CONDA/dynhamr/bin/python scripts/run_dynhamr.py \
    data=ego2robot data.seq=egodex_sample is_static=False run_vis=False

# 5. Read the refined poses back into hand_pose.h5
$CONDA/dynhamr/bin/python -m pipeline.s1_hand_pose.dynhamr_import \
    --run-dir $STORE/third_party/Dyn-HaMR/outputs/logs/video-custom/<date>/<run> \
    --h5 $OUT/hand_pose.h5
```

Step 3 is not optional. WiLoR's camera model uses the heuristic focal length
`5000/256·max(W,H)` (37500 px for 1080p), which puts the hands 16–31 m from the camera;
VIPE's GeoCalib estimates the true value (758 px on the sample clip), and since
`cam_crop_to_full` makes only `tz` proportional to the focal length, rescaling `tz`
recovers metric depth.

### Output format

`hand_pose.h5` has one group per hand (`left`, `right`). Raw per-frame WiLoR arrays sit at
the top of the group (`valid`, `score`, `bbox`, `kp2d`, `kp3d_cam`, `cam_t`,
`global_orient`, `hand_pose`, `betas`); the Dyn-HaMR result is added as a `refined`
subgroup (`trans`, `root_orient`, `hand_pose`, `betas`, `kp3d_world`, `kp3d_cam`, `cam_R`,
`cam_t`, `intrins`, `valid`) with the source checkpoint recorded in its attributes; stage ②
adds a `gripper` subgroup (`position`, `quat`, `width` in the world frame, `position_cam`,
`quat_cam`, `valid`, `frozen`). Quaternions are xyzw.

## Running stage ②

```bash
$CONDA/dynhamr/bin/python -m pipeline.s2_retarget.run_stage2 --h5 $OUT/hand_pose.h5
```

## Running stage ③

```bash
OUT3=$STORE/outputs/s3_egodex_sample
$CONDA/ego2robot/bin/python -m pipeline.s3_arm_seg.run_stage3 \
    --video $VIDEO --out-dir $OUT3
```

## Running stage ④

```bash
OUT4=$STORE/outputs/s4_egodex_sample
$CONDA/ego2robot/bin/python -m pipeline.s4_hand_removal.run_stage4 \
    --video $VIDEO --mask-dir $OUT3/arm_mask --out-dir $OUT4 --resize-ratio 0.5
```

`--resize-ratio 0.5` is what the reference run used: 1080p needs ~58 GB of VRAM, which a
shared GPU does not have. Drop it for full-resolution inpainting if the card is free —
`collect()` composites either way, so only the pixels inside the mask change.

## Running stage ⑤

```bash
OUT5=$STORE/outputs/s5_egodex_sample
$CONDA/ego2robot/bin/python -m pipeline.s5_base_ik.run_stage5 \
    --h5 $OUT/hand_pose.h5 --robot panda --out-dir $OUT5
```

One run per morphology; `--robot` accepts any stem in `pipeline/robots/`. Output is
`robot_<name>.h5` (per hand: `qpos`, `feasible`, `position_error_m`,
`orientation_error_rad`, `width`, plus `base_position_world` / `base_rotation_world` and the
camera intrinsics as attributes, and the per-frame extrinsics `cam_R` / `cam_t` at the top
level so stage ⑥ can put that world pose back into each frame's camera) and
`stage5_<name>_stats.json`.

## Running stage ⑥

```bash
OUT6=$STORE/outputs/s6_panda
VIPE=$STORE/third_party/Dyn-HaMR/third-party/vipe/vipe_results
MUJOCO_GL=osmesa $CONDA/ego2robot/bin/python -m pipeline.s6_composite.run_stage6 \
    --stage5 $OUT5/robot_panda.h5 --inpainted $OUT4/inpainted \
    --mask-dir $OUT3/arm_mask --depth $VIPE/depth/egodex_sample.zip --out-dir $OUT6
```

`MUJOCO_GL=osmesa` is required: this host has no libEGL, so rendering goes through Mesa's
software rasteriser (`apt-get install libosmesa6`). Output is `composited/%06d.png` — the
finished training frames — plus `stage6_<name>_stats.json`.

## Cost per clip

Measured on one A800, warm caches, `HF_HUB_OFFLINE=1`, at 1920×1080:

- ① WiLoR + SAM 3 + association: 4–10 min, scales with frame count
- VIPE: 86 s for 94 frames, 199 s for 272 frames
- Dyn-HaMR eq. (6), 300 `smooth_fit` iterations: 9.7 min for 160 frames with the host to
  itself. It is partly CPU-bound, so concurrent clips stretch it: a 94-frame clip sharing
  the machine with two others took over 17 min
- ② retargeting: under a minute
- ③ SAM 3 arm masks: ~50 s
- ④ ProPainter: 1.3–2.2 min at `--resize-ratio 0.5`
- ⑤ + ⑥ per morphology: 1–9 min, all CPU (mink IK and OSMesa rendering). The spread is the
  base search: it stops early when a candidate scores near 1, so an arm that fits the clip
  costs seconds and one that does not costs minutes (0.6–300 s per arm over the three takes)

So roughly 15–25 minutes for a clip plus two minutes per morphology. Two things dominated
that budget until they were pinned down, and both are worth knowing:

- **Without `HF_HUB_OFFLINE=1`**, huggingface_hub checks each checkpoint for updates on
  every run. Where the network cannot reach it, that is 5 retries with exponential backoff
  per file, and VIPE took **43 minutes instead of 3** — 100 retry lines in the log, no extra
  computation. `run_clip.sh` sets the variable.
- **A scene cut** costs both quality and time: the 120-frame window straddling this clip's
  frame-94 cut took VIPE 83 minutes (that run also downloaded weights), against 86–199 s for
  cut-free clips of 94 and 272 frames.

Clips are independent, and every third-party artifact is keyed by clip name, so the way to
scale is one clip per GPU: `CUDA_VISIBLE_DEVICES=n bash scripts/run_clip.sh ...`.

## Visualization

Each stage renders its own video, so a product can be inspected without the previous
stage's overlays in the way:

```bash
$CONDA/ego2robot/bin/python -m pipeline.viz.stage1 \
    --h5 $OUT/hand_pose.h5 --out $OUT/stage1_hand_pose.mp4
$CONDA/ego2robot/bin/python -m pipeline.viz.stage2 \
    --h5 $OUT/hand_pose.h5 --out $OUT/stage2_gripper.mp4
$CONDA/ego2robot/bin/python -m pipeline.viz.stage3 \
    --stats $OUT3/stage3_stats.json --out $OUT3/stage3_arm_mask.mp4
$CONDA/ego2robot/bin/python -m pipeline.viz.stage4 \
    --stats $OUT4/stage4_stats.json --out $OUT4/stage4_inpainted.mp4
$CONDA/ego2robot/bin/python -m pipeline.viz.stage5 \
    --stats $OUT5/stage5_panda_stats.json --out $OUT5/stage5_panda.mp4
$CONDA/ego2robot/bin/python -m pipeline.viz.stage6 \
    --stats $OUT6/stage6_panda_stats.json --out $OUT6/stage6_panda.mp4 --annotate
```

- `stage1`: raw WiLoR skeleton (grey), refined skeleton (per-hand colour), and a wrist-depth
  strip showing both traces against the [0.05, 0.4] m band
- `stage2`: gripper jaw line, the three grasp-frame axes (red x approach, green y normal,
  blue z grasp) and an opening-width strip with the 1 cm degeneracy threshold
- `stage3`: translucent arm mask with its contour, and a mask-area trace
- `stage4`: the inpainted frame with the original inset in the corner, the mask contour, and
  a trace of how much changed inside the mask
- `stage5`: the IK-solved arms as projected link skeletons, base markers, TCP dots and a
  per-frame IK position error trace
- `stage6`: the finished video, clean by default; `--annotate` adds the robot coverage and a
  trace of how many gripper pixels the depth test hid

## Tests

```bash
python tests/test_pipeline.py
```

### Verification on the sample clip

The sample clip is 632 frames of seated bimanual manipulation at 1920×1080, containing three
takes (cuts at frames 94 and 360). Everything below is the 120-frame window 0–119, which was
processed before the cuts were found and therefore straddles the first one; the three cut-free
takes are reported after it.

- ① tracks: left 92/120 valid frames (12 dropped by the jump filter), right 113/120 (7 dropped)

- SAM 3 hand-mask filter: 0 of 225 detections dropped, and a mask was found on every frame.
  The clip only ever shows the camera wearer's own hands, so the filter has nothing to do
  here; it is exercised, not validated, by this sequence.
- WiLoR inference is not bit-reproducible (cuDNN autotuning): repeated runs move keypoints
  by up to ~0.9 px, which is enough to flip a borderline frame in the jump filter (one
  left-hand frame out of 120). Expect the valid-frame counts to move by ±1 between runs.
- refined wrist depth: 0.266–0.373 m (left), 0.260–0.393 m (right) — inside the paper's
  [0.05, 0.4] m constraint band, from 16–31 m before the focal correction
- reprojection of refined 3D joints against the WiLoR 2D keypoints: 0.50 px median / 1.47 px
  p90 (left), 0.86 px / 3.19 px (right)
- wrist speed in the world frame: 3.2 / 5.9 mm per frame median (left / right), p95 ≈ 20,
  i.e. 10–20 cm/s at 30 fps
- bone lengths from the refined MANO: wrist→index MCP 62 mm, phalanges 25/17/21 mm.
  These shrink as the optimization proceeds (69/28/19/22 mm at iteration 18) because
  `shape_prior` is weak (0.05) and `betas` drift outward to ‖betas‖ ≈ 7. Worth watching:
  stage ② derives gripper width from absolute fingertip distances.

Stage ②, same clip:

- gripper opening: 1.7–6.6 cm (left, median 4.6), 1.6–4.8 cm (right, median 4.0) — a
  plausible parallel-jaw range, and never below the 1 cm degeneracy threshold, so the
  orientation fallback never fired
- gap handling filled all 28 missing left-hand frames and all 7 right-hand ones; every gap
  was ≤ 10 frames, so nothing needed the home configuration
- velocity filter (2 rounds) flagged position/rotation outliers left 3+0 / 3+3, right 0+0 /
  6+0. The rotation count does not always reach zero: removing outliers lowers the median
  that sets the threshold, so the paper's fixed 2 rounds is a stopping rule, not convergence
- smoothing cut median acceleration from 3.4 to 0.69 mm/frame² (left) and 8.1 to 1.4 (right),
  and rotational acceleration from 6.1 to 0.41 °/frame² and 10.1 to 0.45
- it preserves the gross motion: net displacement 7.03 → 6.80 cm (left), 4.64 → 5.17 cm
  (right), spread about the mean within 3%, per-frame deviation median 3.8 / 7.6 mm
- equations (1)-(3) reproduce to 1e-16 for both handedness signs, and the smoothed
  quaternions stay proper rotations (orthonormality 9e-16, det = 1)

Stage ③, same clip:

- SAM 3 returned a mask on 120/120 frames in 89 s, so none of the A.4 repairs had anything
  to do (0 interpolated, 0 area-replaced, 0 missing)
- mask area 2.13%–7.73% of the frame (median 4.84%), covering both sleeves and both hands
  while excluding the held cup
- SAM 3's optional `kernels` post-processing stays off; the paper's own (iii) close covers
  the hole filling it would have done

Stage ④, same clip:

- 120 frames in 328 s end to end (ProPainter's own transformer loop 93 s), at
  `--resize-ratio 0.5`; every paper hyperparameter kept
- inside the stage-③ mask the frame changed by 21.8/255 mean absolute difference; outside
  the dilated mask it is bit-identical to the source (max difference exactly 0)
- 1080p is out of reach on this host: ProPainter grew to 57.8 GB before OOMing on an 80 GB
  A100 that already had ~26 GB of other tenants' processes on it. The transformer sees
  `neighbor_length + subvideo_length / ref_stride` = 19 frames at once, so its cost scales
  with the frame area; 960×540 peaks at 7.2 GB
- two artifacts survive and are inherent to the method, not to this implementation: the
  arms' cast shadow stays (a "person" mask contains no shadow, so ProPainter treats the
  shadow as background worth propagating), and object pixels occluded by the fingers come
  back deformed - the grasped cup melts into the table around frame 60. Step ⑥ renders the
  robot over roughly the same region, which hides part of both

Stage ⑤, frames 0–93 of the same clip (the first take — see "Running stage ①" on the cuts),
six morphologies, base solved in the world frame:

- derived-TCP reach and gripper travel reproduce Table 3 for all six (list under "Robot
  models"); a TCP left at the flange comes out 7–16 cm short, which is what makes this an
  acceptance test rather than a formality
- solved frames per arm, left/right: Panda 94/94 and 94/94, ViperX 93/94 and 94/94, WidowX
  93/94 and 92/94, ARX-L5 91/94 and 78/94, Piper 86/94 and 73/94, YAM 83/94 and 79/94.
  Median IK position error is below 1 µm wherever a frame is solved, p90 1.2–5.9 µm for the
  arms that solve everything; the misses are frames a 6-DOF wrist cannot reach at all, which
  is what the L1 curation is meant to drop, and they show up as a large p90 (15–30 mm) because
  an unsolved frame keeps the closest configuration mink reached
- the base search reaches feasibility rate 1.00 on the keyframes for Panda, ViperX and WidowX
  and 0.95 for the other three, always at `rho_bar` 0.64–0.69 against the 0.65 target
- A.4's reach pruning removes about half the position grid (124–130 of 245), and the score
  bound then visits 1–47 of the ~115 survivors, scoring 45–2115 candidates out of 11,025 —
  0.4–19% of the grid, same argmax. Wall clock per arm is 0.6–243 s; the slow end is a low
  best score, which keeps the bound open longer
- the 25-pair joint check rejected nothing on this clip: the separation it enforces is
  0.437 m for Panda and 0.14–0.22 m for the small arms, below every placement the search picks
- the cut does *not* show up here. On frames 0–119, which straddle the frame-94 cut, ARX-L5
  solves 117/120 and 100/120, the same rate as its 91/94 and 78/94 on take1 alone: eq.(4)
  averages feasibility over keyframes, so a base that suits two scenes at once still scores
  well. The cut's measured cost is in stages ①, ④ and VIPE instead (see the three-take
  section)
- eq.(8) scores nothing but reachability, so nothing holds the base at a plausible mounting
  height. The five 0.75–0.85 m arms all land 0.25–0.45 m below the frame-0 camera and
  0.17–0.73 m in front of it, which is about where a table would be for a seated wearer;
  Panda's 1.27 m reach pushes its base further out and higher, to 0.35 m below the camera and
  0.91 m in front. The paper has no support, visibility or mounting term, so an implausible
  height is the method behaving as specified rather than a bug

Stage ⑥, same window and morphologies:

- 94 frames in 32–64 s per morphology — 0.34–0.68 s per frame at 1920×1080 for three
  rendering passes, on Mesa's *software* rasteriser. Fifteen morphologies would be about
  13 minutes per 100-frame clip
- `camera_check` (rendered depth vs `mj_ray` on the same pixel ray, both restricted to the
  geom groups the renderer draws) has a median of 0.1–9.5 µm over all 36 arm-runs of the three
  takes, and the TCP projects 0–31 px from the jaws in the median frame
- the robot covers 8.4% of the frame for WidowX, 10.6% for Piper, 13.5% for YAM, 13.7% for
  ViperX, 14.5% for ARX-L5 and 24.0% for Panda. A 1.27 m arm is simply large for a camera
  0.4 m from the hands: its base has to sit ~0.9 m away, which is about where the wearer's
  head is, so links pass close to the lens. The paper's own single-morphology ablation uses
  ARX-L5, and 8 of its 15 morphologies reach under 0.92 m
- only 0.01–0.04% of gripper pixels lose the depth test. In this clip the hands are the
  nearest thing to the camera, so there is almost nothing to be occluded by; the branch is
  exercised, not validated, by this sequence
- what remains visible in the output: no robot shadow is cast onto the scene (MuJoCo's render
  is composited, not relit), and stage ④'s residual human-arm shadow is still on the table

### The three cut-free takes, six morphologies each

Splitting the sample clip at its two cuts and running all three takes through the whole
pipeline gives 18 clip-morphology pairs over 526 frames. This is the strongest correctness
signal the reproduction has, because stages ①–④ are shared and stage ⑤ is
morphology-agnostic: an artifact in every column of `pipeline.viz.compare` is an
implementation bug, one in a single column is that arm meeting this clip.

Hand pose, ①'s reprojection error against the WiLoR 2D keypoints (median / p90):

- take1, frames 0–93: left 0.22 / 1.00 px, right 0.29 / 1.98 px
- take2, frames 94–253: left 0.48 / 2.26 px, right 0.34 / 2.10 px
- take3, frames 360–631: left 0.26 / 1.73 px, right 0.19 / 1.95 px
- the window straddling the cut, for comparison: left 0.50 / 1.47 px, right 0.86 / 3.19 px

Stage ② leaves take1 and take3 with every frame valid for both hands and no unfilled gaps
(94/94 and 272/272); take2's left hand is 126/160, that take keeping it near the frame edge.

Stage ⑤, solved frames per arm as a fraction of the frames stage ② left valid, left/right:

- take1 (94/94 valid): Panda 94/94 and 94/94, ViperX 93/94 and 94/94, WidowX 93/94 and 92/94,
  ARX-L5 91/94 and 78/94, Piper 86/94 and 73/94, YAM 83/94 and 79/94
- take2 (126/160 valid): Panda 126/126 and 160/160, ARX-L5 122/126 and 160/160, ViperX
  122/126 and 160/160, WidowX 121/126 and 160/160, YAM 115/126 and 160/160, Piper 96/126
  and 159/160
- take3 (272/272 valid): Panda 272/272 and 272/272, WidowX 267/272 and 269/272, ViperX
  262/272 and 272/272, ARX-L5 226/272 and 270/272, Piper 199/272 and 267/272, YAM 163/272
  and 269/272

Two things are visible in that. The ordering is by wrist DOF and reach, not by clip — 7-DOF
Panda solves every valid frame of all three takes, and the 6-DOF arms lose frames where a
human wrist orientation is out of reach, which is what L1 curation exists to drop. And the
left arm is consistently harder than the right on take2 and take3, where the wearer works
mostly with the right hand and the left one sits near the frame edge with a noisier pose.

Stage ⑥'s `camera_check` has a median of 0.1–9.5 µm over the 36 arm-runs, and a worst frame
under 0.9 mm in 16 of the 18 pairs. The two exceptions (ViperX on take1, 30 mm; Panda on
take3, 48 mm) are single pixels on a geom boundary, where the rasteriser filled the pixel from
one mesh and the ray hit the mesh behind it: all eight neighbours of that pixel agree to
0.1 µm. Compositing is unaffected — eq.(9) never uses the ray.

End-to-end, robot against the human it was retargeted from, for Panda and ARX-L5 on all three
takes (12 combinations):

- commanded grasp point vs the retargeted human grasp point: 0.0000–0.0003 mm, 0.0000–0.0005 px
- robot TCP vs the midpoint of WiLoR's thumb and virtual fingertip in 2D: 8.9–14.8 px, which is
  the offset between a MANO fingertip and a gripper pad, not an error
- commanded gripper width vs human finger distance: r = 1.000 in all 12

What the cut cost, measured: the straddling window degraded ①'s right-hand reprojection by
2.5–4×, made ProPainter draw reference frames from the wrong scene (the grasped cup melts into
the table), and made VIPE take 83 minutes instead of 86–199 s. Stage ⑤ is the one stage it
does not measurably hurt — see the bullet under "Stage ⑤" above.

## Reproduction notes


Deliberate deviations, all forced by the environment rather than by the method:

- `scripts/run_dynhamr.py` stubs out `pyrender` (no EGL/OSMesa on the host) and VPoser's
  loader. Neither is used by eq. (6): `run_prior: False`, and Dyn-HaMR's hand branch keeps
  `latent_pose` as the raw 15×3 axis-angle pose, so nothing decodes a VPoser latent.
- Dyn-HaMR's other default loss terms (`shape_prior`, `bone_length`) are left at their
  repository values; the paper only specifies the three terms of eq. (6).
- `smooth_fit` runs the repository default of 300 iterations. The early exit is disabled in
  this Dyn-HaMR copy (the line setting `reached_max = True` is commented out in
  `optim/optimizers.py:604`), so the stage always runs to the cap, which costs 9.7 min for a
  160-frame clip on an otherwise idle host — cheap enough not to tune. An earlier reading of
  this as ~150 s per iteration came from a run that was interrupted and resumed, not from the
  optimizer.
- Resuming `smooth_fit` from its checkpoint raises `element 0 of tensors does not require
  grad`: `run()` calls `load_checkpoint` after the optimizer has taken references to the
  parameter tensors, and `load_dict` replaces them with non-leaf copies. The result file
  written at resume time is still valid, because it is saved before the first step.
- SAM 3 runs without its `kernels`-based post-processing, which needs `trust_remote_code`.
- Stage ⑤ resolves three things A.4 leaves undefined:
  - **Pitch sign.** A.4 lists pitch `{30°, 45°, 60°}` without saying which way. Every base
    candidate that survives the reach pruning on this clip sits *below* the trajectory, so
    nose-down aims the arm away from the targets: feasibility is 0.00 for every arm, against
    0.71–1.00 with nose-up. Positive pitch is therefore taken as nose-up, about the base's
    own right axis.
  - **Left-hand grasp axis.** Eq.(2) uses `s = -1` for the left hand, so the two hands hand
    over grasp frames that differ by a 180° turn about the approach axis. A parallel jaw is
    symmetric about that axis, so both describe the same physical grasp, but a wrist with
    less than a full turn of travel (Panda's `joint7` stops at ±166°) can only reach one of
    them. The left arm's targets are mirrored back to `s = +1`; without it the left arm
    scores 0.10–0.20 where the right scores 0.90.
  - **Joint verification.** "Jointly verify all 25 left–right combinations" is implemented as
    the best combined score among pairs whose bases are farther apart than the sum of their
    base radii. Arm-versus-arm collision along the trajectory is not checked; that belongs to
    the L1 self-collision curation, which is not built yet.
- Keyframe selection is farthest-point sampling over a position + 0.1 m/rad orientation
  metric. The paper only states the intent ("cover the spatial extremes").
- Reach `r` is taken from Table 3 rather than from the model, because it is what A.4's grid
  is scaled by. The measurement is still run and reported, as a check on the derived TCP.
- mink's IK is regularised with `DampingTask`, not `PostureTask`. A posture task has no null
  space to hide in on a 6-DOF arm, so it trades pose error against posture error and settles
  above the paper's 1e-5 threshold: ARX-L5 then recovered 4 of 30 poses taken from its own
  forward kinematics *even when started from the exact answer*, and its measured feasibility
  rate collapsed to 0. With velocity damping it recovers 30 of 30 in a median of 8 iterations.
- Stage ⑥ takes `D_scene` from the VIPE run stage ① already needed for the camera, not from
  Depth Anything V3 as A.4 states. VIPE's depth is metric, full resolution and expressed in
  the same scale the hand trajectory was solved in, so it can be compared against MuJoCo's
  depth without a further alignment step; DA-V3 would need one. Swapping it is a one-function
  change (`read_depth`).
- Rendering runs on OSMesa because the host has no EGL. That is a speed choice, not a
  fidelity one — the raster output is identical.
- The paper cites "WiLoR [34, 40]" where [40] is AnyHand; this uses the official WiLoR
  weights only.

Two bugs worth recording, because both were invisible in the numbers they were meant to be
checked by:

- **Eq.(4) has to be solved in the world frame.** A.4 describes the base grid in camera
  coordinates and the policy consumes camera-frame actions, so the first version searched, and
  stored, the base pose in the camera frame. It scored *better* — ARX-L5's right arm solved
  94/94 on take1 against 78/94 now — because a base that is re-anchored to the head every
  frame can chase the hand. It is also unphysical: the head moves 95 mm and 77° over take3, so
  that base slid the same amount through the scene, and stage ⑥ rendered the arm pinned to one
  pixel while the room moved behind it. In the world frame the same base sweeps 221–748 px
  across the image, which is what a bolted-down robot looks like from a moving head. The fix
  is stage ⑤ writing `base_position_world` / `base_rotation_world` and `cam_R` / `cam_t`, and
  stage ⑥ moving the arms to `R_i p_world + t_i` per frame.
- **`mj_ray` sees collision geometry the renderer does not draw.** `camera_check` compares the
  rendered depth with an independent ray query, and an unrestricted ray returns the group-3
  collision primitive rather than the group-2 visual mesh the rasteriser drew — on ARX-L5 that
  proxy stands 0.2 m in front of the surface, so the check reported errors up to 198 mm on a
  camera model that was in fact exact. Restricting the ray to `MjvOption.geomgroup` brings the
  same frames to 0.9 mm worst case and ~2 µm median. The composited frames were never affected:
  eq.(9) tests against the rendered depth, not against the ray.

Not built yet: A.1's large-gap fill with the robot home configuration (it needs a
morphology, so it could only land with stage ⑤), the L1/L2/L3 curation, and A.2's VLM
subtask segmentation.





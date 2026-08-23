"""Global paths and constants for the Ego2Robot reproduction.

Layout convention:
- Large files (checkpoints / datasets / outputs) live under $EGO2ROBOT_STORE.
- The workspace only holds source code and third-party repositories.
"""
import os
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
THIRD_PARTY = WORKSPACE / "third_party"
WILOR_ROOT = THIRD_PARTY / "WiLoR"

STORE = Path(os.environ.get("EGO2ROBOT_STORE", "/root/paddlejob/ego"))
CKPT_DIR = STORE / "checkpoints"
DATA_DIR = STORE / "data"
OUT_DIR = STORE / "outputs"

# Large third-party repositories live next to the checkpoints, not in the
# workspace: the workspace disk is a small sshfs mount.
DYNHAMR_ROOT = STORE / "third_party" / "Dyn-HaMR"
VIPE_ROOT = DYNHAMR_ROOT / "third-party" / "vipe"
PROPAINTER_ROOT = STORE / "third_party" / "ProPainter"

WILOR_CKPT = CKPT_DIR / "wilor" / "wilor_final.ckpt"
WILOR_DETECTOR = CKPT_DIR / "wilor" / "detector.pt"
MANO_RIGHT_PKL = CKPT_DIR / "mano" / "MANO_RIGHT.pkl"
# Local copy of the gated facebook/sam3 release.
SAM3_DIR = Path(os.environ.get(
    "EGO2ROBOT_SAM3_DIR", "/root/paddlejob/bosdata/liangjunhao/models_weight/sam3"))

# WiLoR returns 21 hand keypoints in OpenPose hand order:
#   0 = wrist
#   1-4   = thumb  (4  = tip)
#   5-8   = index  (8  = tip)
#   9-12  = middle (12 = tip)
#   13-16 = ring   (16 = tip)
#   17-20 = pinky  (20 = tip)
# This assumes mano_wrapper applied its joint_map remapping. Verify with the
# stage-1 overlay video before trusting it: the retargeting in stage 2 depends
# on thumb/index/middle tips being correct.
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12

HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

# Paper appendix A.1: jump filter threshold = max(4 x median velocity, 0.003 m/frame).
JUMP_VEL_MEDIAN_FACTOR = 4.0
JUMP_VEL_FLOOR_M_PER_FRAME = 0.003

# Paper A.1: a detection is discarded when more than 80% of its projected
# keypoints fall outside the SAM 3 hand mask.
HAND_MASK_OUTSIDE_RATIO = 0.8

# Paper A.1: DynHaMR constrains reconstructed hand depth to [0.05, 0.4] m.
HAND_DEPTH_MIN_M = 0.05
HAND_DEPTH_MAX_M = 0.40

# Paper A.1 gap handling. Gaps longer than 10 frames are filled with the robot's
# home configuration and blended at the boundaries over
#   n = max(5, min(90, ceil(0.6 n_pos + 0.4 n_rot)))
# frames, with n_pos = dp / 3.25mm and n_rot = dtheta / 1.08deg.
GAP_LARGE_FRAMES = 10
BLEND_POS_STEP_M = 0.00325
BLEND_ROT_STEP_DEG = 1.08
BLEND_POS_WEIGHT = 0.6
BLEND_ROT_WEIGHT = 0.4
BLEND_MIN_FRAMES = 5
BLEND_MAX_FRAMES = 90

# Paper A.4: SAM 3 segments with a text prompt, anchored on the middle frame and
# propagated temporally; long videos run in 400-frame chunks with 50-frame
# overlap. Arm segmentation (step 3) prompts "person"; the A.1 detection filter
# needs the hand only.
SAM3_CHUNK_FRAMES = 400
SAM3_CHUNK_OVERLAP = 50
SAM3_ARM_PROMPT = "person"
SAM3_HAND_PROMPT = "hand"

# Paper A.4 arm-mask post-processing:
#   (i)   gaps <= 3 frames are filled by interpolating neighbouring masks;
#   (ii)  frames whose mask area is < 50% of the local median (window 11) are
#         replaced by the nearest valid mask;
#   (iii) morphological close with a 5x5 elliptical kernel.
MASK_GAP_MAX_FRAMES = 3
MASK_AREA_MIN_RATIO = 0.5
MASK_AREA_MEDIAN_WINDOW = 11
MASK_CLOSE_KERNEL = 5

# Paper A.4 hand removal: "ProPainter runs at fp16 with neighbor_length = 10,
# ref_stride = 10, subvideo_length = 80, mask_dilation = 4, RAFT iterations = 20",
# which is ProPainter's own default configuration plus fp16.
PROPAINTER_CKPT_DIR = CKPT_DIR / "propainter"
PROPAINTER_FP16 = True
PROPAINTER_NEIGHBOR_LENGTH = 10
PROPAINTER_REF_STRIDE = 10
PROPAINTER_SUBVIDEO_LENGTH = 80
PROPAINTER_MASK_DILATION = 4
PROPAINTER_RAFT_ITER = 20

# Paper eq.(1): the virtual fingertip blends the index and middle finger tips.
VF_INDEX_WEIGHT = 0.7
VF_MIDDLE_WEIGHT = 0.3

# Paper A.3 "Degenerate Orientation": below this gripper width, or when the grasp
# axis and the wrist-to-fingertip vector are nearly parallel, the grasp frame is
# undefined and the last valid orientation is held instead.
GRIPPER_WIDTH_MIN_M = 0.01
DEGENERATE_CROSS_EPS = 1e-6

# Paper A.3 eq.(7) velocity filter: tau = max(5 x median(v), floor), where the
# floor is 0.9/fps m/frame for position and 10.0/fps rad/frame for rotation.
# Offending frames are re-interpolated from their neighbours, for 2 rounds.
VEL_MEDIAN_FACTOR = 5.0
VEL_POS_FLOOR_M_PER_S = 0.9
VEL_ROT_FLOOR_RAD_PER_S = 10.0
VEL_FILTER_ROUNDS = 2

# Paper A.3 temporal smoothing: Savitzky-Golay on positions and widths, and
# Gaussian-weighted SLERP on orientations.
SAVGOL_MAX_WINDOW = 21
SAVGOL_MAX_ORDER = 3
SLERP_SIGMA_FRAMES = 10.0
SLERP_KERNEL = 21

# Robot models. Only the arm directories are checked out, see scripts/setup_envs.sh.
MENAGERIE_ROOT = STORE / "third_party" / "mujoco_menagerie"
ROBOT_SPEC_DIR = WORKSPACE / "pipeline" / "robots"

# Paper A.4 "Base Pose Search": candidates are generated in the camera frame with
# offsets scaled by the morphology's reach r. Lateral offsets are sign-flipped per
# arm so that each arm sits on its own side of the trajectory.
BASE_LATERAL_FACTORS = (0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2)
BASE_FORWARD_FACTORS = (-0.1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9)
BASE_VERTICAL_FACTORS = (0.4, 0.2, 0.0, -0.2, -0.4)
BASE_PITCH_DEG = (30.0, 45.0, 60.0)
BASE_YAW_DEG = (-45.0, -20.0, 0.0, 20.0, 45.0)
BASE_ROLL_DEG = (-15.0, 0.0, 15.0)

# Paper A.4 pruning: candidates closer than 0.20 m to the camera are discarded, as
# are those with trajectory points beyond 0.9 r or closer than 0.08 m to the base.
BASE_MIN_CAMERA_DIST_M = 0.20
BASE_MAX_REACH_RATIO = 0.9
BASE_MIN_TRAJ_DIST_M = 0.08

# Paper A.4 eq.(8): S = FR - 5.0 |rho_bar - 0.65|, where rho_bar is the mean
# end-effector distance from the base normalised by the reach.
BASE_TARGET_REACH_RATIO = 0.65
BASE_REACH_PENALTY = 5.0
BASE_TOPK_PER_ARM = 5

# Paper A.4: "FR is the IK feasibility rate over up to 20 keyframes (using the mink
# IK solver, quadprog backend, 100 iterations, 1e-5 threshold)".
IK_MAX_KEYFRAMES = 20
IK_ITERS = 100
IK_THRESHOLD = 1e-5
IK_SOLVER = "quadprog"
# Not from the paper: mink needs an integration step and a low-priority regulariser.
# It has to damp velocities rather than pull towards a posture - see
# pipeline/s5_base_ik/ik.py for why a posture task breaks 6-DOF arms.
IK_DT = 1.0
IK_DAMPING_COST = 1e-3

# Running IK on the full 11,025-candidate grid is not affordable (see README).
# The grid is pre-ranked by the analytic reach term of eq.(8) and IK runs on this
# many survivors per arm, which is what "screen the top-5 candidates" implies.
BASE_IK_SHORTLIST = 64

# Paper A.4 depth-aware compositing: "The hand mask is dilated with a 5x5 kernel
# (1 iteration) before compositing to prevent revealing inpainted boundaries along
# the original arm contour."
COMPOSITE_HAND_DILATE_KERNEL = 5
COMPOSITE_HAND_DILATE_ITERATIONS = 1
# Not from the paper: MuJoCo's near plane is a fraction of the model extent, and an
# ego camera has robot links passing within ~0.1 m of the lens.
RENDER_ZNEAR_FRACTION = 0.002

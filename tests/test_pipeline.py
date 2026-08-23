"""Checks for the closed-form parts of stages 1-6: no fixtures, no pytest needed.

    python tests/test_pipeline.py

The stage-5 robot checks need a mujoco_menagerie checkout and skip without one.
"""
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import config, robots                             # noqa: E402
from pipeline.s1_hand_pose import gap_handling                 # noqa: E402
from pipeline.s2_retarget import retarget as rt, smoothing      # noqa: E402
from pipeline.s3_arm_seg import arm_mask                        # noqa: E402
from pipeline.s5_base_ik import candidates, ik, keyframes       # noqa: E402
from pipeline.s6_composite import run_stage6, scene as scene_mod  # noqa: E402


def _square(size: int, x0: int, y0: int, side: int = 20) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[y0:y0 + side, x0:x0 + side] = True
    return mask


def test_retarget_matches_equations():
    """Equations (1)-(3) hold exactly for both handedness signs."""
    kp = np.random.default_rng(0).normal(scale=0.05, size=(40, 21, 3))
    for is_right in (True, False):
        traj = rt.retarget(kp, np.ones(len(kp), bool), is_right=is_right)
        vf = 0.7 * kp[:, config.INDEX_TIP] + 0.3 * kp[:, config.MIDDLE_TIP]
        jaw = kp[:, config.THUMB_TIP] - vf
        width = np.linalg.norm(jaw, axis=-1)
        z = (1.0 if is_right else -1.0) * jaw / width[:, None]
        y = np.cross(z, vf - kp[:, config.WRIST])
        y /= np.linalg.norm(y, axis=-1, keepdims=True)
        rot = Rotation.from_quat(traj.quat).as_matrix()
        assert np.allclose(traj.width, width)
        assert np.allclose(traj.position, 0.5 * (kp[:, config.THUMB_TIP] + vf))
        assert np.allclose(rot[:, :, 2], z)
        assert np.allclose(rot[:, :, 1], y)
        assert np.allclose(rot[:, :, 0], np.cross(y, z))


def test_degenerate_orientation_is_held():
    """Paper A.3: a closed hand keeps the previous orientation instead of a new frame."""
    kp = np.zeros((3, 21, 3))
    kp[:, config.INDEX_TIP] = kp[:, config.MIDDLE_TIP] = [0.0, 0.05, 0.0]
    # Frame 1 closes the thumb onto the virtual fingertip: width 0.5mm < 1cm.
    kp[:, config.THUMB_TIP] = [[0.05, 0.05, 0], [0.0005, 0.05, 0], [0.05, 0.05, 0]]
    traj = rt.retarget(kp, np.ones(3, bool), is_right=True)
    assert traj.width[1] < config.GRIPPER_WIDTH_MIN_M
    assert traj.frozen.tolist() == [False, True, False]
    assert np.allclose(traj.quat[1], traj.quat[0])


def test_blend_length_formula():
    """n = max(5, min(90, ceil(0.6 n_pos + 0.4 n_rot))) with the paper's step sizes."""
    assert gap_handling.blend_length(0.0, 0.0) == config.BLEND_MIN_FRAMES
    assert gap_handling.blend_length(10.0, np.pi) == config.BLEND_MAX_FRAMES
    # 3.25mm and 1.08deg are exactly one frame of position and rotation blending.
    one = gap_handling.blend_length(config.BLEND_POS_STEP_M,
                                   np.radians(config.BLEND_ROT_STEP_DEG))
    assert one == config.BLEND_MIN_FRAMES        # 0.6 + 0.4 = 1 frame, clamped to 5
    assert gap_handling.blend_length(0.1, 0.0) == int(np.ceil(0.6 * 0.1 / 0.00325))


def test_small_gap_is_interpolated():
    positions = np.stack([np.arange(12.0), np.zeros(12), np.zeros(12)], axis=1)
    quats = np.tile([0.0, 0.0, 0.0, 1.0], (12, 1))
    valid = np.ones(12, bool)
    valid[5:8] = False                          # 3-frame gap, well under the 10 limit
    pos, quat, _, out_valid, stats = gap_handling.fill_gaps(positions, quats, valid)
    assert out_valid.all() and stats.small_filled == 3 and stats.unfilled == 0
    assert np.allclose(pos[:, 0], np.arange(12.0))


def test_savgol_window_is_odd_and_bounded():
    assert smoothing.savgol_window(120) == (21, 3)
    assert smoothing.savgol_window(20) == (19, 3)   # min(21, n) forced odd
    assert smoothing.savgol_window(4) == (3, 2)
    assert smoothing.savgol_window(2) == (0, 0)     # too short to filter


def test_gaussian_slerp_preserves_constant_and_smooths_noise():
    n = 60
    const = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))
    assert np.allclose(np.abs(smoothing.gaussian_slerp(const)), np.abs(const))

    rng = np.random.default_rng(1)
    angles = np.linspace(0, 0.5, n) + rng.normal(scale=0.05, size=n)
    noisy = Rotation.from_rotvec(np.stack([angles, angles * 0, angles * 0], 1)).as_quat()
    accel = lambda q: np.median(np.abs(np.diff(          # noqa: E731
        Rotation.from_quat(q).as_rotvec()[:, 0], 2)))
    assert accel(smoothing.gaussian_slerp(noisy)) < 0.1 * accel(noisy)


def test_mask_interpolation_moves_the_shape():
    """A.4 (i): interpolating two overlapping offsets lands between them."""
    left, right = _square(80, 10, 10), _square(80, 24, 10)
    assert np.array_equal(arm_mask.interpolate_masks(left, right, 0.0), left)
    assert np.array_equal(arm_mask.interpolate_masks(left, right, 1.0), right)
    mid = arm_mask.interpolate_masks(left, right, 0.5)
    columns = np.flatnonzero(mid.any(axis=0))
    assert 10 < columns.mean() < 44          # between x=10..30 and x=24..44
    # Disjoint neighbours must not interpolate to an empty mask (step 4 inpaints it).
    assert arm_mask.interpolate_masks(_square(80, 0, 0), _square(80, 60, 60), 0.5).any()


def test_short_gaps_only():
    """A.4 (i) fills gaps of at most 3 frames and leaves longer ones alone."""
    masks = [_square(40, 5, 5), None, None, _square(40, 8, 5),
             None, None, None, None, _square(40, 12, 5)]
    stats = arm_mask.MaskStats()
    arm_mask.fill_short_gaps(masks, stats)
    assert stats.interpolated == [1, 2]
    assert [m is None for m in masks] == [False, False, False, False,
                                         True, True, True, True, False]


def test_small_area_frames_are_replaced():
    """A.4 (ii): a frame under half the local median area takes its neighbour's mask."""
    masks = [_square(80, 5, 5, side=20) for _ in range(9)]
    masks[4] = _square(80, 5, 5, side=5)      # 25 px vs a median of 400
    stats = arm_mask.MaskStats()
    arm_mask.replace_small_masks(masks, stats)
    assert stats.replaced == [4]
    assert masks[4].sum() == 400


def test_close_fills_small_holes():
    """A.4 (iii): the 5x5 elliptical close removes pinholes but keeps the shape."""
    mask = _square(40, 10, 10)
    mask[15, 15] = False
    closed = arm_mask.close_mask(mask)
    assert closed[15, 15]
    assert closed.sum() == 400


def test_base_grid_matches_appendix():
    """A.4's grid is 7 x 7 x 5 positions and 3 x 5 x 3 orientations, scaled by reach."""
    reach = 2.0
    origins = candidates.positions(np.zeros(3), reach, sign=1.0)
    assert origins.shape == (7 * 7 * 5, 3)
    assert candidates.orientations().shape == (3 * 5 * 3, 3, 3)
    lateral = sorted({round(float(p[0]) / reach, 3) for p in origins})
    assert lateral == sorted(config.BASE_LATERAL_FACTORS)
    flipped = candidates.positions(np.zeros(3), reach, sign=-1.0)
    assert np.allclose(sorted(flipped[:, 0]), sorted(-origins[:, 0]))
    # Every orientation is a proper rotation, and the nominal frame looks forward.
    for rot in candidates.orientations():
        assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(rot), 1.0)


def test_base_pruning_and_score():
    """A.4 discards bases that are too near the camera, too far, or too close."""
    reach = 1.0
    traj = np.array([[0.0, 0.0, 1.0]])
    # The camera moves, and A.4's minimum distance has to hold for the whole clip, so
    # pruning is against the closest approach rather than a single camera position.
    cameras = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.4]])
    origins = np.array([[0.0, 0.0, 0.1],      # 0.1 m from the camera at frame 0
                        [0.0, 0.0, 1.05],     # 0.05 m from the trajectory
                        [0.0, 0.0, 0.05],     # both of the above
                        [0.0, 0.0, 0.7]])     # keeper, 0.3 m from the nearer camera
    keep, stats = candidates.prune(origins, traj, reach, cameras)
    assert keep.tolist() == [False, False, False, True]
    assert stats["rejected_camera_distance"] == 2 and stats["rejected_too_close"] == 1
    # 0.5 m from the frame-0 camera, but only 0.1 m from where it moves to.
    assert not candidates.prune(np.array([[0.0, 0.0, 0.5]]), traj, reach, cameras)[0][0]
    far = np.array([[0.0, 0.0, -0.95]])       # 1.95 m > 0.9 x reach
    assert not candidates.prune(far, traj, reach, cameras)[0][0]
    assert np.isclose(candidates.score(1.0, 0.65), 1.0)
    assert np.isclose(candidates.score(0.5, 0.85), 0.5 - 5.0 * 0.2)


def test_keyframes_cover_the_extremes():
    """The keyframe set is capped at 20 and always contains the farthest pair."""
    n = 100
    positions = np.zeros((n, 3))
    positions[:, 0] = np.linspace(0.0, 1.0, n)
    quats = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))
    keys = keyframes.select(positions, quats, np.ones(n, bool))
    assert len(keys) == config.IK_MAX_KEYFRAMES
    assert keys[0] == 0 and keys[-1] == n - 1
    assert len(set(keys.tolist())) == len(keys)
    short = keyframes.select(positions[:5], quats[:5], np.ones(5, bool))
    assert short.tolist() == [0, 1, 2, 3, 4]


def test_robot_specs_reproduce_table_3():
    """The derived TCP reproduces Table 3's reach and stroke; a flange TCP would not."""
    if not config.MENAGERIE_ROOT.exists():
        print("   (skipped: no mujoco_menagerie checkout)")
        return
    for name in robots.available():
        robot = robots.load(name)
        measured = robot.measure_reach(n_samples=20_000)
        assert abs(measured - robot.spec.reach_m) < 0.02 * robot.spec.reach_m, name
        # x points from the wrist towards the pads, z along the jaw travel.
        rot = Rotation.from_quat(np.roll(robot.tcp_quat, -1)).as_matrix()
        assert rot[:, 0] @ robot.tcp_offset > 0.9 * np.linalg.norm(robot.tcp_offset), name
        assert np.isclose(np.linalg.det(rot), 1.0)
        # Commanding the two ends of Table 3's stroke must move the jaws by exactly
        # that stroke, and widths outside it clamp to the ends. Pads are located by
        # their geom centres, so only the travel is compared, not the absolute gap.
        # In between the map is linear in *joint* angle: a slider gripper is then
        # linear in opening too, but YAM's linkage is 3 mm off at mid-stroke.
        low, high = robot.spec.width_range_m
        closed, middle, opened = (robot.measure_opening(w)
                                  for w in (low, 0.5 * (low + high), high))
        assert abs((opened - closed) - (high - low)) < 2e-3, name
        assert closed < middle < opened, name
        assert abs(middle - 0.5 * (closed + opened)) < 4e-3, name
        assert np.allclose(robot.width_to_qpos(low - 1.0), robot.width_to_qpos(low))
        assert np.allclose(robot.width_to_qpos(high + 1.0), robot.width_to_qpos(high))


def test_ik_recovers_a_forward_kinematics_pose():
    """IK must find a pose the robot demonstrably has: one taken from its own FK."""
    if not config.MENAGERIE_ROOT.exists():
        print("   (skipped: no mujoco_menagerie checkout)")
        return
    for name in robots.available():
        robot = robots.load(name)
        solver = ik.IKSolver(robot)
        rng = np.random.default_rng(0)
        joints = [mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_JOINT, j)
                  for j in robot.spec.arm_joints]
        low, high = robot.model.jnt_range[joints, 0], robot.model.jnt_range[joints, 1]
        solved = 0
        for _ in range(5):
            # Stay near the middle of the joint ranges to avoid singular poses.
            arm = 0.5 * (low + high) + 0.15 * (high - low) * rng.uniform(-1, 1, len(joints))
            position, rotation = robot.tcp_pose(robot.qpos_with(arm, 0.02))
            result = solver.solve(position, rotation, 0.02)
            solved += result.feasible
        assert solved == 5, f"{name}: only {solved}/5 FK poses recovered"


def test_depth_compositing_follows_equation_9():
    """A.4: the arm body always wins; a gripper pixel only loses to nearer scene depth."""
    shape = (1, 4)
    rgb = np.full((*shape, 3), 200, np.uint8)
    background = np.zeros((*shape, 3), np.uint8)
    arm = np.array([[True, False, False, False]])
    gripper = np.array([[False, True, True, True]])
    #            arm    gripper: scene nearer   scene farther   nearer but in hand mask
    sim_depth = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    scene_depth = np.array([[0.1, 0.5, 2.0, 0.5]], dtype=np.float32)
    hand = np.array([[False, False, False, True]])
    out, counts = run_stage6.composite(
        scene_mod.Render(rgb, sim_depth, arm, gripper), background, scene_depth, hand)
    assert out[0, 0, 0] == 200, "arm body must be drawn even with the scene in front"
    assert out[0, 1, 0] == 0, "gripper behind the scene must be hidden"
    assert out[0, 2, 0] == 200, "gripper in front of the scene must be drawn"
    assert out[0, 3, 0] == 200, "inside the dilated hand mask the depth test is skipped"
    assert counts == {"arm_px": 1, "gripper_px": 3, "gripper_occluded_px": 1, "robot_px": 3}


def test_hand_mask_dilation_matches_appendix():
    """A.4 dilates the hand mask with a 5x5 kernel for one iteration before compositing."""
    import tempfile

    import cv2

    mask = np.zeros((21, 21), np.uint8)
    mask[10, 10] = 255
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "000000.png"
        cv2.imwrite(str(path), mask)
        dilated = run_stage6.hand_mask(path, mask.shape)
    assert dilated.sum() == config.COMPOSITE_HAND_DILATE_KERNEL ** 2
    assert dilated[8:13, 8:13].all()



def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"ok   {test.__name__}")
    print(f"\n{len(tests)} passed")


if __name__ == "__main__":
    main()

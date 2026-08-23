"""Inverse kinematics through mink, as specified in paper A.4.

"the mink IK solver, quadprog backend, 100 iterations, 1e-5 threshold". mink solves
a differential-IK QP per iteration, so three things the paper leaves out have to be
fixed here: the integration step (``IK_DT``), a low-priority regulariser that keeps
the QP well behaved near singularities (``IK_DAMPING_COST``), and what happens to the
gripper joints - they are not part of the task, so they are pinned to the commanded
opening after every step instead of being left to the solver.

The regulariser has to damp *velocities*, not pull the configuration towards a
posture: a 6-DOF arm has no null space, so a posture task trades pose error against
posture error and settles at an equilibrium that never reaches 1e-5. With
mink.PostureTask at cost 1e-3, ARX-L5 recovered 4 of 30 poses taken from its own
forward kinematics even when started from the exact answer; with DampingTask it
recovers all 30.

Targets arrive in the robot's base frame; the model's world frame is only the same
thing when the base body sits at the origin, so the base pose is composed in.
"""
from __future__ import annotations

import dataclasses

import mink
import mujoco
import numpy as np

from pipeline import config
from pipeline.robots import TCP_SITE, RobotModel


@dataclasses.dataclass
class IKResult:
    qpos: np.ndarray
    feasible: bool
    position_error_m: float
    orientation_error_rad: float
    iterations: int


class IKSolver:
    """Reusable solver for one robot model."""

    def __init__(self, robot: RobotModel):
        self.robot = robot
        self.configuration = mink.Configuration(robot.model)
        self.task = mink.FrameTask(TCP_SITE, "site", position_cost=1.0,
                                   orientation_cost=1.0, lm_damping=1.0)
        self.damping = mink.DampingTask(robot.model, config.IK_DAMPING_COST)
        self.limits = [mink.ConfigurationLimit(robot.model)]
        # Pose of the base body in the model's world frame; constant for a fixed base.
        data = mujoco.MjData(robot.model)
        data.qpos[:] = robot.home_qpos
        mujoco.mj_kinematics(robot.model, data)
        self.world_from_base = mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(data.xmat[robot.base].reshape(3, 3)),
            data.xpos[robot.base].copy())

    def solve(self, position: np.ndarray, rotation: np.ndarray, width: float,
              q_init: np.ndarray | None = None) -> IKResult:
        """Solve for a TCP target given in the robot base frame."""
        robot = self.robot
        target = self.world_from_base @ mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(rotation), position)
        self.task.set_target(target)

        q = robot.home_qpos.copy() if q_init is None else q_init.copy()
        q[robot.gripper_qpos] = robot.width_to_qpos(width)
        self.configuration.update(q)

        pos_err = ori_err = np.inf
        for step in range(1, config.IK_ITERS + 1):
            velocity = mink.solve_ik(self.configuration, [self.task, self.damping],
                                     config.IK_DT, config.IK_SOLVER, 1e-12,
                                     limits=self.limits)
            self.configuration.integrate_inplace(velocity, config.IK_DT)
            q = self.configuration.q.copy()
            q[robot.gripper_qpos] = robot.width_to_qpos(width)
            self.configuration.update(q)

            error = self.task.compute_error(self.configuration)
            pos_err = float(np.linalg.norm(error[:3]))
            ori_err = float(np.linalg.norm(error[3:]))
            if max(pos_err, ori_err) < config.IK_THRESHOLD:
                break
        feasible = max(pos_err, ori_err) < config.IK_THRESHOLD
        return IKResult(self.configuration.q.copy(), feasible, pos_err, ori_err, step)

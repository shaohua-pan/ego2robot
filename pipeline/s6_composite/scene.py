"""The MuJoCo scene stage ⑥ renders: both arms at their stage-⑤ base poses.

The world frame *is* the camera frame here, and the camera moves, so the bases are
re-placed on every frame. Stage ⑤ solves eq.(4) once in the world frame - the robot is
bolted to the scene, not to the wearer's head - and this module maps that pose through
the per-frame camera extrinsics before rendering. Attaching the arms with a static frame
and leaving them there instead makes the base ride along with the head, which slides it
through the scene by however far the wearer moves.

MuJoCo's camera looks down its own ``-z`` with ``+y`` up while OpenCV's looks down ``+z``
with ``+y`` down, hence the 180° turn about ``x``.

Each morphology is attached once per hand from a *freshly parsed* spec: attaching the
same spec object twice leaves MuJoCo with a contact-exclude list it cannot resolve
("incompatible id in exclude array").
"""
from __future__ import annotations

import dataclasses

import mujoco
import numpy as np

from pipeline import config
from pipeline.robots import TCP_SITE, RobotModel


@dataclasses.dataclass
class Render:
    """One frame of the robot, as eq.(9) needs it."""

    rgb: np.ndarray          # (H, W, 3) uint8
    depth: np.ndarray        # (H, W) float32, metres along the view axis
    arm: np.ndarray          # (H, W) bool: robot_mask & ~gripper_mask
    gripper: np.ndarray      # (H, W) bool

    @property
    def robot(self) -> np.ndarray:
        return self.arm | self.gripper


def _quat(rotation: np.ndarray) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.asarray(rotation, dtype=float).reshape(-1))
    return quat


class Scene:
    """Both arms in one model, rendered through the ego camera."""

    CAMERA = "ego"

    def __init__(self, robot: RobotModel, hands: list[str], intrins: np.ndarray,
                 size: tuple[int, int]):
        self.robot, self.hands = robot, list(hands)
        width, height = size
        _, fy, _, _ = intrins

        spec = mujoco.MjSpec()
        spec.visual.global_.offwidth, spec.visual.global_.offheight = width, height
        spec.visual.map.znear = config.RENDER_ZNEAR_FRACTION
        spec.worldbody.add_camera(
            name=self.CAMERA, pos=[0.0, 0.0, 0.0], quat=[0.0, 1.0, 0.0, 0.0],
            fovy=float(np.degrees(2.0 * np.arctan(height / (2.0 * fy)))))
        for hand in self.hands:
            child = mujoco.MjSpec.from_file(str(config.MENAGERIE_ROOT / robot.spec.mjcf))
            child.body(robot.spec.wrist_body).add_site(
                name=TCP_SITE, pos=robot.tcp_offset, quat=robot.tcp_quat)
            # The base pose changes every frame, so the attachment frame is left at the
            # origin and :meth:`place` writes the pose onto the attached root body.
            spec.attach(child, prefix=f"{hand}_", frame=spec.worldbody.add_frame())
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        # MuJoCo draws sites. The injected TCP site is a 5 mm sphere sitting between the
        # jaws, so left visible it lands in the colour frame, writes itself into the
        # depth buffer and shows up in the segmentation under a *site* id that collides
        # with the geom ids the masks are built from. Hiding every site group removes it
        # from all three passes; nothing else in this scene is a site.
        self.options = mujoco.MjvOption()
        self.options.sitegroup[:] = 0
        self.intrins = np.asarray(intrins, dtype=float)

        self._qpos = {hand: self._qpos_map(hand) for hand in self.hands}
        self._gripper_geoms = {hand: self._subtree_geoms(f"{hand}_{robot.spec.wrist_body}")
                               for hand in self.hands}
        self._tcp_site = {hand: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,
                                                  f"{hand}_{TCP_SITE}")
                          for hand in self.hands}
        self._root = {hand: self._root_body(hand) for hand in self.hands}

    def _root_body(self, hand: str) -> int:
        """The attached arm's own root body, i.e. the one parented to the world."""
        for body in range(1, self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
            if self.model.body_parentid[body] == 0 and name.startswith(f"{hand}_"):
                return body
        raise SystemExit(f"scene has no root body for {hand}")

    def place(self, bases: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        """Put each arm's base at a camera-frame pose, for one frame."""
        for hand, (position, rotation) in bases.items():
            root = self._root[hand]
            self.model.body_pos[root] = position
            self.model.body_quat[root] = _quat(rotation)

    def _qpos_map(self, hand: str) -> tuple[np.ndarray, np.ndarray]:
        """(source, destination) qpos addresses between one robot and the scene."""
        source, destination = [], []
        joints = list(self.robot.spec.arm_joints) + list(self.robot.spec.gripper_joints)
        for name, adr in zip(joints, np.concatenate([self.robot.arm_qpos,
                                                     self.robot.gripper_qpos])):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{hand}_{name}")
            if jid < 0:
                raise SystemExit(f"scene has no joint {hand}_{name}")
            source.append(int(adr))
            destination.append(int(self.model.jnt_qposadr[jid]))
        return np.array(source), np.array(destination)

    def _subtree_geoms(self, body_name: str) -> set[int]:
        """Geom ids belonging to ``body_name`` or anything below it.

        This is the paper's ``gripper_mask``: the end-effector assembly, i.e. the part
        that reaches into the scene and can be occluded by objects in it.
        """
        root = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if root < 0:
            raise SystemExit(f"scene has no body {body_name}")
        inside = set()
        for body in range(self.model.nbody):
            node = body
            while node > 0 and node != root:
                node = self.model.body_parentid[node]
            if node == root:
                inside.add(body)
        return {g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] in inside}

    def pose(self, qpos: dict[str, np.ndarray]) -> None:
        """Load one frame of stage-⑤ joint angles, per hand."""
        for hand, values in qpos.items():
            source, destination = self._qpos[hand]
            self.data.qpos[destination] = values[source]
        mujoco.mj_forward(self.model, self.data)

    def tcp_camera(self, hand: str) -> np.ndarray:
        """TCP position in the camera frame - the world frame - after :meth:`pose`."""
        return self.data.site_xpos[self._tcp_site[hand]].copy()

    def project(self, point: np.ndarray) -> tuple[int, int]:
        """Camera-frame point -> pixel, through the intrinsics stage ① estimated."""
        fx, fy, cx, cy = self.intrins
        return (int(round(fx * point[0] / point[2] + cx)),
                int(round(fy * point[1] / point[2] + cy)))

    def ray_depth(self, u: int, v: int) -> float:
        """Geometric z-depth of the frontmost geom along the ray through pixel (u, v).

        The camera sits at the world origin looking down ``+z``, so the ray starts there.
        This is the depth the rendering pass has to reproduce, computed by intersecting
        geometry instead of by rasterising it, which makes the comparison a check on the
        field of view, the principal point and the camera pose at once. ``nan`` when the
        ray misses the robot.

        The ray is restricted to the geom groups the renderer draws. The menagerie models
        carry visual meshes in group 2 and simplified collision primitives in group 3, and
        an unrestricted ray hits the collision proxy, which on ARX-L5 stands up to 0.2 m
        in front of the visual surface - a disagreement with the rasteriser that says
        nothing about the camera.
        """
        fx, fy, cx, cy = self.intrins
        direction = np.array([(u + 0.5 - cx) / fx, (v + 0.5 - cy) / fy, 1.0])
        direction /= np.linalg.norm(direction)
        geom = np.zeros(1, dtype=np.int32)
        distance = mujoco.mj_ray(self.model, self.data, np.zeros(3), direction,
                                 np.asarray(self.options.geomgroup, dtype=np.uint8),
                                 1, -1, geom)
        return float(distance * direction[2]) if distance >= 0.0 else float("nan")

    def render(self) -> Render:
        """Colour, depth and the two masks of eq.(9), in three rendering passes."""
        self.renderer.disable_depth_rendering()
        self.renderer.disable_segmentation_rendering()
        self.renderer.update_scene(self.data, camera=self.CAMERA, scene_option=self.options)
        rgb = self.renderer.render()[..., ::-1].copy()      # to BGR, as OpenCV wants

        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera=self.CAMERA, scene_option=self.options)
        depth = self.renderer.render().copy()

        self.renderer.disable_depth_rendering()
        self.renderer.enable_segmentation_rendering()
        self.renderer.update_scene(self.data, camera=self.CAMERA, scene_option=self.options)
        segmentation = self.renderer.render()[..., 0]       # geom id, -1 where empty

        robot = segmentation >= 0
        gripper = np.zeros_like(robot)
        for geoms in self._gripper_geoms.values():
            gripper |= np.isin(segmentation, list(geoms))
        return Render(rgb, depth, robot & ~gripper, gripper & robot)


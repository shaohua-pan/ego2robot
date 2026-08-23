"""Robot registry: one YAML file per morphology, plus the MuJoCo model it names.

The paper's Table 3 lists 15 morphologies by DOF, gripper stroke and kinematic
reach; MuJoCo Menagerie ships models for 14 of them. A spec only names bodies and
joints - every number the pipeline needs (the tool centre point, the grasp frame,
the gripper mapping) is derived from the model itself, so adding a morphology
means writing eleven lines of YAML rather than measuring anything by hand.

The derived TCP frame follows the same construction as the human grasp frame in
paper eq.(2)-(3): the origin sits between the two jaw pads, ``z`` runs along the
jaw axis, ``x`` points along the approach direction and ``y = z x x``. Retargeted
human poses can therefore be handed to IK unchanged.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import mujoco
import numpy as np
import yaml

from pipeline import config

TCP_SITE = "tcp"


@dataclasses.dataclass(frozen=True)
class RobotSpec:
    """What a morphology needs to declare. See pipeline/robots/panda.yaml."""

    name: str
    mjcf: str
    base_body: str
    wrist_body: str
    jaw_bodies: tuple[str, str]
    arm_joints: tuple[str, ...]
    gripper_joints: tuple[str, ...]
    gripper_signs: tuple[float, ...]
    gripper_range_mm: tuple[float, float]
    home_key: str
    reach_m: float

    @property
    def width_range_m(self) -> tuple[float, float]:
        return self.gripper_range_mm[0] / 1000.0, self.gripper_range_mm[1] / 1000.0


def available() -> list[str]:
    return sorted(p.stem for p in config.ROBOT_SPEC_DIR.glob("*.yaml"))


def load_spec(name: str) -> RobotSpec:
    path = config.ROBOT_SPEC_DIR / f"{name}.yaml"
    if not path.exists():
        raise SystemExit(f"unknown robot {name!r}; available: {', '.join(available())}")
    raw = yaml.safe_load(path.read_text())
    fields = {f.name for f in dataclasses.fields(RobotSpec)}
    unknown = set(raw) - fields
    if unknown:
        raise SystemExit(f"{path.name}: unknown keys {sorted(unknown)}")
    for key in ("jaw_bodies", "arm_joints", "gripper_joints", "gripper_signs",
                "gripper_range_mm"):
        raw[key] = tuple(raw[key])
    return RobotSpec(**raw)


def _subtree(model: mujoco.MjModel, root: int) -> set[int]:
    """``root`` and every body below it."""
    inside = set()
    for body in range(model.nbody):
        node = body
        while node > 0 and node != root:
            node = model.body_parentid[node]
        if node == root:
            inside.add(body)
    return inside


def _pad(model: mujoco.MjModel, data: mujoco.MjData, jaw: int, origin: np.ndarray,
         rot: np.ndarray, approach: np.ndarray, tol: float = 1e-3) -> np.ndarray:
    """Centre of the jaw's contact pad, expressed in the wrist frame.

    Taken as the geoms sitting farthest along the approach direction *anywhere in the
    jaw's subtree*: for a parallel jaw those are the surfaces that touch the object,
    which is what the human TCP of eq.(2) corresponds to. The subtree matters for
    linkage grippers - YAM carries its pads two bodies below the finger it hangs
    them from, and stopping at the finger puts the TCP 5 cm short of Table 3's reach.
    Pads are often several boxes at the same depth, so everything within ``tol`` of
    the farthest is averaged; picking one would offset the TCP sideways.
    """
    geoms = [g for g in range(model.ngeom)
             if model.geom_bodyid[g] in _subtree(model, jaw)]
    if not geoms:
        raise SystemExit(f"jaw body {jaw} has no geoms to locate the pad")
    local = np.array([rot.T @ (data.geom_xpos[g] - origin) for g in geoms])
    depth = local @ approach
    return local[depth >= depth.max() - tol].mean(axis=0)




class RobotModel:
    """A compiled MuJoCo model with a derived ``tcp`` site and its home pose."""

    def __init__(self, spec: RobotSpec, root: Path | None = None):
        self.spec = spec
        path = (root or config.MENAGERIE_ROOT) / spec.mjcf
        if not path.exists():
            raise SystemExit(f"missing MJCF {path}; see README 'Robot models'")
        base = mujoco.MjModel.from_xml_path(str(path))
        pos, quat = self._tcp_frame(base)
        mjspec = mujoco.MjSpec.from_file(str(path))
        mjspec.body(spec.wrist_body).add_site(name=TCP_SITE, pos=pos, quat=quat)
        self.model = mjspec.compile()
        self.data = mujoco.MjData(self.model)
        self.tcp_offset, self.tcp_quat = pos, quat

        self.site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
        self.base = self._body(spec.base_body)
        self.wrist = self._body(spec.wrist_body)
        self.jaws = [self._body(b) for b in spec.jaw_bodies]
        self.arm_qpos = np.array([self._joint_qposadr(j) for j in spec.arm_joints])
        self.gripper_qpos = np.array([self._joint_qposadr(j) for j in spec.gripper_joints])
        self.gripper_signs = np.asarray(spec.gripper_signs, dtype=float)
        limits = np.array([self._joint_range(j) for j in spec.gripper_joints])
        opening = self.gripper_signs > 0
        self._closed_qpos = np.where(opening, limits[:, 0], limits[:, 1])
        self._open_qpos = np.where(opening, limits[:, 1], limits[:, 0])
        key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, spec.home_key)
        if key < 0:
            raise SystemExit(f"{spec.name}: no keyframe {spec.home_key!r}")
        self.home_qpos = self.model.key_qpos[key].copy()

    def _body(self, name: str) -> int:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise SystemExit(f"{self.spec.name}: no body {name!r}")
        return bid

    def _joint_qposadr(self, name: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise SystemExit(f"{self.spec.name}: no joint {name!r}")
        return int(self.model.jnt_qposadr[jid])

    def _joint_range(self, name: str) -> np.ndarray:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return self.model.jnt_range[jid].copy()

    def _tcp_frame(self, base: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
        """Derive the TCP pose in the wrist frame from the jaw geometry."""
        spec = self.spec
        data = mujoco.MjData(base)
        key = mujoco.mj_name2id(base, mujoco.mjtObj.mjOBJ_KEY, spec.home_key)
        mujoco.mj_resetDataKeyframe(base, data, max(key, 0))
        mujoco.mj_kinematics(base, data)

        wrist = mujoco.mj_name2id(base, mujoco.mjtObj.mjOBJ_BODY, spec.wrist_body)
        rot_w = data.xmat[wrist].reshape(3, 3)
        jaws = [mujoco.mj_name2id(base, mujoco.mjtObj.mjOBJ_BODY, b) for b in spec.jaw_bodies]
        # The jaws hang off the wrist along the approach direction, so their mean
        # offset from the wrist origin gives that direction without needing a name.
        approach = rot_w.T @ (np.mean([data.xpos[j] for j in jaws], axis=0) - data.xpos[wrist])
        approach /= np.linalg.norm(approach)

        pads = [_pad(base, data, jaw, data.xpos[wrist], rot_w, approach) for jaw in jaws]
        tcp = np.mean(pads, axis=0)
        self._approach = approach

        # The jaw axis comes from the gripper joint rather than from the pad offsets:
        # pad geoms are not always centred on the jaw, the joint axis always is.
        jid = mujoco.mj_name2id(base, mujoco.mjtObj.mjOBJ_JOINT, spec.gripper_joints[0])
        jaw_axis = rot_w.T @ data.xmat[base.jnt_bodyid[jid]].reshape(3, 3) @ base.jnt_axis[jid]
        jaw_axis /= np.linalg.norm(jaw_axis)
        spread = pads[0] - pads[1]
        # A gripper parked closed has coincident pads; then only the joint axis is
        # meaningful, and its sign does not matter for a symmetric jaw.
        if np.linalg.norm(spread) > 1e-4:
            spread /= np.linalg.norm(spread)
            if abs(jaw_axis @ spread) < 0.9:
                raise SystemExit(f"{spec.name}: gripper joint axis {jaw_axis} is not "
                                 f"aligned with the jaw offset {spread}")
            jaw_axis = jaw_axis * np.sign(jaw_axis @ spread)

        # Same construction as eq.(3), with the approach direction playing the role
        # of the wrist-to-fingertip vector d.
        y = np.cross(jaw_axis, approach)
        y /= np.linalg.norm(y)
        rot = np.stack([np.cross(y, jaw_axis), y, jaw_axis], axis=1)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rot.reshape(-1))
        return tcp, quat

    def width_to_qpos(self, width: float) -> np.ndarray:
        """Map a gripper opening in metres onto the finger joints.

        Table 3's stroke is mapped affinely onto each finger joint's own range: the
        closed end of the stroke goes to the closed end of the joint range, the open
        end to the open end, with ``gripper_signs`` saying which way round that is.
        A plain ``w / 2`` is wrong for jaws whose pads are inset - ViperX travels
        21-57 mm per finger for a 15-87 mm opening, so halving the width would open
        it 27 mm too far - and it comes out identical for the ones that are 1:1
        (Panda, ARX-L5, Piper).
        """
        low, high = self.spec.width_range_m
        fraction = (float(np.clip(width, low, high)) - low) / max(high - low, 1e-9)
        return self._closed_qpos + fraction * (self._open_qpos - self._closed_qpos)

    def qpos_with(self, arm: np.ndarray, width: float | None = None) -> np.ndarray:
        """A full qpos vector: the home pose with the arm and gripper overwritten."""
        q = self.home_qpos.copy()
        q[self.arm_qpos] = arm
        if width is not None:
            q[self.gripper_qpos] = self.width_to_qpos(width)
        return q

    def tcp_pose(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward kinematics: TCP position and rotation in the robot base frame."""
        self.data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, self.data)
        rot_b = self.data.xmat[self.base].reshape(3, 3)
        pos = rot_b.T @ (self.data.site_xpos[self.site] - self.data.xpos[self.base])
        return pos, rot_b.T @ self.data.site_xmat[self.site].reshape(3, 3)

    def body_positions(self, qpos: np.ndarray) -> np.ndarray:
        """All body origins in the base frame, for the stage-5 skeleton overlay."""
        self.data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, self.data)
        rot_b = self.data.xmat[self.base].reshape(3, 3)
        return (self.data.xpos - self.data.xpos[self.base]) @ rot_b

    def base_radius(self) -> float:
        """Radial extent of the base body, used as the two-arm clearance in stage 5."""
        radii = [float(np.linalg.norm(self.model.geom_pos[g][:2]) + self.model.geom_rbound[g])
                 for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == self.base]
        return max(radii, default=0.0)


    def measure_opening(self, width: float) -> float:
        """Pad-to-pad distance the model actually reaches when commanded ``width``.

        The counterpart of :meth:`measure_reach` for Table 3's stroke column: the
        difference between the two ends of the stroke must be the stroke itself, which
        validates a spec's ``gripper_range_mm`` together with the affine map of
        :meth:`width_to_qpos` and would have caught the ``w / 2`` mapping that opened
        ViperX's inset pads 27 mm too far. Pads are located by their geom *centres*, so
        the absolute value runs 1-5 mm wide - one pad thickness - and only differences
        are meaningful.
        """
        self.data.qpos[:] = self.qpos_with(self.home_qpos[self.arm_qpos], width)
        mujoco.mj_kinematics(self.model, self.data)
        rot_tcp = np.zeros(9)
        mujoco.mju_quat2Mat(rot_tcp, self.tcp_quat)
        approach, jaw_axis = rot_tcp.reshape(3, 3)[:, 0], rot_tcp.reshape(3, 3)[:, 2]
        rot_w = self.data.xmat[self.wrist].reshape(3, 3)
        pads = [_pad(self.model, self.data, jaw, self.data.xpos[self.wrist], rot_w, approach)
                for jaw in self.jaws]
        return abs(float((pads[0] - pads[1]) @ jaw_axis))

    def measure_reach(self, n_samples: int = 200_000, seed: int = 0) -> float:
        """Largest TCP distance from the base over the arm's joint limits.

        Table 3's reach column is reproduced by this measurement to within ~1% for
        both morphologies checked so far, which is what validates the derived TCP:
        a TCP left at the flange comes out 7-16 cm short.
        """
        rng = np.random.default_rng(seed)
        jnt = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
               for j in self.spec.arm_joints]
        lo, hi = self.model.jnt_range[jnt, 0], self.model.jnt_range[jnt, 1]
        best = 0.0
        for sample in rng.uniform(lo, hi, size=(n_samples, len(jnt))):
            pos, _ = self.tcp_pose(self.qpos_with(sample))
            best = max(best, float(np.linalg.norm(pos)))
        return best


def load(name: str, root: Path | None = None) -> RobotModel:
    return RobotModel(load_spec(name), root)

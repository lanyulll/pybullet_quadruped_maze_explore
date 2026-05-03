"""
PyBullet demo for a 12-axis quadruped walking forward with gait planning
and inverse kinematics.

Run:
    python eight_axis_quadruped_control.py

Optional:
    python eight_axis_quadruped_control.py --direct
"""

from __future__ import annotations

import math
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

try:
    import pybullet as p
except ModuleNotFoundError as exc:
    raise SystemExit(
        "This demo requires pybullet. Install it with a Python version that has "
        "a prebuilt wheel, for example: python -m pip install pybullet"
    ) from exc


@dataclass(frozen=True)
class LegConfig:
    name: str
    hip_offset: tuple[float, float, float]
    phase: float


class EightAxisQuadrupedDemo:
    """A compact four-leg, 12-actuator PyBullet walking example."""

    LEGS = (
        LegConfig("front_left", (0.26, 0.13, 0.0), 0.0),
        LegConfig("front_right", (0.26, -0.13, 0.0), 0.5),
        LegConfig("rear_left", (-0.26, 0.13, 0.0), 0.5),
        LegConfig("rear_right", (-0.26, -0.13, 0.0), 0.0),
    )

    def __init__(self, gui: bool = True) -> None:
        self.gui = gui
        self.body_id: int | None = None
        self.foot_links: dict[str, int] = {}
        self.leg_joints: dict[str, tuple[int, ...]] = {}
        self.lower_limits: list[float] = []
        self.upper_limits: list[float] = []
        self.joint_ranges: list[float] = []
        self.rest_poses: list[float] = []
        self.ik_index_by_joint: dict[int, int] = {}

        self.nominal_stance_z = -0.385
        self.step_length = 0.5
        self.step_height = 0.15
        self.cycle_time = 0.15
        self.kp = 0.05
        self.kd = 0.75
        self.max_torque = 10.0
        self.desired_base_height = 0.42
        self.foot_radius = 0.035
        self.hold_stance_z = -0.32
        self.hold_settle_time = 0.2
        self.hold_elapsed = 0.0
        self.hold_start_feet_body: dict[str, tuple[float, float, float]] | None = None
    def load_robot(self) -> None:
        urdf_path = self._write_robot_urdf()
        self.body_id = p.loadURDF(
            str(urdf_path),
            basePosition=(0.0, 0.0, 0.46),
            baseOrientation=p.getQuaternionFromEuler((0.0, 0.0, 0.0)),
            flags=p.URDF_USE_INERTIA_FROM_FILE,
        )

        assert self.body_id is not None
        p.changeDynamics(self.body_id, -1, linearDamping=0.04, angularDamping=0.04)

        for joint_id in range(p.getNumJoints(self.body_id)):
            info = p.getJointInfo(self.body_id, joint_id)
            joint_name = info[1].decode("utf-8")
            link_name = info[12].decode("utf-8")
            p.changeDynamics(
                self.body_id,
                joint_id,
                lateralFriction=1.3,
                spinningFriction=0.02,
                rollingFriction=0.01,
            )

            if joint_name.endswith("_hip_abduction_joint"):
                leg_name = joint_name.removesuffix("_hip_abduction_joint")
                hip_pitch_joint = self._find_joint(f"{leg_name}_hip_pitch_joint")
                knee_joint = self._find_joint(f"{leg_name}_knee_joint")
                self.leg_joints[leg_name] = (joint_id, hip_pitch_joint, knee_joint)

            if link_name.endswith("_foot"):
                leg_name = link_name.removesuffix("_foot")
                self.foot_links[leg_name] = joint_id

        self._cache_joint_limits()
        self._initialize_stand_pose(seconds=1.0)

    def walk_forward(self, sim_t: float) -> None:
        """Move forward with a diagonal trot gait."""
        self._reset_hold_transition()
        self._step_controller(sim_t, self._forward_foot_target)

    def walk_left(self, sim_t: float) -> None:
        """Move left with the same trot timing as forward walking."""
        self.walk_lateral(sim_t, "left")

    def walk_right(self, sim_t: float) -> None:
        """Move right with the same trot timing as forward walking."""
        self.walk_lateral(sim_t, "right")

    def walk_lateral(self, sim_t: float, direction: str) -> None:
        """Move sideways by swinging the feet along the body y axis."""
        if direction not in ("left", "right"):
            raise ValueError('direction must be "left" or "right"')

        self._reset_hold_transition()
        self._step_controller(
            sim_t,
            lambda leg, t: self._lateral_foot_target(leg, t, direction),
        )

    def walk_toward_target(
        self,
        target_xy: tuple[float, float],
        sim_t: float,
        stop_radius: float = 0.18,
    ) -> bool:
        """Walk forward until the base is close to target_xy, then hold still.

        The caller should make the robot face the target before calling this
        method.  Returns True once the robot is close enough and the controller
        has switched from gait generation to standing hold.
        """
        if self.body_id is None:
            raise RuntimeError("Call load_robot() before walking.")

        base_pos, _ = p.getBasePositionAndOrientation(self.body_id)
        dx = target_xy[0] - base_pos[0]
        dy = target_xy[1] - base_pos[1]
        if math.hypot(dx, dy) <= stop_radius:
            self.hold_still()
            return True

        self.walk_forward(sim_t)
        return False

    def hold_still(self) -> None:
        """Move all feet in straight lines to the points below their hips."""
        self._settle_feet_under_hips()

    def turn(self, sim_t: float, direction: str = "left") -> None:
        """Turn in place around the robot's current XY position.

        Args:
            sim_t: Simulation time from the run loop.
            direction: "left" or "right".
        """
        if direction not in ("left", "right"):
            raise ValueError('direction must be "left" or "right"')

        self._reset_hold_transition()
        self._step_controller(
            sim_t,
            lambda leg, t: self._turn_foot_target(leg, t, direction),
        )

    def _step_controller(
        self,
        sim_t: float,
        foot_target_function: Callable[[LegConfig, float], tuple[float, float, float]],
    ) -> None:
        assert self.body_id is not None
        base_pos, base_orn = p.getBasePositionAndOrientation(self.body_id)
        yaw = p.getEulerFromQuaternion(base_orn)[2]
        yaw_rot = p.getMatrixFromQuaternion(p.getQuaternionFromEuler((0.0, 0.0, yaw)))

        for leg in self.LEGS:
            foot_body = foot_target_function(leg, sim_t)
            foot_world = self._transform_body_to_world(base_pos, yaw_rot, foot_body)
            foot_world = (
                foot_world[0],
                foot_world[1],
                max(self.foot_radius, foot_world[2]),
            )
            joint_targets = p.calculateInverseKinematics(
                self.body_id,
                self.foot_links[leg.name],
                foot_world,
                lowerLimits=self.lower_limits,
                upperLimits=self.upper_limits,
                jointRanges=self.joint_ranges,
                restPoses=self.rest_poses,
                maxNumIterations=40,
                residualThreshold=1e-4,
            )
            for joint_id in self.leg_joints[leg.name]:
                target = joint_targets[self.ik_index_by_joint[joint_id]]
                p.setJointMotorControl2(
                    self.body_id,
                    joint_id,
                    p.POSITION_CONTROL,
                    targetPosition=target,
                    positionGain=self.kp,
                    velocityGain=self.kd,
                    force=self.max_torque,
                )

    def _forward_foot_target(
        self, leg: LegConfig, sim_t: float
    ) -> tuple[float, float, float]:
        return self._gait_foot_target(leg, sim_t, stride_scale=1)

    def _lateral_foot_target(
        self, leg: LegConfig, sim_t: float, direction: str
    ) -> tuple[float, float, float]:
        direction_sign = 1.0 if direction == "left" else -1.0
        return self._sideways_gait_foot_target(
            leg,
            sim_t,
            stride_scale=0.3*direction_sign,
        )

    def _turn_foot_target(
        self, leg: LegConfig, sim_t: float, direction: str
    ) -> tuple[float, float, float]:
        phase = ((sim_t / (self.cycle_time)) + leg.phase) % 1.0
        x0, y0, _ = leg.hip_offset
        radius = max(math.hypot(x0, y0), 1e-6)
        yaw_sign = 1.0 if direction == "left" else -1.0
        tangent_x = -y0 / radius * yaw_sign
        tangent_y = x0 / radius * yaw_sign
        turn_step = self.step_length * 0.3

        if phase < 0.5:
            swing = phase / 0.5
            offset = -turn_step / 2.0 + turn_step * swing
            z = self.nominal_stance_z + self.step_height * math.sin(math.pi * swing)
        else:
            stance = (phase - 0.5) / 0.5
            offset = turn_step / 2.0 - turn_step * stance
            z = self.nominal_stance_z

        return (x0 + tangent_x * offset, y0 + tangent_y * offset, z)

    def _gait_foot_target(
        self, leg: LegConfig, sim_t: float, stride_scale: float
    ) -> tuple[float, float, float]:
        phase = ((sim_t / self.cycle_time) + leg.phase) % 1.0
        x0, y0, _ = leg.hip_offset
        step_length = self.step_length * stride_scale

        if phase < 0.5:
            # Swing: move the foot forward through the air.
            swing = phase / 0.5
            x = x0 - step_length / 2.0 + step_length * swing
            z = self.nominal_stance_z + self.step_height * math.sin(math.pi * swing)
        else:
            # Stance: keep the foot on the ground and let it travel backward
            # relative to the body, which produces forward body motion.
            stance = (phase - 0.5) / 0.5
            x = x0 + step_length / 2.0 - step_length * stance
            z = self.nominal_stance_z

        return (x, y0, z)

    def _sideways_gait_foot_target(
        self, leg: LegConfig, sim_t: float, stride_scale: float
    ) -> tuple[float, float, float]:
        phase = ((sim_t / self.cycle_time) + leg.phase) % 1.0
        x0, y0, _ = leg.hip_offset
        step_length = self.step_length * stride_scale

        if phase < 0.5:
            # Swing: move the foot sideways through the air.
            swing = phase / 0.5
            y = y0 - step_length / 2.0 + step_length * swing
            z = self.nominal_stance_z + self.step_height * math.sin(math.pi * swing)
        else:
            # Stance: keep the foot on the ground and let it travel opposite
            # the commanded body motion.
            stance = (phase - 0.5) / 0.5
            y = y0 + step_length / 2.0 - step_length * stance
            z = self.nominal_stance_z

        return (x0, y, z)

    def _stand_pose(self, seconds: float) -> None:
        assert self.body_id is not None
        for _ in range(int(seconds * 240)):
            self.hold_still()
            p.stepSimulation()
            if self.gui:
                time.sleep(1.0 / 240.0)

    def _initialize_stand_pose(self, seconds: float) -> None:
        assert self.body_id is not None
        self._reset_hold_transition()
        self._reset_to_nominal_stand_pose()
        for _ in range(int(seconds * 240)):
            self._stand_controller(position_gain=0.25, velocity_gain=0.08)
            p.stepSimulation()
            if self.gui:
                time.sleep(1.0 / 240.0)
        self._reset_hold_transition()

    def _reset_to_nominal_stand_pose(self) -> None:
        assert self.body_id is not None
        base_pos, base_orn = p.getBasePositionAndOrientation(self.body_id)
        yaw = p.getEulerFromQuaternion(base_orn)[2]
        yaw_rot = p.getMatrixFromQuaternion(
            p.getQuaternionFromEuler((0.0, 0.0, yaw))
        )

        for leg in self.LEGS:
            foot_world = self._transform_body_to_world(
                base_pos,
                yaw_rot,
                (leg.hip_offset[0], leg.hip_offset[1], self.nominal_stance_z),
            )
            foot_world = (
                foot_world[0],
                foot_world[1],
                max(self.foot_radius, foot_world[2]),
            )
            targets = p.calculateInverseKinematics(
                self.body_id,
                self.foot_links[leg.name],
                foot_world,
                lowerLimits=self.lower_limits,
                upperLimits=self.upper_limits,
                jointRanges=self.joint_ranges,
                restPoses=self.rest_poses,
                maxNumIterations=80,
                residualThreshold=1e-4,
            )
            for joint_id in self.leg_joints[leg.name]:
                joint_target = targets[self.ik_index_by_joint[joint_id]]
                p.resetJointState(self.body_id, joint_id, joint_target, targetVelocity=0.0)

        p.resetBaseVelocity(self.body_id, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    def _stand_controller(
        self,
        position_gain: float = 0.9,
        velocity_gain: float = 0.1,
    ) -> None:
        assert self.body_id is not None
        base_pos, base_orn = p.getBasePositionAndOrientation(self.body_id)
        yaw = p.getEulerFromQuaternion(base_orn)[2]
        yaw_rot = p.getMatrixFromQuaternion(
            p.getQuaternionFromEuler((0.0, 0.0, yaw))
        )
        for leg in self.LEGS:
            foot_world = self._transform_body_to_world(
                base_pos,
                yaw_rot,
                (leg.hip_offset[0], leg.hip_offset[1], self.nominal_stance_z),
            )
            foot_world = (
                foot_world[0],
                foot_world[1],
                max(self.foot_radius, foot_world[2]),
            )
            targets = p.calculateInverseKinematics(
                self.body_id,
                self.foot_links[leg.name],
                foot_world,
                lowerLimits=self.lower_limits,
                upperLimits=self.upper_limits,
                jointRanges=self.joint_ranges,
                restPoses=self.rest_poses,
                maxNumIterations=40,
            )
            for joint_id in self.leg_joints[leg.name]:
                target = targets[self.ik_index_by_joint[joint_id]]
                p.setJointMotorControl2(
                    self.body_id,
                    joint_id,
                    p.POSITION_CONTROL,
                    targetPosition=target,
                    positionGain=position_gain,
                    velocityGain=velocity_gain,
                    force=self.max_torque,
                )

    def _reset_hold_transition(self) -> None:
        self.hold_elapsed = 0.0
        self.hold_start_feet_body = None

    def _settle_feet_under_hips(self) -> None:
        assert self.body_id is not None
        if self.hold_start_feet_body is None:
            self.hold_start_feet_body = self._current_feet_body_positions()
            self.hold_elapsed = 0.0

        duration = max(self.hold_settle_time, 1e-6)
        alpha = min(1.0, self.hold_elapsed / duration)
        feet_body: dict[str, tuple[float, float, float]] = {}
        for leg in self.LEGS:
            start = self.hold_start_feet_body[leg.name]
            target = (leg.hip_offset[0], leg.hip_offset[1], self.hold_stance_z)
            feet_body[leg.name] = (
                start[0] + (target[0] - start[0]) * alpha,
                start[1] + (target[1] - start[1]) * alpha,
                start[2] + (target[2] - start[2]) * alpha,
            )

        self._hold_feet_at_body_targets(feet_body)
        self.hold_elapsed = min(duration, self.hold_elapsed + 1.0 / 240.0)

    def _hold_feet_at_body_targets(
        self, feet_body: dict[str, tuple[float, float, float]]
    ) -> None:
        assert self.body_id is not None
        base_pos, base_orn = p.getBasePositionAndOrientation(self.body_id)
        yaw = p.getEulerFromQuaternion(base_orn)[2]
        yaw_rot = p.getMatrixFromQuaternion(
            p.getQuaternionFromEuler((0.0, 0.0, yaw))
        )

        for leg in self.LEGS:
            foot_body = feet_body[leg.name]
            foot_world = self._transform_body_to_world(base_pos, yaw_rot, foot_body)
            foot_world = (
                foot_world[0],
                foot_world[1],
                max(self.foot_radius, foot_world[2]),
            )
            targets = p.calculateInverseKinematics(
                self.body_id,
                self.foot_links[leg.name],
                foot_world,
                lowerLimits=self.lower_limits,
                upperLimits=self.upper_limits,
                jointRanges=self.joint_ranges,
                restPoses=self.rest_poses,
                maxNumIterations=40,
            )
            for joint_id in self.leg_joints[leg.name]:
                target_position = targets[self.ik_index_by_joint[joint_id]]
                p.setJointMotorControl2(
                    self.body_id,
                    joint_id,
                    p.POSITION_CONTROL,
                    targetPosition=target_position,
                    positionGain=0.45,
                    velocityGain=0.18,
                    force=self.max_torque,
                )

    def _current_feet_body_positions(self) -> dict[str, tuple[float, float, float]]:
        assert self.body_id is not None
        base_pos, base_orn = p.getBasePositionAndOrientation(self.body_id)
        yaw = p.getEulerFromQuaternion(base_orn)[2]
        yaw_rot = p.getMatrixFromQuaternion(
            p.getQuaternionFromEuler((0.0, 0.0, yaw))
        )
        feet: dict[str, tuple[float, float, float]] = {}
        for leg in self.LEGS:
            foot_world = p.getLinkState(self.body_id, self.foot_links[leg.name])[0]
            feet[leg.name] = self._transform_world_to_body(base_pos, yaw_rot, foot_world)
        return feet

    def _cache_joint_limits(self) -> None:
        assert self.body_id is not None
        self.lower_limits.clear()
        self.upper_limits.clear()
        self.joint_ranges.clear()
        self.rest_poses.clear()
        self.ik_index_by_joint.clear()

        for joint_id in range(p.getNumJoints(self.body_id)):
            info = p.getJointInfo(self.body_id, joint_id)
            joint_type = info[2]
            if joint_type not in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                continue

            lower = float(info[8])
            upper = float(info[9])
            if lower >= upper:
                lower, upper = -math.pi, math.pi
            self.ik_index_by_joint[joint_id] = len(self.lower_limits)
            self.lower_limits.append(lower)
            self.upper_limits.append(upper)
            self.joint_ranges.append(upper - lower)
            self.rest_poses.append(0.0)

    def _find_joint(self, name: str) -> int:
        assert self.body_id is not None
        for joint_id in range(p.getNumJoints(self.body_id)):
            if p.getJointInfo(self.body_id, joint_id)[1].decode("utf-8") == name:
                return joint_id
        raise ValueError(f"Joint not found: {name}")

    @staticmethod
    def _transform_body_to_world(
        base_pos: tuple[float, float, float],
        rot: tuple[float, ...],
        local: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        x = base_pos[0] + rot[0] * local[0] + rot[1] * local[1] + rot[2] * local[2]
        y = base_pos[1] + rot[3] * local[0] + rot[4] * local[1] + rot[5] * local[2]
        z = base_pos[2] + rot[6] * local[0] + rot[7] * local[1] + rot[8] * local[2]
        return (x, y, z)

    @staticmethod
    def _transform_world_to_body(
        base_pos: tuple[float, float, float],
        rot: tuple[float, ...],
        world: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        dx = world[0] - base_pos[0]
        dy = world[1] - base_pos[1]
        dz = world[2] - base_pos[2]
        x = rot[0] * dx + rot[3] * dy + rot[6] * dz
        y = rot[1] * dx + rot[4] * dy + rot[7] * dz
        z = rot[2] * dx + rot[5] * dy + rot[8] * dz
        return (x, y, z)

    @staticmethod
    def _write_robot_urdf() -> Path:
        urdf = _build_urdf()
        path = Path(tempfile.gettempdir()) / "eight_axis_quadruped.urdf"
        path.write_text(urdf, encoding="utf-8")
        return path


def _inertial(mass: float, ixx: float, iyy: float, izz: float) -> str:
    return f"""
    <inertial>
      <mass value="{mass}"/>
      <inertia ixx="{ixx}" ixy="0" ixz="0" iyy="{iyy}" iyz="0" izz="{izz}"/>
    </inertial>"""


def _leg_urdf(name: str, x: float, y: float) -> str:
    return f"""
  <link name="{name}_hip_abduction">
    {_inertial(0.08, 0.00012, 0.00012, 0.00012)}
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="0.055 0.055 0.045"/></geometry>
      <material name="leg_gray"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="0.05 0.05 0.04"/></geometry>
    </collision>
  </link>
  <joint name="{name}_hip_abduction_joint" type="revolute">
    <parent link="base"/>
    <child link="{name}_hip_abduction"/>
    <origin xyz="{x} {y} 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.65" upper="0.65" effort="18" velocity="8"/>
    <dynamics damping="0.04" friction="0.01"/>
  </joint>

  <link name="{name}_hip">
    {_inertial(0.16, 0.00035, 0.00035, 0.00035)}
    <visual>
      <origin xyz="0 0 -0.11" rpy="0 0 0"/>
      <geometry><box size="0.045 0.045 0.22"/></geometry>
      <material name="leg_gray"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.11" rpy="0 0 0"/>
      <geometry><box size="0.04 0.04 0.22"/></geometry>
    </collision>
  </link>
  <joint name="{name}_hip_pitch_joint" type="revolute">
    <parent link="{name}_hip_abduction"/>
    <child link="{name}_hip"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.2" upper="1.2" effort="18" velocity="8"/>
    <dynamics damping="0.04" friction="0.01"/>
  </joint>

  <link name="{name}_shank">
    {_inertial(0.12, 0.00025, 0.00025, 0.00025)}
    <visual>
      <origin xyz="0 0 -0.11" rpy="0 0 0"/>
      <geometry><box size="0.038 0.038 0.22"/></geometry>
      <material name="dark_leg"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.11" rpy="0 0 0"/>
      <geometry><box size="0.034 0.034 0.22"/></geometry>
    </collision>
  </link>
  <joint name="{name}_knee_joint" type="revolute">
    <parent link="{name}_hip"/>
    <child link="{name}_shank"/>
    <origin xyz="0 0 -0.22" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.8" upper="0.2" effort="18" velocity="8"/>
    <dynamics damping="0.04" friction="0.01"/>
  </joint>

  <link name="{name}_foot">
    {_inertial(0.04, 0.00005, 0.00005, 0.00005)}
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><sphere radius="0.035"/></geometry>
      <material name="foot_black"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><sphere radius="0.035"/></geometry>
    </collision>
  </link>
  <joint name="{name}_ankle_fixed" type="fixed">
    <parent link="{name}_shank"/>
    <child link="{name}_foot"/>
    <origin xyz="0 0 -0.22" rpy="0 0 0"/>
  </joint>
"""


def _build_urdf() -> str:
    legs = "\n".join(
        _leg_urdf(leg.name, leg.hip_offset[0], leg.hip_offset[1])
        for leg in EightAxisQuadrupedDemo.LEGS
    )
    return f"""<?xml version="1.0"?>
<robot name="eight_axis_quadruped">
  <material name="body_blue"><color rgba="0.1 0.35 0.75 1"/></material>
  <material name="leg_gray"><color rgba="0.55 0.58 0.62 1"/></material>
  <material name="dark_leg"><color rgba="0.15 0.16 0.18 1"/></material>
  <material name="foot_black"><color rgba="0.02 0.02 0.02 1"/></material>

  <link name="base">
    {_inertial(2.6, 0.035, 0.09, 0.11)}
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="0.62 0.24 0.12"/></geometry>
      <material name="body_blue"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="0.62 0.24 0.12"/></geometry>
    </collision>
  </link>

{legs}
</robot>
"""

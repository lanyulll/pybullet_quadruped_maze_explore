"""
PyBullet quadruped maze demo.

This file ports the MiniWorld maze demo to PyBullet while keeping the original
maze logic in env.viscnt_env.MazeEnv.  The RL model still outputs one of four grid
actions.  The PyBullet layer renders the maze, moving obstacles, and an
12-axis quadruped that turns toward the requested direction before walking
one logical grid cell.

Run:
    python run_demo.py

Optional:
    python run_demo.py --direct --duration 30
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np

try:
    import cv2
except ModuleNotFoundError as exc:
    raise SystemExit(
        "This demo requires opencv-python for RGB optical flow. Install it with: "
        "python -m pip install opencv-python"
    ) from exc

try:
    import pybullet as p
    import pybullet_data
except ModuleNotFoundError as exc:
    raise SystemExit(
        "This demo requires pybullet. Install it first, for example: "
        "python -m pip install pybullet"
    ) from exc

try:
    from stable_baselines3 import PPO
except ModuleNotFoundError:
    PPO = None

PROJECT_DIR = Path(__file__).resolve().parent

from env.viscnt_env import MazeEnv
from quadruped_control import EightAxisQuadrupedDemo


ACTION_TO_MOVE = {
    0: np.array([-1, 0], dtype=np.int32),
    1: np.array([1, 0], dtype=np.int32),
    2: np.array([0, -1], dtype=np.int32),
    3: np.array([0, 1], dtype=np.int32),
}

ACTION_TO_NAME = {
    0: "up",
    1: "down",
    2: "left",
    3: "right",
}

DIRECTION_TO_YAW = {
    "right": 0.0,
    "down": math.pi / 2.0,
    "left": math.pi,
    "up": -math.pi / 2.0,
}

DEFAULT_CELL_STOP_RADIUS = 1.0
GOAL_STOP_RADIUS = 1.3


@dataclass
class MovingObstacle:
    body_id: int
    center_pos: np.ndarray
    move_axis: np.ndarray
    amplitude: float
    speed: float
    offset: float = 0.0

    def __post_init__(self) -> None:
        self.center_pos = np.asarray(self.center_pos, dtype=np.float32)
        self.move_axis = np.asarray(self.move_axis, dtype=np.float32)
        norm = float(np.linalg.norm(self.move_axis))
        if norm > 0.0:
            self.move_axis = self.move_axis / norm
        direction = 1.0 if np.random.rand() > 0.5 else -1.0
        self.velocity = self.move_axis * self.speed * direction

    @property
    def current_pos(self) -> np.ndarray:
        return self.center_pos + self.move_axis * self.offset

    def update(self, dt: float) -> None:
        delta = float(np.dot(self.velocity, self.move_axis) * dt)
        self.offset += delta

        if self.offset > self.amplitude:
            self.offset = self.amplitude
            self.velocity = -self.velocity
        elif self.offset < -self.amplitude:
            self.offset = -self.amplitude
            self.velocity = -self.velocity

        pos = self.current_pos
        p.resetBasePositionAndOrientation(
            self.body_id,
            (float(pos[0]), float(pos[1]), float(pos[2])),
            p.getQuaternionFromEuler((0.0, 0.0, 0.0)),
        )


class PyBulletQuadrupedMazeEnv(gym.Env):
    """MazeEnv logic with a PyBullet quadruped renderer/executor."""

    metadata = {"render_modes": ["human", "direct"]}

    def __init__(
        self,
        size: int = 15,
        max_steps: int = 400,
        test_mode: int = 0,
        curriculum_levels: list[tuple[int, int]] | None = None,
        gui: bool = True,
        cell_size: float = 1.0,
        road_width: int = 5,
        wall_height: float = 1.2,
        obstacle_prob: float = 0.2,
        obstacle_speed_min: float = 0.5,
        obstacle_speed_max: float = 1.0,
        physics_dt: float = 1.0 / 120.0,
        wait_dt: float = 0.10,
        max_obstacle_wait_seconds: float = 8.0,
        camera_size: tuple[int, int] = (160, 120),
        topdown_debug: bool = True,
        auto_reset_scene: bool = True,
    ) -> None:
        super().__init__()
        self.logic_env = MazeEnv(
            size=size,
            max_steps=max_steps,
            test_mode=test_mode,
            curriculum_levels=curriculum_levels,
        )
        self.action_space = self.logic_env.action_space
        self.observation_space = self.logic_env.observation_space

        self.gui = gui
        self.cell_size = float(cell_size)
        self.road_width = int(road_width)
        self.wall_height = float(wall_height)
        self.obstacle_prob = float(obstacle_prob)
        self.obstacle_speed_min = float(obstacle_speed_min)
        self.obstacle_speed_max = float(obstacle_speed_max)
        self.physics_dt = float(physics_dt)
        self.wait_dt = float(wait_dt)
        self.max_obstacle_wait_seconds = float(max_obstacle_wait_seconds)
        self.camera_w, self.camera_h = camera_size
        self.topdown_debug = bool(topdown_debug)

        self.block = self.cell_size * self.road_width
        self.quadruped: EightAxisQuadrupedDemo | None = None
        self.moving_obstacles: list[MovingObstacle] = []
        self.static_body_ids: list[int] = []
        self.current_direction = "right"
        self.connected = False
        self.collision_last_state = False

        self.depth_threshold_s = self.block * 0.6
        self.flow_speed_threshold = 0.05
        self.collision_radius = self.block * 0.45
        self.danger_region = (0.3, 0.4, 0.7, 0.6)
        self.flow_roi = (0.2, 0.2, 0.8, 0.6)
        self.camera_near = 0.02
        self.camera_far = self.block * 2.0
        self.yaw_tolerance = 0.04
        self.lateral_position_tolerance = 0.2
        self.debug_action: int | None = None
        self.debug_target_pos: np.ndarray | None = None
        self.debug_can_move = False
        self.debug_step_number = 0
        self.debug_command = "idle"
        self.debug_window_name = "Maze top-down debug"
        self.flow_debug_window_name = "PyBullet optical flow debug"

        self._connect()
        if auto_reset_scene:
            self.reset()

    @property
    def grid(self) -> np.ndarray:
        return self.logic_env.grid

    @property
    def agent_pos(self) -> np.ndarray:
        return self.logic_env.agent_pos

    @property
    def goal_pos(self) -> np.ndarray:
        return self.logic_env.goal_pos

    def _connect(self) -> None:
        if self.connected and p.isConnected():
            return

        mode = p.GUI if self.gui else p.DIRECT
        p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0.0, 0.0, -9.81)
        p.setTimeStep(self.physics_dt)
        p.setPhysicsEngineParameter(numSolverIterations=80)
        self.connected = True

    def reset(self, seed=None, options=None):
        obs, info = self.logic_env.reset(seed=seed, options=options)
        self._create_pybullet_scene()
        self.debug_action = None
        self.debug_target_pos = None
        self.debug_can_move = False
        self.debug_step_number = 0
        self.debug_command = "idle"
        self._show_topdown_debug()
        return obs, info

    def step(self, action):
        action = int(np.asarray(action).item())
        direction_name = self._action_to_direction(action)
        old_pos = self.logic_env.agent_pos.copy()
        target_pos = old_pos + ACTION_TO_MOVE[action]
        can_move = self._can_logic_move(target_pos)
        self.debug_action = action
        self.debug_target_pos = target_pos.copy()
        self.debug_can_move = can_move
        self.debug_step_number = self.logic_env.steps + 1
        self._show_topdown_debug()
        self._ensure_facing_direction(direction_name)

        if can_move:
            self._correct_lateral_position(target_pos, direction_name)
            self._ensure_facing_direction(direction_name)
            self._safe_action_by_depth_flow(action)
            self._walk_one_cell_to_grid_position(target_pos)
        else:
            self.debug_command = f"blocked: {direction_name}"
            self._show_topdown_debug()

        obs, reward, terminated, truncated, info = self.logic_env.step(action)
        self._check_robot_obstacle_collision()
        self.debug_command = "idle"
        self._show_topdown_debug()
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        if self.topdown_debug:
            try:
                cv2.destroyWindow(self.debug_window_name)
                cv2.destroyWindow(self.flow_debug_window_name)
            except cv2.error:
                pass
        if self.connected and p.isConnected():
            p.disconnect()
        self.connected = False

    def _create_pybullet_scene(self) -> None:
        p.resetSimulation()
        p.setGravity(0.0, 0.0, -9.81)
        p.setTimeStep(self.physics_dt)
        p.setPhysicsEngineParameter(numSolverIterations=80)
        p.loadURDF("plane.urdf")
        self.static_body_ids.clear()
        self.moving_obstacles.clear()

        self._build_walls()
        self._build_goal()
        self._build_moving_obstacles()
        self._load_quadruped()
        self._sync_robot_to_grid(self.logic_env.agent_pos)
        self._update_camera()

    def _build_walls(self) -> None:
        half = self.block / 2.0
        wall_z = self.wall_height / 2.0
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=(half, half, wall_z),
        )
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=(half, half, wall_z),
            rgbaColor=(0.45, 0.45, 0.45, 1.0),
        )

        for r in range(self.grid.shape[0]):
            for c in range(self.grid.shape[1]):
                if self.grid[r, c] != 1:
                    continue

                x, y, _ = self._grid_to_world((r, c), z=0.0)
                body_id = p.createMultiBody(
                    baseMass=0.0,
                    baseCollisionShapeIndex=collision_shape,
                    baseVisualShapeIndex=visual_shape,
                    basePosition=(x, y, wall_z),
                )
                self.static_body_ids.append(body_id)

    def _build_goal(self) -> None:
        half = self.block * 0.18
        height = self.block * 0.10
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=(half, half, height),
        )
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=(half, half, height),
            rgbaColor=(0.0, 0.8, 0.15, 1.0),
        )
        x, y, _ = self._grid_to_world(self.logic_env.goal_pos, z=0.0)
        body_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=(x, y, height),
        )
        self.static_body_ids.append(body_id)

    def _build_moving_obstacles(self) -> None:
        half_x = self.block * 0.05
        half_y = self.block * 0.05
        half_z = self.block * 0.10
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=(half_x, half_y, half_z),
        )
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=(half_x, half_y, half_z),
            rgbaColor=(0.9, 0.05, 0.05, 1.0),
        )

        for r in range(self.grid.shape[0]):
            for c in range(self.grid.shape[1]):
                if self.grid[r, c] != 0:
                    continue
                if np.array_equal([r, c], self.logic_env.agent_pos):
                    continue
                if np.array_equal([r, c], self.logic_env.goal_pos):
                    continue

                move_axis = self._road_cross_axis(r, c)
                if move_axis is None:
                    continue
                if np.random.rand() > self.obstacle_prob:
                    continue

                x, y, _ = self._grid_to_world((r, c), z=0.0)
                center = np.array([x, y, half_z], dtype=np.float32)
                body_id = p.createMultiBody(
                    baseMass=0.0,
                    baseCollisionShapeIndex=collision_shape,
                    baseVisualShapeIndex=visual_shape,
                    basePosition=tuple(center),
                )
                speed = float(
                    np.random.uniform(
                        self.obstacle_speed_min,
                        self.obstacle_speed_max,
                    )
                )
                self.moving_obstacles.append(
                    MovingObstacle(
                        body_id=body_id,
                        center_pos=center,
                        move_axis=move_axis,
                        amplitude=self.block * 0.45,
                        speed=speed,
                    )
                )

    def _load_quadruped(self) -> None:
        self.quadruped = EightAxisQuadrupedDemo(gui=self.gui)
        self.quadruped.body_id = None
        self.quadruped.load_robot()

    def _grid_to_world(self, pos, z: float | None = None) -> tuple[float, float, float]:
        row, col = int(pos[0]), int(pos[1])
        x = col * self.block + self.block / 2.0
        y = row * self.block + self.block / 2.0
        if z is None:
            z = 0.46
        return (float(x), float(y), float(z))

    def _show_topdown_debug(self) -> None:
        if not self.topdown_debug:
            return

        grid = self.grid
        if grid is None:
            return

        rows, cols = grid.shape
        cell_px = max(18, min(42, int(720 / max(rows, cols))))
        panel_h = 100
        image = np.full(
            (rows * cell_px + panel_h, cols * cell_px, 3),
            245,
            dtype=np.uint8,
        )
        map_img = image[: rows * cell_px]

        for r in range(rows):
            for c in range(cols):
                x0 = c * cell_px
                y0 = r * cell_px
                x1 = x0 + cell_px
                y1 = y0 + cell_px
                color = (235, 235, 235) if grid[r, c] == 0 else (45, 45, 45)
                cv2.rectangle(map_img, (x0, y0), (x1, y1), color, thickness=-1)
                cv2.rectangle(map_img, (x0, y0), (x1, y1), (185, 185, 185), thickness=1)

        goal_r, goal_c = self.goal_pos
        self._draw_grid_marker(map_img, int(goal_r), int(goal_c), cell_px, (70, 180, 70), "G")

        agent_r, agent_c = self.agent_pos
        self._draw_grid_marker(map_img, int(agent_r), int(agent_c), cell_px, (220, 80, 60), "A")

        if self.debug_target_pos is not None:
            target_r, target_c = self.debug_target_pos
            if 0 <= target_r < rows and 0 <= target_c < cols:
                target_color = (60, 170, 240) if self.debug_can_move else (40, 40, 220)
                self._draw_grid_marker(
                    map_img,
                    int(target_r),
                    int(target_c),
                    cell_px,
                    target_color,
                    "T",
                )

            start = self._grid_cell_center_px(int(agent_r), int(agent_c), cell_px)
            delta = ACTION_TO_MOVE[int(self.debug_action)] if self.debug_action is not None else np.zeros(2)
            end = (
                int(start[0] + delta[1] * cell_px * 0.75),
                int(start[1] + delta[0] * cell_px * 0.75),
            )
            cv2.arrowedLine(map_img, start, end, (0, 140, 255), 3, tipLength=0.35)

        robot_px = self._robot_position_px(cell_px)
        if robot_px is not None:
            cv2.circle(map_img, robot_px, max(4, cell_px // 5), (255, 90, 30), -1)
            cv2.circle(map_img, robot_px, max(5, cell_px // 5 + 1), (20, 20, 20), 1)

        action_text = "action=none"
        if self.debug_action is not None:
            action_text = (
                f"step={self.debug_step_number} "
                f"action={ACTION_TO_NAME[int(self.debug_action)]} "
                f"target={self.debug_target_pos.tolist() if self.debug_target_pos is not None else None} "
                f"can_move={self.debug_can_move}"
            )
        info_y = rows * cell_px + 25
        cv2.putText(
            image,
            action_text,
            (8, info_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"logical_agent={self.agent_pos.tolist()} goal={self.goal_pos.tolist()} blue=robot",
            (8, info_y + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"command={self.debug_command}",
            (8, info_y + 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 70, 180),
            2,
            cv2.LINE_AA,
        )

        try:
            cv2.imshow(self.debug_window_name, image)
            cv2.waitKey(1)
        except cv2.error as exc:
            print(f"[TopDownDebug] disabled because OpenCV cannot open a window: {exc}")
            self.topdown_debug = False

    def _draw_grid_marker(
        self,
        image: np.ndarray,
        row: int,
        col: int,
        cell_px: int,
        color: tuple[int, int, int],
        label: str,
    ) -> None:
        center = self._grid_cell_center_px(row, col, cell_px)
        radius = max(5, cell_px // 3)
        cv2.circle(image, center, radius, color, -1)
        cv2.circle(image, center, radius, (25, 25, 25), 1)
        cv2.putText(
            image,
            label,
            (center[0] - cell_px // 7, center[1] + cell_px // 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.4, cell_px / 48.0),
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    def _grid_cell_center_px(self, row: int, col: int, cell_px: int) -> tuple[int, int]:
        return (int((col + 0.5) * cell_px), int((row + 0.5) * cell_px))

    def _robot_position_px(self, cell_px: int) -> tuple[int, int] | None:
        if self.quadruped is None or self.quadruped.body_id is None:
            return None
        pos, _ = p.getBasePositionAndOrientation(self.quadruped.body_id)
        col = float(pos[0]) / self.block
        row = float(pos[1]) / self.block
        return (int(col * cell_px), int(row * cell_px))

    def _is_free(self, r: int, c: int) -> bool:
        if not (0 <= r < self.grid.shape[0] and 0 <= c < self.grid.shape[1]):
            return False
        return self.grid[r, c] == 0

    def _can_logic_move(self, pos: np.ndarray) -> bool:
        return self._is_free(int(pos[0]), int(pos[1]))

    def _road_cross_axis(self, r: int, c: int) -> np.ndarray | None:
        up = self._is_free(r - 1, c)
        down = self._is_free(r + 1, c)
        left = self._is_free(r, c - 1)
        right = self._is_free(r, c + 1)

        vertical = up and down and not left and not right
        horizontal = left and right and not up and not down

        if vertical:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if horizontal:
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return None

    def _action_to_direction(self, action: int) -> str:
        return ACTION_TO_NAME[int(action)]

    def _needs_turn(self, direction_name: str) -> bool:
        return direction_name != self.current_direction

    def _ensure_facing_direction(self, direction_name: str) -> None:
        target_yaw = DIRECTION_TO_YAW[direction_name]
        if self._needs_turn(direction_name) or abs(self._yaw_error(target_yaw)) >= self.yaw_tolerance:
            self._turn_to_direction(direction_name)

    def _turn_to_direction(self, direction_name: str) -> None:
        target_yaw = DIRECTION_TO_YAW[direction_name]
        turn_side = self._turn_side_to_target(target_yaw)
        sim_t = 0.0
        self.debug_command = f"turn {turn_side} -> {direction_name}"

        while True:
            self._step_world_without_moving_robot()
            self._show_topdown_debug()
            assert self.quadruped is not None
            self.quadruped.turn(sim_t, turn_side)
            p.stepSimulation()
            self._sleep_if_gui()

            if abs(self._yaw_error(target_yaw)) < self.yaw_tolerance:
                break
            sim_t += self.physics_dt

        self.current_direction = direction_name

    def _correct_lateral_position(
        self,
        target_grid_pos: np.ndarray,
        direction_name: str,
    ) -> None:
        assert self.quadruped is not None and self.quadruped.body_id is not None
        target = np.array(self._grid_to_world(target_grid_pos), dtype=np.float32)
        yaw = DIRECTION_TO_YAW[direction_name]
        left_axis = np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float32)
        sim_t = 0.0

        while True:
            pos, _ = p.getBasePositionAndOrientation(self.quadruped.body_id)
            delta = target[:2] - np.array(pos[:2], dtype=np.float32)
            lateral_error = float(np.dot(delta, left_axis))
            if abs(lateral_error) <= self.lateral_position_tolerance:
                break

            self._step_world_without_moving_robot()
            lateral_direction = "left" if lateral_error > 0.0 else "right"
            self.debug_command = (
                f"lateral {lateral_direction} error={lateral_error:.2f}"
            )
            self._show_topdown_debug()
            self.quadruped.walk_lateral(sim_t, lateral_direction)
            p.stepSimulation()
            self._sleep_if_gui()
            sim_t += self.physics_dt

    def _turn_side_to_target(self, target_yaw: float) -> str:
        error = self._yaw_error(target_yaw)
        return "left" if error > 0.0 else "right"

    def _yaw_error(self, target_yaw: float) -> float:
        assert self.quadruped is not None and self.quadruped.body_id is not None
        _, orn = p.getBasePositionAndOrientation(self.quadruped.body_id)
        yaw = p.getEulerFromQuaternion(orn)[2]
        return math.atan2(math.sin(target_yaw - yaw), math.cos(target_yaw - yaw))

    def _wait_one_pybullet_step(self) -> None:
        wait_steps = max(1, int(self.wait_dt / self.physics_dt))
        for _ in range(wait_steps):
            self._step_world_without_moving_robot()
            self.debug_command = "wait / hold"
            self._show_topdown_debug()
            assert self.quadruped is not None
            self.quadruped.hold_still()
            p.stepSimulation()
            self._sleep_if_gui()

    def _step_world_without_moving_robot(self) -> None:
        for obstacle in self.moving_obstacles:
            obstacle.update(self.physics_dt)
        self._check_robot_obstacle_collision()
        self._update_camera()

    def _walk_one_cell_to_grid_position(self, grid_pos: np.ndarray) -> bool:
        assert self.quadruped is not None and self.quadruped.body_id is not None
        target = np.array(self._grid_to_world(grid_pos), dtype=np.float32)
        stop_radius = (
            GOAL_STOP_RADIUS
            if np.array_equal(grid_pos, self.logic_env.goal_pos)
            else DEFAULT_CELL_STOP_RADIUS
        )
        sim_t = 0.0

        while True:
            pos, _ = p.getBasePositionAndOrientation(self.quadruped.body_id)
            dist = float(np.linalg.norm(target[:2] - np.array(pos[:2])))
            if dist <= stop_radius:
                break

            self._step_world_without_moving_robot()
            self.debug_command = f"straight dist={dist:.2f}"
            self._show_topdown_debug()
            reached = self.quadruped.walk_toward_target(
                (float(target[0]), float(target[1])),
                sim_t,
                stop_radius=stop_radius,
            )
            p.stepSimulation()
            self._sleep_if_gui()
            if reached:
                break
            sim_t += self.physics_dt

        self._stand_and_stop()
        return True

    def _stand_and_stop(self, seconds: float = 0.35) -> None:
        assert self.quadruped is not None and self.quadruped.body_id is not None
        steps = max(1, int(seconds / self.physics_dt))
        for _ in range(steps):
            self._step_world_without_moving_robot()
            self.debug_command = "stop / hold_still"
            self._show_topdown_debug()
            self.quadruped.hold_still()
            p.stepSimulation()
            self._sleep_if_gui()

    def _sync_robot_to_grid(self, grid_pos: np.ndarray) -> None:
        assert self.quadruped is not None and self.quadruped.body_id is not None
        target = self._grid_to_world(grid_pos)
        yaw = DIRECTION_TO_YAW[self.current_direction]
        p.resetBasePositionAndOrientation(
            self.quadruped.body_id,
            target,
            p.getQuaternionFromEuler((0.0, 0.0, yaw)),
        )
        p.resetBaseVelocity(self.quadruped.body_id, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        self.quadruped._reset_hold_transition()
        self.quadruped._reset_to_nominal_stand_pose()
        self.quadruped._initialize_stand_pose(seconds=1.0)


    def _check_robot_obstacle_collision(self) -> bool:
        if self.quadruped is None or self.quadruped.body_id is None:
            return False

        pos, _ = p.getBasePositionAndOrientation(self.quadruped.body_id)
        robot_xy = np.array(pos[:2], dtype=np.float32)

        for i, obstacle in enumerate(self.moving_obstacles):
            dist = float(np.linalg.norm(robot_xy - obstacle.current_pos[:2]))
            if dist <= self.collision_radius:
                if not self.collision_last_state:
                    print(
                        f"[Collision] step={self.logic_env.steps}, "
                        f"obstacle_id={i}, "
                        f"robot=({robot_xy[0]:.2f}, {robot_xy[1]:.2f}), "
                        f"obstacle=({obstacle.current_pos[0]:.2f}, "
                        f"{obstacle.current_pos[1]:.2f}), dist={dist:.2f}"
                    )
                self.collision_last_state = True
                return True

        self.collision_last_state = False
        return False

    def _safe_action_by_depth_flow(self, action: int) -> None:
        direction_name = self._action_to_direction(action)
        start = time.time()

        while True:
            img1, depth1 = self._capture_action_camera(direction_name)
            has_close, _ = self._has_close_depth_points(depth1, self.depth_threshold_s)
            if not has_close:
                return

            self._wait_one_pybullet_step()
            img2, depth2 = self._capture_action_camera(direction_name)
            motion_info = self._estimate_obstacle_motion_from_rgb_pair(
                img1_rgb=img1,
                img2_rgb=img2,
                flow_threshold=self.flow_speed_threshold,
                direction_name=direction_name,
                depth1=depth1,
                depth2=depth2,
                depth_threshold=self.depth_threshold_s,
            )
            self._show_flow_debug_panel(img1, img2, motion_info, direction_name)

            if not motion_info["valid"]:
                return
            if not self._will_collide_by_optical_flow(motion_info, direction_name):
                return
            if time.time() - start >= self.max_obstacle_wait_seconds:
                print(
                    "[FlowWait] obstacle stayed in danger region too long; "
                    "continuing with the requested action."
                )
                return

    def _capture_action_camera(
        self,
        direction_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self.quadruped is not None and self.quadruped.body_id is not None
        base_pos, _ = p.getBasePositionAndOrientation(self.quadruped.body_id)
        yaw = DIRECTION_TO_YAW[direction_name]
        forward = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float32)
        eye = np.array(base_pos, dtype=np.float32) + np.array([0.0, 0.0, 0.16])
        target = eye + forward * self.block
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=tuple(eye),
            cameraTargetPosition=tuple(target),
            cameraUpVector=(0.0, 0.0, 1.0),
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=70.0,
            aspect=float(self.camera_w) / float(self.camera_h),
            nearVal=self.camera_near,
            farVal=self.camera_far,
        )
        _, _, rgba, depth_buffer, _ = p.getCameraImage(
            width=self.camera_w,
            height=self.camera_h,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL if self.gui else p.ER_TINY_RENDERER,
        )
        rgba = np.reshape(rgba, (self.camera_h, self.camera_w, 4)).astype(np.uint8)
        rgb = rgba[..., :3]
        depth_buffer = np.reshape(depth_buffer, (self.camera_h, self.camera_w))
        depth = self._linearize_depth(depth_buffer)
        return rgb, depth

    def _linearize_depth(self, depth_buffer: np.ndarray) -> np.ndarray:
        near = self.camera_near
        far = self.camera_far
        depth = far * near / (far - (far - near) * depth_buffer)
        depth = depth.astype(np.float32, copy=False)
        depth[~np.isfinite(depth)] = np.inf
        return depth

    def _has_close_depth_points(
        self,
        depth: np.ndarray,
        threshold: float,
    ) -> tuple[bool, np.ndarray]:
        mask = np.isfinite(depth) & (depth > 0) & (depth < threshold)
        return bool(np.any(mask)), mask

    def _depth_close_mask(self, depth: np.ndarray, depth_threshold: float) -> np.ndarray:
        return (np.isfinite(depth) & (depth > 0) & (depth < depth_threshold)).astype(bool)

    def _apply_depth_mask_to_rgb(
        self,
        img_rgb: np.ndarray,
        depth: np.ndarray,
        depth_threshold: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        mask = self._depth_close_mask(depth, depth_threshold)
        if mask.shape != img_rgb.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (img_rgb.shape[1], img_rgb.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        masked = np.zeros_like(img_rgb)
        masked[mask] = img_rgb[mask]
        return masked, mask

    def _estimate_obstacle_motion_from_rgb_pair(
        self,
        img1_rgb: np.ndarray,
        img2_rgb: np.ndarray,
        flow_threshold: float,
        direction_name: str | None = None,
        depth1: np.ndarray | None = None,
        depth2: np.ndarray | None = None,
        depth_threshold: float | None = None,
    ) -> dict[str, np.ndarray | bool]:
        masked_img1_rgb = img1_rgb
        masked_img2_rgb = img2_rgb
        close_mask = None

        if depth1 is not None and depth2 is not None and depth_threshold is not None:
            masked_img1_rgb, close_mask1 = self._apply_depth_mask_to_rgb(
                img1_rgb,
                depth1,
                depth_threshold,
            )
            masked_img2_rgb, close_mask2 = self._apply_depth_mask_to_rgb(
                img2_rgb,
                depth2,
                depth_threshold,
            )
            close_mask = close_mask1 | close_mask2

        roi_mask = self._flow_roi_mask(masked_img1_rgb.shape[:2])
        masked_img1_rgb = masked_img1_rgb.copy()
        masked_img2_rgb = masked_img2_rgb.copy()
        masked_img1_rgb[~roi_mask] = 0
        masked_img2_rgb[~roi_mask] = 0
        close_mask = roi_mask if close_mask is None else close_mask & roi_mask

        gray1 = cv2.cvtColor(masked_img1_rgb, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(masked_img2_rgb, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            gray1,
            gray2,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        moving_mask = mag > flow_threshold
        if close_mask is not None:
            moving_mask &= close_mask

        if not np.any(moving_mask):
            return {
                "valid": False,
                "mean_flow": np.array([0.0, 0.0], dtype=np.float32),
                "moving_mask": moving_mask,
                "flow": flow,
            }

        mean_flow = np.array(
            [
                np.mean(flow[..., 0][moving_mask]),
                np.mean(flow[..., 1][moving_mask]),
            ],
            dtype=np.float32,
        )
        return {
            "valid": True,
            "mean_flow": mean_flow,
            "moving_mask": moving_mask,
            "flow": flow,
        }

    def _danger_region_bounds(self, width: int, height: int) -> tuple[int, int, int, int]:
        x_min, y_min, x_max, y_max = self.danger_region
        return (
            int(round(width * x_min)),
            int(round(height * y_min)),
            int(round(width * x_max)),
            int(round(height * y_max)),
        )

    def _flow_roi_bounds(self, width: int, height: int) -> tuple[int, int, int, int]:
        x_min, y_min, x_max, y_max = self.flow_roi
        return (
            int(round(width * x_min)),
            int(round(height * y_min)),
            int(round(width * x_max)),
            int(round(height * y_max)),
        )

    def _flow_roi_mask(self, shape: tuple[int, int]) -> np.ndarray:
        height, width = shape
        x1, y1, x2, y2 = self._flow_roi_bounds(width, height)
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))
        mask = np.zeros((height, width), dtype=bool)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True
        return mask

    def _draw_danger_region(self, img: np.ndarray) -> np.ndarray:
        out = img.copy()
        height, width = out.shape[:2]
        x1, y1, x2, y2 = self._danger_region_bounds(width, height)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            out,
            "Danger",
            (x1 + 4, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
        return out

    def _draw_flow_roi(self, img: np.ndarray) -> np.ndarray:
        out = img.copy()
        height, width = out.shape[:2]
        x1, y1, x2, y2 = self._flow_roi_bounds(width, height)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(
            out,
            "Flow ROI",
            (x1 + 4, max(18, y1 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 220, 0),
            1,
            cv2.LINE_AA,
        )
        return out

    def _flow_to_bgr(self, flow: np.ndarray) -> np.ndarray:
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = (ang * 90.0 / np.pi).astype(np.uint8)
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _prepare_flow_debug_image(self, img: np.ndarray, title: str) -> np.ndarray:
        out = img.copy()
        cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), -1)
        cv2.putText(
            out,
            title,
            (6, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return out

    def _show_flow_debug_panel(
        self,
        img1_rgb: np.ndarray,
        img2_rgb: np.ndarray,
        motion_info: dict[str, np.ndarray | bool],
        direction_name: str,
    ) -> None:
        if not self.topdown_debug:
            return

        frame1 = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2BGR)
        frame1 = self._draw_danger_region(frame1)
        frame1 = self._draw_flow_roi(frame1)
        frame1 = self._prepare_flow_debug_image(
            frame1,
            f"Frame 1 / {direction_name}",
        )

        frame2 = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2BGR)
        frame2 = self._draw_flow_roi(frame2)
        frame2 = self._prepare_flow_debug_image(frame2, "Frame 2")

        flow = motion_info.get("flow")
        moving_mask = motion_info.get("moving_mask")
        if isinstance(flow, np.ndarray):
            flow_vis = self._flow_to_bgr(flow)
        else:
            flow_vis = np.zeros_like(frame1)
        flow_vis = self._draw_flow_roi(flow_vis)
        if isinstance(moving_mask, np.ndarray):
            contours, _ = cv2.findContours(
                (moving_mask.astype(np.uint8) * 255),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(flow_vis, contours, -1, (255, 255, 255), 1)
        flow_vis = self._prepare_flow_debug_image(flow_vis, "Optical Flow")

        panel = np.hstack([frame1, frame2, flow_vis])
        cv2.imshow(self.flow_debug_window_name, panel)
        cv2.waitKey(1)

    def _will_collide_by_optical_flow(
        self,
        motion_info: dict[str, np.ndarray | bool],
        direction_name: str,
    ) -> bool:
        if not motion_info["valid"]:
            return False

        moving_mask = motion_info["moving_mask"]
        flow = motion_info["flow"]
        if not isinstance(moving_mask, np.ndarray) or not isinstance(flow, np.ndarray):
            return False
        if not np.any(moving_mask):
            return False

        height, width = moving_mask.shape
        _, xs = np.where(moving_mask)
        mean_x = float(np.mean(xs))
        mean_vx = float(np.mean(flow[..., 0][moving_mask]))
        pred_x = mean_x + mean_vx

        danger_x_min, _, danger_x_max, _ = self._danger_region_bounds(width, height)
        in_danger_now = danger_x_min <= mean_x <= danger_x_max
        in_danger_after_move = danger_x_min <= pred_x <= danger_x_max
        will_collide = in_danger_now or in_danger_after_move

        print(
            f"[FlowPredict] dir={direction_name}, "
            f"mean_x={mean_x:.1f}, mean_vx={mean_vx:.2f}, "
            f"pred_x={pred_x:.1f}, will_collide={will_collide}"
        )
        return will_collide

    def _update_camera(self) -> None:
        if not self.gui or self.quadruped is None or self.quadruped.body_id is None:
            return
        pos, _ = p.getBasePositionAndOrientation(self.quadruped.body_id)
        p.resetDebugVisualizerCamera(
            cameraDistance=8.0,
            cameraYaw=45,
            cameraPitch=-45,
            cameraTargetPosition=pos,
        )

    def _sleep_if_gui(self) -> None:
        if self.gui:
            time.sleep(self.physics_dt)

class PyBulletQuadrupedMazeDemo:
    """Load an RL model and let it command the PyBullet quadruped maze env."""

    def __init__(
        self,
        model_path: str,
        size: int = 15,
        max_steps: int = 400,
        test_mode: int = 0,
        curriculum_levels: list[tuple[int, int]] | None = None,
        deterministic: bool = True,
        gui: bool = True,
        road_width: int = 5,
        obstacle_prob: float = 0.2,
        topdown_debug: bool = True,
    ) -> None:
        if PPO is None:
            raise RuntimeError(
                "stable-baselines3 is required to load a PPO model for the demo."
            )

        self.model = PPO.load(model_path)
        self.deterministic = deterministic
        self.env = PyBulletQuadrupedMazeEnv(
            size=size,
            max_steps=max_steps,
            test_mode=test_mode,
            curriculum_levels=curriculum_levels,
            gui=gui,
            road_width=road_width,
            obstacle_prob=obstacle_prob,
            topdown_debug=topdown_debug,
        )

    def run_one_episode(self, duration: float | None = None) -> None:
        obs, _ = self.env.reset()
        terminated = False
        truncated = False
        reward = 0.0
        start = time.time()

        while not (terminated or truncated):
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            obs, reward, terminated, truncated, _ = self.env.step(action)
            print(
                f"step={self.env.logic_env.steps}, "
                f"action={ACTION_TO_NAME[int(action)]}, reward={reward:.2f}, "
                f"agent={self.env.agent_pos.tolist()}, "
                f"goal={self.env.goal_pos.tolist()}"
            )

            if duration is not None and time.time() - start >= duration:
                break

        if terminated:
            print("Reached goal.")
        elif truncated:
            print("Reached max steps.")

    def close(self) -> None:
        self.env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PyBullet quadruped maze demo using the existing PPO maze model."
    )
    parser.add_argument(
        "--model",
        default=str(PROJECT_DIR / "model" / "ppo_maze.zip"),
        help="Path to a Stable-Baselines3 PPO model.",
    )
    parser.add_argument("--size", type=int, default=9)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--obstacle-prob", type=float, default=0.5)
    parser.add_argument(
        "--no-topdown",
        action="store_true",
        help="Disable the OpenCV top-down maze debug window.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demo = PyBulletQuadrupedMazeDemo(
        model_path=args.model,
        size=args.size,
        max_steps=args.max_steps,
        test_mode=1,
        curriculum_levels=[(1, 1000)],
        deterministic=True,
        gui=not args.direct,
        obstacle_prob=args.obstacle_prob,
        topdown_debug=not args.no_topdown,
    )
    try:
        demo.run_one_episode(duration=args.duration)
    finally:
        demo.close()


if __name__ == "__main__":
    main()

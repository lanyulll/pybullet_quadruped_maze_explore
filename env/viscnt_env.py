import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from collections import deque


class MazeEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, size=9, max_steps=400, test_mode=0, curriculum_levels=None):
        super().__init__()

        self.size = size
        self.max_steps = max_steps
        self.test_mode = test_mode

        if curriculum_levels is None:
            self.curriculum_levels = [
                (1, 10),
                (11, 20),
                (21, 30),
            ]
        else:
            self.curriculum_levels = curriculum_levels

        self.current_level = None
        self.current_shortest_path_len = None

        # 动作空间：
        # 0: 上
        # 1: 下
        # 2: 左
        # 3: 右
        self.action_space = spaces.Discrete(4)

        # 状态空间 6 维：
        # 上、下、左、右位置的 visit_count，dx，dy
        # 若某方向是墙或边界，则 visit_count 固定为 20
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, -1.0, -1.0], dtype=np.float32),
            high=np.array([20, 20, 20, 20, 1.0, 1.0], dtype=np.float32),
            shape=(6,),
            dtype=np.float32,
        )

        self.reset()

    def _generate_maze(self):
        size = self.size
        if size % 2 == 0:
            size += 1
            self.size = size

        maze = np.ones((size, size), dtype=np.int32)

        stack = [(1, 1)]
        maze[1, 1] = 0

        while stack:
            x, y = stack[-1]
            directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
            random.shuffle(directions)

            found = False
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 1 <= nx < size - 1 and 1 <= ny < size - 1:
                    if maze[nx, ny] == 1:
                        maze[x + dx // 2, y + dy // 2] = 0
                        maze[nx, ny] = 0
                        stack.append((nx, ny))
                        found = True
                        break

            if not found:
                stack.pop()

        return maze

    def _shortest_path_length(self, start, goal):
        start = tuple(start.tolist()) if isinstance(start, np.ndarray) else tuple(start)
        goal = tuple(goal.tolist()) if isinstance(goal, np.ndarray) else tuple(goal)

        if start == goal:
            return 0

        q = deque([(start, 0)])
        visited = {start}
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while q:
            (x, y), dist = q.popleft()

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                nxt = (nx, ny)

                if not (0 <= nx < self.size and 0 <= ny < self.size):
                    continue
                if self.grid[nx, ny] == 1:
                    continue
                if nxt in visited:
                    continue

                if nxt == goal:
                    return dist + 1

                visited.add(nxt)
                q.append((nxt, dist + 1))

        return -1

    def _generate_episode_by_level(self, level_min, level_max, max_trials=1000):
        for _ in range(max_trials):
            self.grid = self._generate_maze()
            self.size = self.grid.shape[0]

            free_spaces = np.argwhere(self.grid == 0)

            self.agent_pos = free_spaces[np.random.choice(len(free_spaces))]
            self.start_pos = self.agent_pos.copy()

            while True:
                self.goal_pos = free_spaces[np.random.choice(len(free_spaces))]
                if not np.array_equal(self.goal_pos, self.agent_pos):
                    break

            sp_len = self._shortest_path_length(self.agent_pos, self.goal_pos)
            if level_min <= sp_len <= level_max:
                self.current_shortest_path_len = sp_len
                return True

        return False

    def _get_pos_visit(self, pos):
        x, y = pos

        if not (0 <= x < self.size and 0 <= y < self.size):
            return 20.0

        if self.grid[x, y] == 1:
            return 20.0

        return float(min(self.visit_count[x, y], 15))

    def _get_obs(self):
        norm = max(self.size - 1, 1)

        x, y = self.agent_pos

        up_vis = self._get_pos_visit(np.array([x - 1, y]))
        down_vis = self._get_pos_visit(np.array([x + 1, y]))
        left_vis = self._get_pos_visit(np.array([x, y - 1]))
        right_vis = self._get_pos_visit(np.array([x, y + 1]))

        dx = (self.goal_pos[0] - self.agent_pos[0]) / norm
        dy = (self.goal_pos[1] - self.agent_pos[1]) / norm

        return np.array(
            [up_vis, down_vis, left_vis, right_vis, dx, dy],
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0

        if self.test_mode == 1:
            self.current_level = random.choice(self.curriculum_levels)
            level_min, level_max = self.current_level

            ok = self._generate_episode_by_level(level_min, level_max)
            if not ok:
                raise RuntimeError(
                    f"测试模式下无法在限定次数内生成最短路径长度位于 "
                    f"[{level_min}, {level_max}] 的迷宫。"
                )
        else:
            self.grid = self._generate_maze()
            self.size = self.grid.shape[0]

            free_spaces = np.argwhere(self.grid == 0)

            self.agent_pos = free_spaces[np.random.choice(len(free_spaces))]
            self.start_pos = self.agent_pos.copy()

            while True:
                self.goal_pos = free_spaces[np.random.choice(len(free_spaces))]
                if not np.array_equal(self.goal_pos, self.agent_pos):
                    break

            self.current_shortest_path_len = self._shortest_path_length(
                self.agent_pos, self.goal_pos
            )
            self.current_level = None

        self.visit_count = np.zeros_like(self.grid, dtype=np.int32)
        self.visit_count[tuple(self.agent_pos)] = 1

        return self._get_obs(), {}

    def step(self, action):
        action = int(np.asarray(action).item())

        self.steps += 1

        reward = 0.0
        terminated = False
        truncated = False

        old_dist = np.linalg.norm(self.agent_pos - self.goal_pos)

        move = {
            0: np.array([-1, 0]),
            1: np.array([1, 0]),
            2: np.array([0, -1]),
            3: np.array([0, 1]),
        }

        new_pos = self.agent_pos + move[action]
        moved = False

        if 0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size:
            if self.grid[tuple(new_pos)] == 1:
                reward -= 10
            else:
                self.agent_pos = new_pos
                moved = True
        else:
            reward -= 10

        new_dist = np.linalg.norm(self.agent_pos - self.goal_pos)

        if new_dist < old_dist:
            reward += 5
        elif new_dist > old_dist:
            reward -= 3

        if moved:
            pos = tuple(self.agent_pos)
            current_visits = min(self.visit_count[pos], 15)

            if current_visits == 0:
                reward += 2
            else:
                reward -= 0.5 * current_visits

            self.visit_count[pos] = min(self.visit_count[pos] + 1, 15)

        if np.array_equal(self.agent_pos, self.goal_pos):
            reward += 100
            terminated = True

        if self.steps >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        symbols = {
            0: ".",
            1: "#",
        }
        lines = []
        for i in range(self.size):
            cells = []
            for j in range(self.size):
                pos = np.array([i, j])
                if np.array_equal(pos, self.agent_pos):
                    cells.append("A")
                elif np.array_equal(pos, self.goal_pos):
                    cells.append("G")
                else:
                    cells.append(symbols[int(self.grid[i, j])])
            lines.append(" ".join(cells))

        print("\n".join(lines))

    def render_visit_count(self):
        lines = []
        for i in range(self.size):
            cells = []
            for j in range(self.size):
                if self.grid[i, j] == 1:
                    cells.append("##")
                else:
                    cells.append(f"{int(self.visit_count[i, j]):02d}")
            lines.append(" ".join(cells))

        print("\n".join(lines))


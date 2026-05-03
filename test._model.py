import os
from collections import deque

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

from env.viscnt_env import MazeEnv


def shortest_path_length(grid: np.ndarray, start: np.ndarray, goal: np.ndarray) -> int:
    """
    用 BFS 计算迷宫中从 start 到 goal 的最短路径长度。
    若不可达，返回 -1。
    """
    start = tuple(start.tolist())
    goal = tuple(goal.tolist())

    if start == goal:
        return 0

    rows, cols = grid.shape
    q = deque([(start, 0)])
    visited = {start}

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while q:
        (x, y), dist = q.popleft()

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            nxt = (nx, ny)

            if not (0 <= nx < rows and 0 <= ny < cols):
                continue
            if grid[nx, ny] == 1:
                continue
            if nxt in visited:
                continue

            if nxt == goal:
                return dist + 1

            visited.add(nxt)
            q.append((nxt, dist + 1))

    return -1


def evaluate_model(
    model_path: str,
    n_episodes: int = 200,
    maze_size: int = 9,
    max_steps: int = 400,
    deterministic: bool = True,
):
    """
    在随机迷宫上评估模型表现。
    返回逐 episode 结果的 DataFrame。
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    env = MazeEnv(size=maze_size, max_steps=max_steps, test_mode=1,curriculum_levels = [
                (1, 1000)])
    model = PPO.load(model_path)

    records = []

    for ep in range(n_episodes):
        obs, _ = env.reset()

        # 记录该迷宫的最短路径长度
        sp_len = shortest_path_length(env.grid, env.start_pos, env.goal_pos)

        terminated = False
        truncated = False
        step_count = 0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            step_count += 1

        success = bool(terminated and np.array_equal(env.agent_pos, env.goal_pos))

        records.append({
            "episode": ep + 1,
            "success": int(success),
            "steps": step_count,
            "shortest_path_len": sp_len,
            "start_x": int(env.start_pos[0]),
            "start_y": int(env.start_pos[1]),
            "goal_x": int(env.goal_pos[0]),
            "goal_y": int(env.goal_pos[1]),
        })

        print(
            f"Episode {ep + 1:03d} | "
            f"success={success} | "
            f"steps={step_count} | "
            f"shortest_path_len={sp_len}"
        )

    df = pd.DataFrame(records)
    return df


def summarize_results(df: pd.DataFrame):
    total_success_rate = df["success"].mean()
    avg_steps_all = df["steps"].mean()

    success_df = df[df["success"] == 1]
    avg_steps_success = success_df["steps"].mean() if not success_df.empty else np.nan

    print("\n===== 总体评估结果 =====")
    print(f"测试总局数: {len(df)}")
    print(f"总体成功率: {total_success_rate:.4f} ({total_success_rate * 100:.2f}%)")
    print(f"全部样本平均步数: {avg_steps_all:.2f}")
    if np.isnan(avg_steps_success):
        print("成功样本平均步数: 无成功样本")
    else:
        print(f"成功样本平均步数: {avg_steps_success:.2f}")


def add_distance_bins(df: pd.DataFrame, bin_width: int = 5) -> pd.DataFrame:
    """
    按最短路径长度分段，例如:
    1-5, 6-10, 11-15, ...
    """
    valid_df = df[df["shortest_path_len"] >= 0].copy()
    if valid_df.empty:
        raise ValueError("所有样本 shortest_path_len 都是 -1，无法分段统计。")

    max_len = int(valid_df["shortest_path_len"].max())
    upper = ((max_len // bin_width) + 1) * bin_width
    bins = list(range(0, upper + bin_width, bin_width))

    labels = []
    for i in range(len(bins) - 1):
        left = bins[i] + 1
        right = bins[i + 1]
        labels.append(f"{left}-{right}")

    valid_df["distance_bin"] = pd.cut(
        valid_df["shortest_path_len"],
        bins=bins,
        labels=labels,
        include_lowest=False,
        right=True
    )

    return valid_df


def plot_by_distance_bins(df: pd.DataFrame, bin_width: int = 5, save_dir: str = "eval_results"):
    os.makedirs(save_dir, exist_ok=True)

    binned_df = add_distance_bins(df, bin_width=bin_width)

    grouped = (
        binned_df.groupby("distance_bin", observed=False)
        .agg(
            avg_success_rate=("success", "mean"),
            avg_steps=("steps", "mean"),
            count=("episode", "count"),
        )
        .reset_index()
    )

    # 去掉没有样本的分段
    grouped = grouped[grouped["count"] > 0].copy()

    print("\n===== 分距离段统计 =====")
    print(grouped)

    # 图1：不同距离段的平均成功率
    plt.figure(figsize=(10, 5))
    plt.plot(grouped["distance_bin"].astype(str), grouped["avg_success_rate"], marker="o")
    plt.title("Average Success Rate by Shortest Path Length Bin")
    plt.xlabel("Shortest Path Length Bin")
    plt.ylabel("Average Success Rate")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "success_rate_by_distance_bin.png"), dpi=150)

    # 图2：不同距离段的平均步数
    plt.figure(figsize=(10, 5))
    plt.plot(grouped["distance_bin"].astype(str), grouped["avg_steps"], marker="o")
    plt.title("Average Steps by Shortest Path Length Bin")
    plt.xlabel("Shortest Path Length Bin")
    plt.ylabel("Average Steps")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "avg_steps_by_distance_bin.png"), dpi=150)

    plt.show()

    return grouped


def main():
    model_path = "./model/ppo_maze.zip"   # 改成你的模型路径
    n_episodes = 1000              # 测试局数
    maze_size = 25          # 迷宫尺寸
    max_steps = 1000             # 每局最大步数
    bin_width = 50                 # 距离分段宽度

    df = evaluate_model(
        model_path=model_path,
        n_episodes=n_episodes,
        maze_size=maze_size,
        max_steps=max_steps,
        deterministic=True,
    )


    os.makedirs("eval_results", exist_ok=True)
    df.to_csv("eval_results/test_results.csv", index=False, encoding="utf-8-sig")

    summarize_results(df)

    grouped = plot_by_distance_bins(df, bin_width=bin_width, save_dir="eval_results")
    grouped.to_csv("eval_results/grouped_results.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
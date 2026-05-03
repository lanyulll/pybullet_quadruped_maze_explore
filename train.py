import gymnasium as gym
import os
import sys

from env.viscnt_env import MazeEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.callbacks import BaseCallback
import copy

class EvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=10000):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            print("\n===== Evaluation =====")

            obs, _ = self.eval_env.reset()

            done = False
            truncated = False

            while not (done or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = self.eval_env.step(action)

            # 输出路径访问次数
            print('done:', done)
            self.eval_env.render()
            self.eval_env.render_visit_count()

        return True

# 1. 创建环境
env = MazeEnv(size=9)
env = Monitor(env, 'log/')
env_eval = MazeEnv(size=32, test_mode=True,max_steps=1000,curriculum_levels = [
                (1, 1000),
            ])
callback = EvalCallback(env_eval, eval_freq=10000)
# 2. 检查环境（强烈建议）
check_env(env)

# 3. 创建模型
model = PPO(
    policy="MlpPolicy",   # 当前是低维状态，用MLP
    env=env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    device="cpu",
)

# 4. 开始训练
model.learn(
    total_timesteps=200_000,
    callback=callback
)

# 5. 保存模型
model.save("./model/ppo_maze")

import pandas as pd
import matplotlib.pyplot as plt

# 读取日志
df = pd.read_csv("log/monitor.csv", skiprows=1)

# 平滑（可选）
df['r_smooth'] = df['r'].rolling(window=20).mean()
df['l_smooth'] = df['l'].rolling(window=20).mean()

# 画 reward
plt.figure()
plt.plot(df['r_smooth'])
plt.title("Average Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.savefig("reward_curve.png")

# 画 episode length
plt.figure()
plt.plot(df['l_smooth'])
plt.title("Episode Length")
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.savefig("length_curve.png")

plt.show()

print("训练完成！")
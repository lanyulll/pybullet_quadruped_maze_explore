# PyBullet Quadruped Maze Navigation

A reinforcement learning project for maze exploration with a four-legged robot. The project combines a lightweight Gymnasium maze environment, a Stable-Baselines3 PPO policy, and a PyBullet visualization/control layer for a 12-actuator quadruped.

The trained policy chooses one of four grid actions: up, down, left, or right. `run_demo.py` converts those high-level actions into PyBullet robot motion, renders maze walls and moving obstacles, and uses RGB optical-flow checks to delay unsafe moves.

## Features

- Random maze generation with start/goal sampling.
- PPO training and evaluation with Stable-Baselines3.
- Text-based maze environment for fast policy training.
- PyBullet quadruped simulation with inverse-kinematics walking control.
- Moving obstacle visualization and optical-flow safety prediction.
- Included example model at `model/ppo_maze.zip`.

## Project Structure

```text
.
|-- env/
|   |-- __init__.py
|   `-- viscnt_env.py                 # Gymnasium maze environment
|-- model/
|   `-- ppo_maze.zip                  # Pretrained PPO model
|-- eight_axis_quadruped_control.py   # PyBullet 12-axis quadruped URDF, IK, gait control
|-- run_demo.py                       # Full PyBullet maze demo
|-- train.py                          # PPO training script
|-- test._model.py                    # Batch evaluation and plotting script
|-- requirements.txt
`-- README.md
```

## Installation

Python 3.9 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Linux or macOS, activate the virtual environment with:

```bash
source .venv/bin/activate
```

## Quick Start

Run the full PyBullet demo with the included PPO model:

```powershell
python run_demo.py
```

Run a short headless check without the PyBullet GUI:

```powershell
python run_demo.py --direct --duration 30 --no-topdown
```

Use a custom model:

```powershell
python run_demo.py --model .\model\ppo_maze.zip --size 15 --max-steps 1000
```

## Training

Train a PPO policy on the lightweight maze environment:

```powershell
python train.py
```

The script saves the trained model to:

```text
model/ppo_maze.zip
```

It also writes Stable-Baselines3 monitor logs under `log/` and training curves such as `reward_curve.png` and `length_curve.png`.

## Evaluation

Evaluate the saved model across many randomly generated mazes:

```powershell
python test._model.py
```

Evaluation outputs are written to `eval_results/`, including per-episode results and plots grouped by shortest-path length.

## Main Components

- `env/viscnt_env.py`: Defines `MazeEnv`, a Gymnasium environment with six-dimensional observations: visit counts for the four neighboring cells plus normalized goal direction.
- `train.py`: Creates `MazeEnv`, validates it with Stable-Baselines3, trains PPO, and saves learning curves.
- `test._model.py`: Loads a PPO model, computes shortest-path lengths with BFS, measures success rate and step count, and saves evaluation charts.
- `eight_axis_quadruped_control.py`: Generates a compact 12-axis quadruped URDF at runtime and controls the robot with inverse kinematics and gait planning.
- `run_demo.py`: Wraps the maze environment in a PyBullet scene, executes PPO actions with the quadruped, and renders walls, goals, moving obstacles, camera views, and debug overlays.

## Notes

- The PyBullet GUI and OpenCV debug windows require a desktop display. Use `--direct --no-topdown` for headless checks.
- Maze size is forced to an odd value internally because the maze generator carves paths on an odd grid.
- `test._model.py` is executable as a script despite the unusual filename.
- Generated outputs such as `log/`, `eval_results/`, and plot images can be safely deleted and regenerated.

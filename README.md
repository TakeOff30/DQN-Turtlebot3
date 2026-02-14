# DQN Reinforcement Learning for Turtlebot3 Navigation

This project trains a **Dueling Deep Q-Network (DQN)** agent to navigate a Turtlebot3 robot in progressively complex environments using **curriculum learning**. The robot learns to reach goal positions while avoiding obstacles in Gazebo simulation.

## Features

- **Dueling DQN Architecture**: Separates value and advantage streams for improved learning stability
- **Double DQN**: Reduces Q-value overestimation using separate policy and target networks
- **Curriculum Learning**: 5-stage progressive difficulty (empty → static obstacles → dynamic obstacles)
- **Soft Target Updates**: Stabilizes training with gradual target network updates (τ=0.005)
- **Experience Replay**: 100k capacity memory with prioritized warm-start
- **Real-time Visualization**: Live training metrics via ROS topics
- **Checkpoint Management**: Automatic model saving and resumable training
- **Comprehensive Reporting**: Episode statistics, reward breakdowns, and training summaries

## Project Structure

```
DQN-Turtlebot3/
├── curriculum_learning/          # Main training package
│   ├── config/                   # Stage configurations (YAML)
│   │   ├── stage1_params.yaml    # Empty 4x4m arena
│   │   ├── stage2_params.yaml    # Static obstacles
│   │   ├── stage3_params.yaml    # More obstacles
│   │   ├── stage4_params.yaml    # Dynamic obstacles
│   │   └── stage5_params.yaml    # Complex maze
│   ├── launch/                   # ROS launch files
│   ├── scripts/                  # Training implementation
│   │   ├── start_deepqlearning.py      # Main DQN training loop
│   │   ├── start_deepqinference.py     # Model inference
│   │   ├── checkpoint_manager.py       # Save/load models
│   │   ├── training_manager.py         # Training orchestration
│   │   ├── training_reporter.py        # Report generation
│   │   └── result_graph.py             # Live plotting
│   ├── trained_models/           # Saved checkpoints & best models
│   └── training_reports/         # Training summaries
├── openai_ros/                   # Custom OpenAI Gym environments
│   └── src/openai_ros/
│       ├── robot_envs/           # Robot interface (sensors, actuators)
│       └── task_envs/            # Task-specific logic (rewards, goals)
└── turtlebot3/                   # Turtlebot3 ROS packages
```

## Quick Start

### Docker Setup (Windows/Linux)

**Windows:** Start VcXsrv (XLaunch) for GUI visualization, then:

```bash
docker compose up -d
docker exec -it final_project bash
```

**Linux:** Enable Docker display access:

```bash
xhost +local:docker
docker compose -f docker-compose.linux.yml up -d
docker exec -it final_project_linux bash
```

### First-Time Setup (Inside Container)

```bash
source /opt/ros/noetic/setup.bash
cd ~/simulation_ws
catkin build
source devel/setup.bash
```

## Training

### Start Training from Scratch

```bash
# Stage 1: Empty arena (1000 episodes)
roslaunch curriculum_learning start_training_stage1.launch

# Stage 2: Static obstacles (1000 episodes)
roslaunch curriculum_learning start_training_stage2.launch

# Stage 3: More obstacles (1000 episodes)
roslaunch curriculum_learning start_training_stage3.launch

# Stage 4: Dynamic obstacles (2000 episodes)
roslaunch curriculum_learning start_training_stage4.launch

# Stage 5: Complex maze (2000 episodes)
roslaunch curriculum_learning start_training_stage5.launch
```

### Resume Training from Checkpoint

Edit the stage config file (e.g., `config/stage2_params.yaml`):

```yaml
load_pretrained_model: true
checkpoint_file: "best_model_stage1.pth"  # Path relative to trained_models/
```

Then launch normally:

```bash
roslaunch curriculum_learning start_training_stage2.launch
```

## Inference

Run a trained model:

```bash
roslaunch curriculum_learning start_inference_final.launch
```

Configure which model to load in `config/inference_final_params.yaml`.

## Training Configuration

Key hyperparameters (editable per stage in `config/stage*_params.yaml`):

```yaml
gamma: 0.99                    # Discount factor
learning_rate: 0.0001          # Adam optimizer LR
epsilon_start: 1.0             # Initial exploration
epsilon_end: 0.05              # Final exploration
epsilon_decay: 50000           # Decay rate (steps)
batch_size: 128                # Replay batch size
tau: 0.005                     # Soft target update rate
replay_memory_size: 100000     # Experience buffer capacity
n_episodes: 1000               # Training episodes
```

## Observation & Action Space

**Observation** (26-dim vector):
- 24 LiDAR readings (360° → downsampled)
- 2 goal coordinates (relative x, y)

**Actions** (discrete):
- 0: Forward (linear=0.15 m/s)
- 1: Turn Left (angular=0.5 rad/s)
- 2: Turn Right (angular=-0.5 rad/s)

## Reward Function

- **Distance improvement**: Δd × scaling factor
- **Goal reached**: +100 (terminal)
- **Collision**: -100 (terminal)
- **Angle alignment**: Bonus for heading toward goal
- **Sway penalty**: Penalizes excessive turning

## Monitoring Training

Trained models are auto-saved to `trained_models/stage_X_YYYYMMDD-HHMMSS/`:
- `models/`: Best model + periodic checkpoints (every 500 episodes)
- `plots/`: Training curves (rewards, distances, epsilon)
- `report/`: Text summary with statistics

**Live Visualization**: If you have RQT or custom plotters, subscribe to:
- `/result`: [avg_max_Q, episode_reward, epsilon]
- `/get_action`: [action_index, cumulative_reward, step_reward]

## Curriculum Learning Stages

| Stage | Environment | Episodes | Min Range | Key Challenge |
|-------|-------------|----------|-----------|---------------|
| 1 | Empty 4×4m | 1000 | 0.18m | Basic navigation |
| 2 | Static obstacles | 1000 | 0.12m | Obstacle avoidance |
| 3 | More obstacles | 1000 | 0.12m | Complex paths |
| 4 | Dynamic obstacles | 2000 | 0.12m | Moving targets |
| 5 | Maze-like | 2000 | 0.12m | Long-horizon planning |

Each stage loads the best model from the previous stage for transfer learning.

## Troubleshooting

**"Package not found"**: Forgot to `source devel/setup.bash` after `catkin build`

**Gazebo won't start**: Check X11 forwarding (Windows: VcXsrv running, Linux: `xhost +`)

**Training hangs**: Verify `roscore` is running (auto-started by `roslaunch`)

**CUDA errors**: GPU training optional; CPU fallback automatic

## ROS Dependencies

- ROS Noetic
- Turtlebot3 packages (turtlebot3, turtlebot3_simulations, turtlebot3_msgs)
- openai_ros (custom Gym wrappers)
- Gazebo 11

## Python Dependencies

- PyTorch (CUDA optional)
- OpenAI Gym 0.25.0
- NumPy
- PyQt5 + pyqtgraph (for live plots)

## References

- **Dueling DQN**: Wang et al. 2016, "Dueling Network Architectures for Deep RL"
- **Double DQN**: Van Hasselt et al. 2016, "Deep Reinforcement Learning with Double Q-learning"
- **Curriculum Learning**: Bengio et al. 2009, "Curriculum Learning"

## License

See LICENSE file for details.

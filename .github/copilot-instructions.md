# AI Agent Instructions: DQN-Turtlebot3 Navigation

## Project Overview
ROS Noetic-based reinforcement learning system training a Turtlebot3 robot to navigate using Deep Q-Networks (DQN) with curriculum learning. Robot learns goal-reaching in Gazebo simulation across 3 progressive difficulty stages.

## Architecture & Data Flow

### Layer Structure
1. **ROS Robot Environment** ([openai_ros/robot_envs/turtlebot3_env.py](openai_ros/openai_ros/src/openai_ros/robot_envs/turtlebot3_env.py))
   - Base robot interface: subscribes to `/odom`, `/imu`, `/scan`
   - Publishes to `/cmd_vel` for movement control
   - Inherits from `robot_gazebo_env.RobotGazeboEnv` for Gazebo pause/unpause control

2. **Task Environment** ([openai_ros/task_envs/turtlebot3/turtlebot3_world.py](openai_ros/openai_ros/src/openai_ros/task_envs/turtlebot3/turtlebot3_world.py))
   - Defines observation space (24 laser readings + 2 goal coordinates)
   - Action space: 3 discrete actions (forward, turn-left, turn-right)
   - Reward calculation: distance-to-goal improvement, goal-reaching, collision penalties
   - Random goal/robot spawning within arena boundaries

3. **DQN Training Loop** ([curriculum_learning/scripts/start_deepqlearning.py](curriculum_learning/scripts/start_deepqlearning.py))
   - Policy network: 4-layer MLP (512→256→128→3 actions) with Xavier initialization
   - Target network updated every 10 episodes
   - Replay memory with batch sampling (128 samples)
   - Epsilon-greedy exploration with exponential decay

### Critical Data Flow
```
Gazebo → /scan, /odom → turtlebot3_env → task_env._get_obs() → state tensor →
DQN policy_net → action → task_env.step() → /cmd_vel → Gazebo
```

## Development Workflows

### Docker-Based Development (Standard)
```bash
# Build and start container
docker-compose up -d

# Enter container
docker exec -it final_project bash

# First time setup inside container
source /opt/ros/noetic/setup.bash
cd ~/simulation_ws
catkin build
source devel/setup.bash

# Launch training (auto-launches Gazebo via ROSLauncher)
roslaunch curriculum_learning start_training_stage1.launch
```

**Key**: ROS workspace is at `/home/ubuntu/simulation_ws` in container, volume-mounted from Windows host. Always `source devel/setup.bash` before ROS commands.

### Curriculum Learning Progression
Stage-specific configs in [curriculum_learning/config/](curriculum_learning/config/):
- **Stage 1**: Empty 4x4m arena, 1000 episodes, permissive collision (min_range: 0.18m)
- **Stage 2**: Static obstacles added, tighter collision (min_range: 0.12m)
- **Stage 3**: Moving obstacles, maze-like environment, 2000 episodes

Launch with `roslaunch curriculum_learning start_training_stage{1,2,3}.launch`. Each stage loads its YAML params into ROS parameter server (`/turtlebot3/*`).

### Checkpoint Management
Checkpoints auto-saved every 50 episodes to [curriculum_learning/trained_models/](curriculum_learning/trained_models/):
- Contains: policy/target network states, optimizer state, reward history, epsilon decay state
- Resume training: set `resume_from_checkpoint: true` in stage config YAML
- **CheckpointManager** ([checkpoint_manager.py](curriculum_learning/scripts/checkpoint_manager.py)) handles save/load

## Project-Specific Patterns

### 1. ROS Parameter-Driven Configuration
All hyperparameters loaded via `rosparam` from YAML files, NOT hardcoded. Access in Python:
```python
self.linear_forward_speed = rospy.get_param('/turtlebot3/linear_forward_speed')
```
When modifying training params, edit stage YAML files, not Python code.

### 2. OpenAI Gym Environment Registration
Environments registered dynamically via [openai_ros/task_envs/task_envs_list.py](openai_ros/openai_ros/src/openai_ros/task_envs/task_envs_list.py). Use `StartOpenAI_ROS_Environment('TurtleBot3World-v0')` helper function, which:
- Registers task environment with Gym
- Launches required ROS launch files (Gazebo world, robot spawn)
- Returns ready-to-use `gym.make()` environment

### 3. Gazebo Simulation Management
`gazebo.pauseSim()` / `gazebo.unpauseSim()` critical for synchronous training:
- Simulation PAUSED during DQN optimization (no physics updates)
- UNPAUSED only during `env.step()` execution
- Prevents time drift between ROS and training loop

### 4. PyTorch Device Handling
Check tensor device before forward pass:
```python
if not x.is_cuda and device.type == 'cuda':
    x = x.to(device)
```
GPU availability auto-detected; code runs on CPU fallback seamlessly.

### 5. Modular Monitoring Components
- **TrainingLogger** ([training_logger.py](curriculum_learning/scripts/training_logger.py)): Formatted rospy logging with timestamps
- **TrainingReporter** ([training_reporter.py](curriculum_learning/scripts/training_reporter.py)): Generates text summaries every 100 episodes
- **plot_training_metrics()** in [utils.py](curriculum_learning/scripts/utils.py): Live matplotlib/PyQt5 plots (requires X11 forwarding on Windows via VcXsrv)

## Integration Points

### Gazebo Services
- `/gazebo/set_model_state`: Move goal marker to new position each episode (called in `_init_env_variables()`)
- Robot respawn handled by `gazebo.resetSim()` (full simulation reset)

### ROS Topic Contract
Task environment expects:
- `/scan` (LaserScan): 360 readings, downsampled to 24
- `/odom` (Odometry): Robot pose (x, y, yaw) for distance-to-goal calculation
- `/cmd_vel` (Twist): 3 discrete actions mapped to (linear.x, angular.z) velocities

### External Dependencies
- **gym==0.25.0**: Specific version for API compatibility
- **torch**: No version pinned, uses latest
- **PyQt5 + pyqtgraph**: For live training visualizations (optional)

## Common Pitfalls

1. **Workspace Sourcing**: Forgetting `source devel/setup.bash` causes "package not found" errors even after `catkin build`
2. **X11 Display**: GUI apps (Gazebo, matplotlib) need `DISPLAY=host.docker.internal:0.0` and VcXsrv running on Windows
3. **ROS Master**: If training hangs, check `roscore` is running (auto-started by `roslaunch`)
4. **Checkpoint Compatibility**: Loading checkpoints across different stage configs fails silently; ensure network architecture matches

## Key Files Reference
- DQN implementation: [curriculum_learning/scripts/start_deepqlearning.py](curriculum_learning/scripts/start_deepqlearning.py) (lines 44-73: network architecture)
- Reward function: [openai_ros/task_envs/turtlebot3/turtlebot3_world.py](openai_ros/openai_ros/src/openai_ros/task_envs/turtlebot3/turtlebot3_world.py) (line ~300+, see `_compute_reward()`)
- Launch system: [openai_ros/openai_ros_common.py](openai_ros/openai_ros/src/openai_ros/openai_ros_common.py) (ROSLauncher class)

## TODO Context (from PLAN.md)
Active refactoring goals:
- Remove unnecessary reward components (keep distance-based, goal-reaching, angle-to-goal, sway penalty)
- Clean up training loop in start_deepqlearning.py
- Finalize stage 2/3 obstacle implementations (currently stage 1 validated)

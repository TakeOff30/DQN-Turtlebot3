# DQN Reinforcement Learning for Turtlebot3 Navigation

This project trains a Deep Reinforcement Learning agent to navigate a Turtlebot3 robot in progressively complex environments using **curriculum learning**. The robot learns to reach goal positions while avoiding obstacles in Gazebo simulation.

The robot is equipped with LiDAR sernsors that allow to perceive the surrounding environment, take decisions about the next action and then perform it in the Gazebo environment.

We implemented the standard Deep Q-Network architecture (_Playing Atari with Deep Reinforcement Learning, 2013, Mnih et al._) as well as the Dueling DQN architecture (_Dueling Network Architectures for Deep Reinforcement Learning, 2016, Wang et al._).

In both cases we applied soft updates to the target network for faster convergence and we used the Double DQN optimizer (_Deep Reinforcement Learning with Double Q-learning, 2015, van Hasselt et al._) to prevent from the overestimation of Q-values from which the DQN implentation suffers.

We ran trainings adopting 3 different approaches and compared them:

- Dueling DQN architecture receiving in input a state vector composed by 180 laser readings pointing in front of the Turtlebot, the Euclidean distance to the goal and the sine and cosine of the angle of the goal with respect to the Turtlebot.
- Dueling DQN architecture receiving in input a state vector composed by 24 laser readings which are the result of a min-pooling process. It divides the 180 front laser readings in 24 chunks and from each chunk takes the minimum values recorded which contains the most relevant information for the agent to take a decision. On top of the 24 laser readings we add the positional information.
- Standard DQN architecture with 24 readings resulting from min-pooling process and positional information.

At each step the agent takes one of 5 possible actions ([1.5, 0.75, 0, -0.75, -1.5]) each consisting in the angular velocity to apply allowing to either turn left or turn right and with different intensities or do not turn at all.

N.B. The linear velocity applied at every episode is the same. We believe that this way of moving the robot is the primary point of possible improvement, because in frequent situations, to best avoid a moving obstacle the robot should just stop or slow down.

The Markov Decision Process is structured as follows:

- Positive reward when the robot reached the goal position.
- Negative reward and end the episode when the robot collides with an obstacle.
- We impose a maximum number of steps after which the episdode ends with the current cumulated reward.
- At each step we calculate the reward which is made of the following components:
    - we cache the distance from the goal in the previous step and compare it with the current distance; the difference is multiplied by a scaling factor.
    - as the robot approaches the goal, it receives a progressively larger positive shaping reward.
    - we add an orientation term that rewards heading alignment toward the goal direction.
    - we apply a sway/turn penalty to discourage unnecessary oscillations and unstable steering.

The overall goal is to train the agent to move in a restaurant-like environment characterized by static obstacles (e.g. tables) and moving obstacles (e.g. people).
During inference our evaluation criteria is the following:

- We define an episode as successful when the turtlebot reaches 3 goals in a row without colliding with obstacles along the way.
- We simulate 300 episodes and take into consideration the perccentage of successful episode and the average number of goals reached per episode.

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

### Build ROS workspace

```bash
catkin build
source devel/setup.bash
```

## Training

### Start Training for a stage

```bash
roslaunch curriculum_learning start_training_stage1.launch
```

### Start training from pretrained model

```yaml
load_pretrained_model: true
checkpoint_file: 'best_model_stage1.pth' # Path relative to trained_models/
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
gamma: 0.99 # Discount factor
learning_rate: 0.0001 # Adam optimizer LR
epsilon_start: 1.0 # Initial exploration
epsilon_end: 0.05 # Final exploration
epsilon_decay: 50000 # Decay rate (steps)
batch_size: 128 # Replay batch size
tau: 0.005 # Soft target update rate
replay_memory_size: 100000 # Experience buffer capacity
n_episodes: 1000 # Training episodes
```

## Monitoring Training

Trained models are auto-saved to `trained_models/stage_X_YYYYMMDD-HHMMSS/`:

- `models/`: Best model + periodic checkpoints (every 500 episodes)
- `plots/`: Training curves (rewards, distances, epsilon)
- `report/`: Text summary with statistics

**Live Visualization**:

- `/result`: [avg_max_Q, episode_reward, epsilon]
- `/get_action`: [action_index, cumulative_reward, step_reward]

## Troubleshooting

**"Package not found"**: Forgot to `source devel/setup.bash` after `catkin build`
**Gazebo won't start**: Check X11 forwarding (Windows: VcXsrv running, Linux: `xhost +`)

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

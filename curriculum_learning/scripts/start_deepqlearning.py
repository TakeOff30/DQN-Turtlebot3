#!/usr/bin/env python3

import gym
import numpy
import math
import random
from collections import namedtuple, deque
from itertools import count

import time
from gym import wrappers
import os
# ROS packages required
import rospy
import rospkg
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from checkpoint_manager import CheckpointManager
from training_logger import TrainingLogger
from training_reporter import TrainingReporter
from training_manager import TrainingManager
from std_msgs.msg import Float32MultiArray

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DuelingDQN(nn.Module):
    """Dueling DQN: separates Value and Advantage streams for better learning.
    Reference: Wang et al. 2016, 'Dueling Network Architectures for Deep RL'
    """

    def __init__(self, inputs, outputs):
        super(DuelingDQN, self).__init__()
        
        # Shared feature extraction
        self.feature = nn.Sequential(
            nn.Linear(inputs, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        
        # Value stream: V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Advantage stream: A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, outputs)
        )
        
        # He (Kaiming) initialization for ReLU networks
        self.apply(self._init_weights)
    
    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = x.to(device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


def select_action(state, eps_start, eps_end, eps_decay):
    global steps_done
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done / eps_decay)
    steps_done += 1
    
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            if state.dim() == 1:
                state = state.unsqueeze(0) 
            return policy_net(state).max(1)[1].view(1, 1), eps_threshold
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long), eps_threshold


def optimize_model(batch_size, gamma, tau):
    """Double DQN optimization with soft target network updates.
    - Uses policy_net to SELECT best actions (reduces overestimation)
    - Uses target_net to EVALUATE those actions
    - Soft-updates target_net after each optimization step
    """
    global loss_values, last_loss_value
    if len(memory) < batch_size:
        last_loss_value = None
        return
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    # Mask for non-terminal next states
    non_final_mask = torch.tensor(
        tuple(s is not None for s in batch.next_state),
        device=device, dtype=torch.bool
    )
    non_final_next_states = None
    if non_final_mask.any():
        non_final_next_states = torch.stack(
            [s for s in batch.next_state if s is not None]
        )

    state_batch = torch.stack(batch.state)
    # action_batch shape: (B, 1, 1) -> (B, 1)
    action_batch = torch.stack(batch.action).view(-1, 1)
    # reward_batch shape: (B, 1) -> (B,)
    reward_batch = torch.stack(batch.reward).view(-1)

    # Q(s_t, a_t): gather Q-values for the taken actions
    state_action_values = policy_net(state_batch).gather(1, action_batch).squeeze(1)

    # Double DQN: policy_net selects actions, target_net evaluates them
    next_state_values = torch.zeros(batch_size, device=device)
    if non_final_next_states is not None:
        with torch.no_grad():
            # Policy net picks the best action for each next state
            best_actions = policy_net(non_final_next_states).argmax(1, keepdim=True)
            # Target net evaluates Q-value of those actions
            next_state_values[non_final_mask] = target_net(
                non_final_next_states
            ).gather(1, best_actions).squeeze(1)

    # TD target: r + gamma * Q_target(s', argmax_a Q_policy(s', a))
    expected_state_action_values = reward_batch + (gamma * next_state_values)

    # Huber loss (SmoothL1) for robustness to outliers
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.detach())
    loss_values.append(loss.item())
    last_loss_value = loss.item()

    optimizer.zero_grad()
    loss.backward()
    # Global gradient norm clipping (more stable than per-param clamping)
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10.0)
    optimizer.step()

    # Soft update target network: theta_target = tau*theta_policy + (1-tau)*theta_target
    for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

if __name__ == '__main__':
  
    rospy.init_node('turtlebot3_world_qlearn', anonymous=True, log_level=rospy.INFO)

    # Init OpenAI_ROS ENV
    task_and_robot_environment_name = rospy.get_param(
        '/turtlebot3/task_and_robot_environment_name')
    env = StartOpenAI_ROS_Environment(
        task_and_robot_environment_name)
    rospy.loginfo("Gym environment ready")
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('curriculum_learning')
    outdir = pkg_path + '/training_results'

    trained_models_root = os.path.join(pkg_path, 'trained_models')
    os.makedirs(trained_models_root, exist_ok=True)
    
    last_time_steps = numpy.ndarray(0)

    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    gamma = rospy.get_param("/turtlebot3/gamma")
    epsilon_start = rospy.get_param("/turtlebot3/epsilon_start")
    epsilon_end = rospy.get_param("/turtlebot3/epsilon_end")
    epsilon_decay = rospy.get_param("/turtlebot3/epsilon_decay")
    n_episodes = rospy.get_param("/turtlebot3/n_episodes")
    batch_size = rospy.get_param("/turtlebot3/batch_size")
    lr = rospy.get_param("/turtlebot3/learning_rate", 0.0001)
    running_step = rospy.get_param("/turtlebot3/running_step")
    resume_training = rospy.get_param("/turtlebot3/load_pretrained_model", False)
    checkpoint_file = rospy.get_param("/turtlebot3/checkpoint_file", "best_model.pth")
    stage = rospy.get_param("/turtlebot3/stage")
    tau = rospy.get_param('/turtlebot3/tau', 0.005)
    replay_memory_size = rospy.get_param('/turtlebot3/replay_memory_size', 100000)
    
    run_id = f"stage_{stage}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = os.path.join(trained_models_root, run_id)

    models_dir = os.path.join(run_dir, 'models')
    plots_dir = os.path.join(run_dir, 'plots')
    report_dir = os.path.join(run_dir, 'report')

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    
    # Sends metrics to result_graph.py
    result_pub = rospy.Publisher('/result', Float32MultiArray, queue_size=10)   # Sends metrics to result_action.py
    result_action_pub = rospy.Publisher('/get_action', Float32MultiArray, queue_size=10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

    # Get number of actions from gym action space
    n_actions = env.action_space.n
    initial_obs = env.reset()
    assert env.observation_space.contains(initial_obs), \
         f"Observation {initial_obs} outside declared space {env.observation_space}"
    n_observations = len(initial_obs)

    policy_net = DuelingDQN(n_observations, n_actions).to(device)
    target_net = DuelingDQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
        
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    memory = ReplayMemory(replay_memory_size)
    episode_durations = []
    steps_done = 0
    start_episode = 0
    loss_values = []
    last_loss_value = None
    last_rewards = deque([], maxlen=50)
    max_avg_reward = 0
    
    checkpoint_manager = CheckpointManager(models_dir)
    logger = TrainingLogger()
    reporter = TrainingReporter(report_dir)
    training_manager = TrainingManager(checkpoint_manager, reporter, plots_dir)
    
    reporter.write_header()
    reporter.write_configuration(n_episodes, gamma, epsilon_start, epsilon_end, epsilon_decay, batch_size, tau)

    
    if resume_training:
        checkpoint_path = os.path.join(trained_models_root, checkpoint_file)
        if not os.path.isfile(checkpoint_path):
            rospy.logerr(f"Checkpoint file not found: {checkpoint_path}")
            env.close()
            exit(1)
        
        rospy.logwarn(f"Loading trained model from: {checkpoint_path}")
        max_avg_reward = checkpoint_manager.load_checkpoint(checkpoint_path, policy_net, target_net)
    
    # Warm-start replay memory before training
    MIN_REPLAY_SIZE = batch_size * 15
    rospy.logwarn("=== START WARM UP ===")
    
    warm_start_obs = env.reset()
    warm_start_state = torch.tensor(warm_start_obs, device=device, dtype=torch.float)

    while len(memory) < MIN_REPLAY_SIZE:
        action = torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
        observation, reward, done, _ = env.step(action.item())
        reward_tensor = torch.tensor([reward], device=device, dtype=torch.float32)

        if done:
            next_state = None
            memory.push(warm_start_state, action, next_state, reward_tensor)
            warm_start_obs = env.reset()
            warm_start_state = torch.tensor(warm_start_obs, device=device, dtype=torch.float)
        else:
            next_state = torch.tensor(observation, device=device, dtype=torch.float)
            memory.push(warm_start_state, action, next_state, reward_tensor)
            warm_start_state = next_state

    rospy.loginfo(f"✓ Replay memory warmed up with {len(memory)} experiences")
    
    highest_reward = 0
    
    for i_episode in range(start_episode, n_episodes):
        logger.log_episode_start(i_episode)

        max_episode_duration = 0
        max_distance_traveled = 0.0
        cumulated_reward = 0
        episode_distance = 0.0
        last_distance_check = 0.0
        previous_odom = None
        done = False
        
        # Initialize the environment and get first state of the robot
        observation = env.reset()
        state = torch.tensor(observation, device=device, dtype=torch.float)

        # iterates over steps
        for t in count():
            logger.log_step_start(t)
            action, epsilon = select_action(state, epsilon_start, epsilon_end, epsilon_decay)
            rospy.loginfo(f"Epsilon: {epsilon:.4f} | Step: {t} | Action: {action.item()}")
            observation, reward, done, info = env.step(action.item())
            rospy.logwarn(f"=== CURRENT REWARD: {reward} ===")
            
            training_manager.accumulate_breakdown(reward)
            
            cumulated_reward += reward
            
            # Prepare and publish data for action_graph.py
            # Format expected: [action_index, ..., total_reward, step_reward]
            action_msg = Float32MultiArray()
            action_msg.data = [float(action.item()), float(cumulated_reward), float(reward)]
            result_action_pub.publish(action_msg)
            
            try:
                current_odom = env.unwrapped.get_odom()
                if previous_odom is not None:
                    dx = current_odom.pose.pose.position.x - previous_odom.pose.pose.position.x
                    dy = current_odom.pose.pose.position.y - previous_odom.pose.pose.position.y
                    episode_distance += numpy.sqrt(dx*dx + dy*dy)
                previous_odom = current_odom
            except (AttributeError, TypeError, RuntimeError) as e:
                rospy.logwarn(f"Odometry unavailable: {e}")
                
            reward = torch.tensor([reward], device=device, dtype=torch.float32)

            if done:
                next_state = None
            else:
                next_state = torch.tensor(observation, device=device, dtype=torch.float)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            optimize_model(batch_size, gamma, tau)

            if done:
                episode_durations.append(t + 1)
                last_time_steps = numpy.append(last_time_steps, [int(t + 1)])
                current_eps = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1. * steps_done / epsilon_decay)
                training_manager.update_metrics(cumulated_reward, episode_distance, t, current_eps)
                
                if t + 1 > max_episode_duration:
                    max_episode_duration = t + 1
                    
                if episode_distance > max_distance_traveled:
                    max_distance_traveled = episode_distance
                
                break
            else:
                state = next_state

        current_eps = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1. * steps_done / epsilon_decay)
        logger.log_episode_end(i_episode, gamma, current_eps, cumulated_reward, episode_distance)
        
        # Publish episode results for result_graph visualization
        # data[0]: Average max Q-value from last batch
        # data[1]: Total episode reward
        # data[2]: Loss (if available)
        with torch.no_grad():
            if len(memory) > 0:
                # Calculate average max Q-value over recent experiences
                recent_states = [memory.memory[i][0] for i in range(max(0, len(memory)-100), len(memory))]
                if recent_states:
                    state_batch = torch.stack(recent_states)
                    avg_max_q = policy_net(state_batch).max(1)[0].mean().item()
                else:
                    avg_max_q = 0.0
            else:
                avg_max_q = 0.0
                

        result_msg = Float32MultiArray()
        # Send epsilon value
        result_msg.data = [float(avg_max_q), float(cumulated_reward), float(current_eps)]
        result_pub.publish(result_msg)
        if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward
        last_rewards.append(cumulated_reward)
        
        # Save best model when we have at least 50 episodes and current average beats historical best
        if len(last_rewards) == 50:
            current_avg_reward = numpy.mean(last_rewards)
            if current_avg_reward > max_avg_reward:
                max_avg_reward = current_avg_reward
                best_policy = policy_net
                final_model_path = checkpoint_manager.save_final_model(policy_net, max_avg_reward, f"best_model_stage{stage}", timestamp=False)
                rospy.loginfo(f"New best model saved! Avg reward: {max_avg_reward:.2f}")

        
        # Save periodic checkpoints
        if (i_episode + 1) % 500 == 0:
            plot_filename = training_manager.save_checkpoint_plots(i_episode)
            training_time = time.time() - logger.start_time
            final_model_path = checkpoint_manager.save_final_model(policy_net, max_avg_reward, f"checkpoint_model_stage{stage}")

    final_training_time = time.time() - logger.start_time
    
    reporter.write_training_results(final_training_time, highest_reward, last_time_steps)
    reporter.write_episode_statistics(training_manager.episode_rewards_history,
                                      training_manager.episode_durations_history,
                                      training_manager.episode_distances_history)
    
    logger.log_training_complete(max_episode_duration, max_distance_traveled, final_model_path, reporter.report_path)

    env.close()

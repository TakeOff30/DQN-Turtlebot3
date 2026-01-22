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

from utils import plot_training_metrics
from checkpoint_manager import CheckpointManager
from training_logger import TrainingLogger
from training_reporter import TrainingReporter
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


class DQN(nn.Module):

    def __init__(self, inputs, outputs):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(inputs, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.head = nn.Linear(64, outputs)
        
                
    def forward(self, x):
        x = x.to(device)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.head(x)


def select_action(state, eps_start, eps_end, eps_decay):
    global steps_done
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * \
        math.exp(-1. * steps_done / eps_decay)
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


def optimize_model(batch_size, gamma):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(
        tuple(s is not None for s in batch.next_state),
        device=device, dtype=torch.bool
    )
     # Stack only non-final next states (if any)
    non_final_next_states = None
    if non_final_mask.any():
        non_final_next_states = torch.stack(
            [s for s in batch.next_state if s is not None]
        )
    state_batch = torch.stack(batch.state)
    action_batch = torch.stack(batch.action)
    reward_batch = torch.stack(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t)
    state_action_values = policy_net(state_batch).gather(1, torch.squeeze(action_batch, 2))

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(batch_size, device=device)
    
    if non_final_next_states is not None:
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values.unsqueeze(1) * gamma) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


# import our training environment
if __name__ == '__main__':
  
    rospy.init_node('turtlebot3_world_qlearn', anonymous=True, log_level=rospy.INFO)

    # Init OpenAI_ROS ENV
    task_and_robot_environment_name = rospy.get_param(
        '/turtlebot3/task_and_robot_environment_name')
    env = StartOpenAI_ROS_Environment(
        task_and_robot_environment_name)
    # Create the Gym environment
    rospy.loginfo("Gym environment done")
    rospy.loginfo("Starting Learning")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('curriculum_learning')
    outdir = pkg_path + '/training_results'
    # env = wrappers.Monitor(env, outdir, force=True)
    rospy.loginfo("Monitor Wrapper started")

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
    target_update = rospy.get_param("/turtlebot3/target_update")
    lr = rospy.get_param("/turtlebot3/learning_rate", 0.001)
    running_step = rospy.get_param("/turtlebot3/running_step")
    
    # Log loaded hyperparameters for verification
    rospy.loginfo("=== DQN Hyperparameters Loaded ===")
    rospy.loginfo("Gamma: %.3f" % gamma)
    rospy.loginfo("Epsilon Start: %.3f" % epsilon_start)
    rospy.loginfo("Epsilon End: %.3f" % epsilon_end)
    rospy.loginfo("Epsilon Decay: %d" % epsilon_decay)
    rospy.loginfo("Episodes: %d" % n_episodes)
    rospy.loginfo("Batch Size: %d" % batch_size)
    rospy.loginfo("Target Update: %d" % target_update)
    rospy.loginfo("Learning Rate: %.5f" % lr)
    rospy.loginfo("==================================")
    
    # Create directories for outputs
    model_path = pkg_path + '/trained_models'
    reports_dir = pkg_path + '/training_reports'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
    rospy.loginfo("Model checkpoints will be saved to: " + model_path)
    rospy.loginfo("Training reports will be saved to: " + reports_dir)
    
    # Checkpoint Loading Parameters
    resume_training = rospy.get_param("/turtlebot3/resume_from_checkpoint", False)
    checkpoint_file = rospy.get_param("/turtlebot3/checkpoint_file", "best_model.pth")

    # Initialises the algorithm that we are going to use for learning if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward'))

    # Get number of actions from gym action space
    n_actions = env.action_space.n
    initial_obs = env.reset()
    assert env.observation_space.contains(initial_obs), \
         f"Observation {initial_obs} outside declared space {env.observation_space}"
    rospy.loginfo("✓ Observation space bounds match actual observations")
    n_observations = len(initial_obs)  # Dynamically get observation size

    # initialize networks with input and output sizes
    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    memory = ReplayMemory(50000)
    episode_durations = []
    steps_done = 0
    start_episode = 0
    
    # Initialize helper classes
    checkpoint_mgr = CheckpointManager(model_path)
    logger = TrainingLogger()
    reporter = TrainingReporter(reports_dir, model_path)
    
    # Warm-start replay memory before training
    MIN_REPLAY_SIZE = batch_size * 10  # At least 10 batches worth
    rospy.loginfo("=== Warming up replay memory ===")
    rospy.loginfo(f"Target: {MIN_REPLAY_SIZE} experiences")
    
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

    # Initialize ROS publishers for real-time metrics monitoring and visualization
    metrics_pub = rospy.Publisher('/training_metrics', Float32MultiArray, queue_size=10)
    result_pub = rospy.Publisher('/result', Float32MultiArray, queue_size=10)  # For result_graph.py
    action_pub = rospy.Publisher('/get_action', Float32MultiArray, queue_size=10)  # For action_graph.py
    
    highest_reward = 0
    
    # Training metrics tracking
    max_episode_duration = 0
    max_distance_traveled = 0.0
    episode_rewards_history = []
    episode_durations_history = []
    episode_distances_history = []
    episode_epsilon_history = []
    reward_breakdown_history = []  # Track reward components per episode
    
    # Starts the main training loop: the one about the episodes to do
    for i_episode in range(start_episode, n_episodes):
        logger.log_episode_start(i_episode)

        cumulated_reward = 0
        episode_distance = 0.0
        last_distance_check = 0.0
        previous_odom = None
        done = False
        
        # Track reward breakdown for this episode
        episode_breakdown = {
            'navigation': 0.0,
            'collision': 0.0,
            'success': 0.0,
            'failure': 0.0,
            'other': 0.0
        }

        # Initialize the environment and get first state of the robot
        observation = env.reset()
        state = torch.tensor(observation, device=device, dtype=torch.float)

        for t in count():
            logger.log_step_start(t)
            # Select and perform an action
            action, epsilon = select_action(state, epsilon_start, epsilon_end, epsilon_decay)

            observation, reward, done, info = env.step(action.item())
            rospy.logwarn(f"=== CURRENT REWARD: {reward}===")
            
            # Categorize reward for breakdown tracking
            if reward > 100:
                episode_breakdown['success'] += reward
            elif reward < -50:
                episode_breakdown['failure'] += reward
            elif reward < 0:
                episode_breakdown['collision'] += reward
            else:
                episode_breakdown['navigation'] += reward
            
            cumulated_reward += reward
            
            # Track distance traveled
            try:
                current_odom = env.unwrapped.get_odom()
                if previous_odom is not None:
                    dx = current_odom.pose.pose.position.x - previous_odom.pose.pose.position.x
                    dy = current_odom.pose.pose.position.y - previous_odom.pose.pose.position.y
                    episode_distance += numpy.sqrt(dx*dx + dy*dy)
                previous_odom = current_odom

            except (AttributeError, TypeError, RuntimeError) as e:
                rospy.logwarn(f"Odometry unavailable: {e}")
                pass

            # Check if likely stuck (e.g. not moving effectively)
            if t > 0 and t % 15 == 0:
                # Check if distance traveled in last 15 steps is significant (> 0.1m)
                if (episode_distance - last_distance_check) < 0.05:
                    logger.log_robot_stuck()
                    reward = -20
                    episode_breakdown['failure'] += -50
                    cumulated_reward += reward
                    done = True
                last_distance_check = episode_distance
                
            reward = torch.tensor([reward], device=device, dtype=torch.float32)

            if done:
                next_state = None
            else:
                next_state = torch.tensor(observation, device=device, dtype=torch.float)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Calculate average max Q-value for visualization
            with torch.no_grad():
                state_for_q = state.unsqueeze(0) if state.dim() == 1 else state
                avg_max_q = policy_net(state_for_q).max(1)[0].item()    
                       
            # Publish action and reward data for action_graph visualization
            action_msg = Float32MultiArray()
            action_msg.data = [float(action.item()), float(cumulated_reward), float(reward.item())]
            action_pub.publish(action_msg)

            optimize_model(batch_size, gamma)
                
            if done:
                episode_durations.append(t + 1)
                last_time_steps = numpy.append(last_time_steps, [int(t + 1)])
                
                # Track metrics
                episode_rewards_history.append(cumulated_reward)
                episode_durations_history.append(t + 1)
                episode_distances_history.append(episode_distance)
                reward_breakdown_history.append(episode_breakdown)

                # Track current epsilon
                current_eps = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1. * steps_done / epsilon_decay)
                episode_epsilon_history.append(current_eps)
                
                if t + 1 > max_episode_duration:
                    max_episode_duration = t + 1
                    
                if episode_distance > max_distance_traveled:
                    max_distance_traveled = episode_distance
                
                break
            else:
                state = next_state

        if (i_episode+1) % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())
                rospy.loginfo(f"Target network updated at episode: {i_episode+1}")

        # Calculate current epsilon for logging
        current_eps = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1. * steps_done / epsilon_decay)
        logger.log_episode_end(i_episode, gamma, current_eps, cumulated_reward, episode_distance)
        
        # Publish metrics for real-time monitoring and visualization
        metrics_msg = Float32MultiArray()
        metrics_msg.data = [float(i_episode), float(cumulated_reward), float(episode_distance)]
        metrics_pub.publish(metrics_msg)
        
        # Publish episode results for result_graph visualization
        # data[0]: Average max Q-value from last batch
        # data[1]: Total episode reward
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
        result_msg.data = [float(avg_max_q), float(cumulated_reward)]
        result_pub.publish(result_msg)
        
        # Save best model when new highest reward is achieved
        if cumulated_reward > highest_reward:
            best_model_data = {
                'episode': i_episode,
                'policy_net_state_dict': policy_net.state_dict(),
                'target_net_state_dict': target_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'reward': cumulated_reward,
                'steps_done': steps_done,
            }
            best_model_path = os.path.join(model_path, "checkpoint_best.pth")
            torch.save(best_model_data, best_model_path)
            rospy.loginfo(f"New best model saved! Reward: {cumulated_reward:.2f} at episode {i_episode}")
        
        if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward
            
        
        # Save periodic checkpoints every 500 episodes
        if (i_episode + 1) % 500 == 0:
            # Convert reward breakdown from list-of-dicts to dict-of-lists for plotting
            breakdown_dict = {}
            if reward_breakdown_history:
                # Initialize with keys from first episode
                for key in reward_breakdown_history[0].keys():
                    breakdown_dict[key] = [ep[key] for ep in reward_breakdown_history]
            
            # Generate and save plot
            plot_filename = reporter.generate_plot(i_episode + 1, outdir, episode_rewards_history,
                                                   episode_durations_history, episode_distances_history,
                                                   episode_epsilon_history, breakdown_dict)
            
            # Save checkpoint
            checkpoint_mgr.save_checkpoint(i_episode, policy_net, target_net, optimizer,
                                          cumulated_reward, steps_done, episode_rewards_history,
                                          reward_breakdown_history, episode_durations_history, episode_distances_history,
                                          episode_epsilon_history)
            
            logger.log_checkpoint_saved(i_episode, plot_filename)

    # Log final scores
    logger.log_final_scores(n_episodes, gamma, epsilon_start, epsilon_decay, highest_reward,
                           last_time_steps, episode_rewards_history)

    # Save final model
    final_training_time = time.time() - logger.start_time
    final_model_path = checkpoint_mgr.save_final_model(n_episodes, policy_net, target_net, optimizer,
                                                        cumulated_reward, highest_reward, max_episode_duration,
                                                        max_distance_traveled, final_training_time, gamma,
                                                        epsilon_start, epsilon_end, epsilon_decay)
    
    # Generate comprehensive training report
    report_path = reporter.generate_text_report(
        n_episodes, gamma, epsilon_start, epsilon_end, epsilon_decay, batch_size, target_update,
        final_training_time, highest_reward, episode_rewards_history, episode_durations_history,
        episode_distances_history, max_episode_duration, max_distance_traveled, last_time_steps
    )
    
    logger.log_training_complete(max_episode_duration, max_distance_traveled, final_model_path, report_path)

    env.close()

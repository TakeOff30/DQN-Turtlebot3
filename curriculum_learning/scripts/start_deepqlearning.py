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


class DQN(nn.Module):

    def __init__(self, inputs, outputs, resume_training=False):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(inputs, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.head = nn.Linear(32, outputs)
        
        # HE initialization
        if resume_training == False:
            nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='leaky_relu')
            nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='leaky_relu')
            nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='leaky_relu')
            nn.init.xavier_uniform_(self.head.weight)
            nn.init.zeros_(self.fc1.bias)
            nn.init.zeros_(self.fc2.bias)
            nn.init.zeros_(self.fc3.bias)
            nn.init.zeros_(self.head.bias)
        
                
    def forward(self, x):
        x = x.to(device)

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
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
    global loss_values, last_loss_value
    if len(memory) < batch_size:
        last_loss_value = None
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
    expected_state_action_values = (next_state_values.unsqueeze(1) * gamma) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    loss_values.append(loss.item())
    last_loss_value = loss.item()

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

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

    # Create directories for outputs
    model_path = pkg_path + '/trained_models'
    reports_dir = pkg_path + '/training_reports'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)


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
    resume_training = rospy.get_param("/turtlebot3/load_pretrained_model", False)
    checkpoint_file = rospy.get_param("/turtlebot3/checkpoint_file", "best_model.pth")
    stage = rospy.get_param("/turtlebot3/stage")
    
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

    policy_net = DQN(n_observations, n_actions, resume_training=resume_training).to(device)
    target_net = DQN(n_observations, n_actions, resume_training=resume_training).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
        
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    memory = ReplayMemory(50000)
    episode_durations = []
    steps_done = 0
    start_episode = 0
    loss_values = []
    last_loss_value = None
    last_rewards = deque([], maxlen=50)
    max_avg_reward = 0
    
    checkpoint_manager = CheckpointManager(model_path)
    logger = TrainingLogger()
    reporter = TrainingReporter(reports_dir, model_path)
    training_manager = TrainingManager(checkpoint_manager, reporter)
    
    if resume_training:
        checkpoint_path = os.path.join(model_path, checkpoint_file)
        if not os.path.isfile(checkpoint_path):
            rospy.logerr(f"Checkpoint file not found: {checkpoint_path}")
            env.close()
            exit(1)
        
        rospy.loginfo(f"Loading trained model from: {checkpoint_path}")
        max_avg_reward = checkpoint_manager.load_checkpoint(checkpoint_path, policy_net, target_net)
        # checkpoint = torch.load(checkpoint_path, map_location=device)
        # max_avg_reward = checkpoint['max_avg_reward']
        # policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        # target_net.load_state_dict(policy_net.state_dict())
        # target_net.eval()
        rospy.loginfo("Model loaded successfully!")
    
    # Warm-start replay memory before training
    MIN_REPLAY_SIZE = batch_size * 10
    rospy.loginfo("=== Warming up replay memory ===")
    rospy.loginfo(f"Target: {MIN_REPLAY_SIZE} experiences")
    warm_start_obs = env.reset()
    warm_start_state = torch.tensor(warm_start_obs, device=device, dtype=torch.float)

    rospy.logwarn("=== START WARM UP ===")

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

            optimize_model(batch_size, gamma)
                
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

        # update target network
        if (i_episode+1) % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())
                rospy.loginfo(f"Target network updated at episode: {i_episode+1}")

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
        # Send loss if available, else omit
        if last_loss_value is not None:
            result_msg.data = [float(avg_max_q), float(cumulated_reward), float(last_loss_value)]
        else:
            result_msg.data = [float(avg_max_q), float(cumulated_reward)]
        result_pub.publish(result_msg)
        
        if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward
        last_rewards.append(cumulated_reward)
        if len(last_rewards) == 50 and numpy.mean(last_rewards) > max_avg_reward:
            best_policy = policy_net
            final_model_path = checkpoint_manager.save_final_model(policy_net, max_avg_reward, f"best_model_stage{stage}", timestamp=False)

        
        # Save periodic checkpoints
        if (i_episode + 1) % 500 == 0:
            plot_filename = training_manager.save_checkpoint_plots(i_episode, outdir)
            training_time = time.time() - logger.start_time
            final_model_path = checkpoint_manager.save_final_model(policy_net, max_avg_reward, f"checkpoint_model_stage{stage}")

    final_training_time = time.time() - logger.start_time
    reporter.write_header()
    reporter.write_configuration(n_episodes, gamma, epsilon_start, epsilon_end, epsilon_decay, batch_size, target_update)
    reporter.write_training_results(final_training_time, highest_reward, last_time_steps)
    reporter.write_episode_statistics(training_manager.episode_rewards_history,
                                      training_manager.episode_durations_history,
                                      training_manager.episode_distances_history)
    
    logger.log_training_complete(max_episode_duration, max_distance_traveled, final_model_path, reporter.report_path)

    env.close()

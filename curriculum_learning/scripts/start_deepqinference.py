#!/usr/bin/env python3
"""
DQN Inference Script for TurtleBot3 Navigation

This script loads a trained DQN model and runs evaluation episodes
without any training or network updates.
"""

import gym
import numpy
import math
import random
from collections import namedtuple
from itertools import count
import os

# ROS packages required
import rospy
import rospkg
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, inputs, outputs):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(inputs, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.head = nn.Linear(32, outputs)
        
        # HE initialization
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


if __name__ == '__main__':
  
    rospy.init_node('turtlebot3_world_inference', anonymous=True, log_level=rospy.INFO)

    # Init OpenAI_ROS ENV
    task_and_robot_environment_name = rospy.get_param(
        '/turtlebot3/task_and_robot_environment_name')
    env = StartOpenAI_ROS_Environment(
        task_and_robot_environment_name)
    
    rospy.loginfo("Gym environment done")
    rospy.loginfo("Starting Inference")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('curriculum_learning')
    model_path = pkg_path + '/trained_models'

    # Load inference parameters
    checkpoint_file = rospy.get_param("/turtlebot3/best_model", "best_model.pth")
    n_eval_episodes = rospy.get_param("/turtlebot3/n_episodes", 100)
    success_reward_threshold = rospy.get_param("/turtlebot3/success_reward_threshold", 500)
    
    rospy.loginfo("=== Inference Settings ===")
    rospy.loginfo("Model used: %s" % checkpoint_file)
    rospy.loginfo("Number of evaluation episodes: %d" % n_eval_episodes)
    rospy.loginfo("==========================")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rospy.loginfo(f"Using device: {device}")

    # Get environment dimensions
    n_actions = env.action_space.n
    initial_obs = env.reset()
    n_observations = len(initial_obs)

    # Initialize policy network
    policy_net = DQN(n_observations, n_actions).to(device)
    policy_net.eval()  # Set to evaluation mode

    # Load trained model
    checkpoint_path = os.path.join(model_path, checkpoint_file)
    if not os.path.isfile(checkpoint_path):
        rospy.logerr(f"Checkpoint file not found: {checkpoint_path}")
        rospy.logerr("Cannot run inference without a trained model!")
        env.close()
        exit(1)
    
    rospy.loginfo(f"Loading trained model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    rospy.loginfo("Model loaded successfully!")
    
    # Display model info if available
    if 'episode' in checkpoint:
        rospy.loginfo(f"Model trained for episode: {checkpoint['episode']}")
    if 'reward' in checkpoint:
        rospy.loginfo(f"Model best reward: {checkpoint['reward']:.2f}")

    # Run inference episodes
    rospy.loginfo("="*50)
    rospy.loginfo("RUNNING INFERENCE (NO TRAINING)")
    rospy.loginfo("="*50)
    
    eval_rewards = []
    eval_successes = 0
    eval_distances = []
    eval_steps = []
    
    for i_episode in range(n_eval_episodes):
        rospy.loginfo(f"\n=== Evaluation Episode {i_episode + 1}/{n_eval_episodes} ===")
        
        cumulated_reward = 0
        episode_distance = 0.0
        previous_odom = None
        done = False
        
        observation = env.reset()
        state = torch.tensor(observation, device=device, dtype=torch.float)
        
        for t in count():
            # Select action greedily (no exploration)
            with torch.no_grad():
                state_batch = state.unsqueeze(0) if state.dim() == 1 else state
                action = policy_net(state_batch).max(1)[1].view(1, 1)            
           
            observation, reward, done, info = env.step(action.item())
            cumulated_reward += reward
            
            # Track distance traveled
            try:
                current_odom = env.unwrapped.get_odom()
                if previous_odom is not None:
                    dx = current_odom.pose.pose.position.x - previous_odom.pose.pose.position.x
                    dy = current_odom.pose.pose.position.y - previous_odom.pose.pose.position.y
                    episode_distance += math.sqrt(dx**2 + dy**2)
                previous_odom = current_odom
            except (AttributeError, TypeError, RuntimeError):
                pass
            
            if done:
                eval_rewards.append(cumulated_reward)
                eval_distances.append(episode_distance)
                eval_steps.append(t + 1)
                
                if cumulated_reward > success_reward_threshold:  # Success threshold
                    eval_successes += 1
                    rospy.loginfo("✓ Goal reached!")
                else:
                    rospy.loginfo("✗ Episode failed")
                
                rospy.loginfo(f"Episode reward: {cumulated_reward:.2f}")
                rospy.loginfo(f"Distance traveled: {episode_distance:.2f}m")
                rospy.loginfo(f"Steps taken: {t + 1}")
                break
            
            state = torch.tensor(observation, device=device, dtype=torch.float)
    
    # Print evaluation summary
    avg_reward = sum(eval_rewards) / len(eval_rewards) if eval_rewards else 0
    avg_distance = sum(eval_distances) / len(eval_distances) if eval_distances else 0
    avg_steps = sum(eval_steps) / len(eval_steps) if eval_steps else 0
    success_rate = (eval_successes / n_eval_episodes) * 100
    
    rospy.loginfo("\n" + "="*50)
    rospy.loginfo("EVALUATION COMPLETE")
    rospy.loginfo("="*50)
    rospy.loginfo(f"Episodes: {n_eval_episodes}")
    rospy.loginfo(f"Success rate: {success_rate:.1f}% ({eval_successes}/{n_eval_episodes})")
    rospy.loginfo(f"Average reward: {avg_reward:.2f}")
    rospy.loginfo(f"Average distance: {avg_distance:.2f}m")
    rospy.loginfo(f"Average steps: {avg_steps:.1f}")
    rospy.loginfo(f"Best reward: {max(eval_rewards):.2f}")
    rospy.loginfo(f"Worst reward: {min(eval_rewards):.2f}")
    rospy.loginfo("="*50)
    
    env.close()

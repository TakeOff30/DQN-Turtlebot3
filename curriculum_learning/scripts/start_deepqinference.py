#!/usr/bin/env python3
"""
DQN Inference Script for TurtleBot3 Navigation

Loads a trained Dueling DQN model and runs evaluation episodes.
In inference mode the episode does NOT reset when the robot reaches a goal;
instead a new goal is spawned and the robot keeps navigating until collision
or max_episode_steps is hit.

Metrics reported:
  - Highest number of goals reached in a single episode
  - Percentage of episodes where the robot reached at least one goal
  - Average / min / max goals per episode
"""

import gym
import numpy
import math
from itertools import count
import os

# ROS packages required
import rospy
import rospkg
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment

import torch
import torch.nn as nn

class DuelingDQN(nn.Module):
    """Dueling DQN: separates Value and Advantage streams."""

    def __init__(self, inputs, outputs):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(inputs, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, outputs),
        )

    def forward(self, x):
        x = x.to(device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

if __name__ == '__main__':

    rospy.init_node('turtlebot3_world_inference', anonymous=True, log_level=rospy.INFO)

    # Init OpenAI_ROS ENV
    task_and_robot_environment_name = rospy.get_param(
        '/turtlebot3/task_and_robot_environment_name')
    env = StartOpenAI_ROS_Environment(task_and_robot_environment_name)

    rospy.loginfo("Gym environment done")
    rospy.loginfo("Starting Inference")

    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('curriculum_learning')
    trained_models_root = os.path.join(pkg_path, 'trained_models')

    # Load inference parameters
    model_file = rospy.get_param("/turtlebot3/best_model", "best_model_stage1.pth")
    n_eval_episodes = rospy.get_param("/turtlebot3/n_episodes", 100)

    rospy.loginfo("=== Inference Settings ===")
    rospy.loginfo("Model: %s" % model_file)
    rospy.loginfo("Evaluation episodes: %d" % n_eval_episodes)
    rospy.loginfo("==========================")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rospy.loginfo("Using device: %s" % device)

    # Get environment dimensions
    n_actions = env.action_space.n
    initial_obs = env.reset()
    n_observations = len(initial_obs)

    # Initialize & load policy network
    policy_net = DuelingDQN(n_observations, n_actions).to(device)
    policy_net.eval()

    model_path = os.path.join(trained_models_root, model_file)
    if not os.path.isfile(model_path):
        rospy.logerr("Model file not found: %s" % model_path)
        env.close()
        exit(1)

    rospy.loginfo("Loading trained model from: %s" % model_path)
    checkpoint = torch.load(model_path, map_location=device)
    policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    if 'max_avg_reward' in checkpoint:
        rospy.loginfo("Model avg reward at save: %.2f" % checkpoint['max_avg_reward'])
    rospy.loginfo("Model loaded successfully!")

    rospy.loginfo("=" * 50)
    rospy.loginfo("RUNNING INFERENCE – goals do NOT end the episode")
    rospy.loginfo("=" * 50)

    episode_goals = []       # goals reached per episode
    episode_distances = []   # distance traveled per episode
    episode_steps_list = []  # steps per episode

    for i_episode in range(n_eval_episodes):
        rospy.loginfo("\n=== Evaluation Episode %d/%d ===" % (i_episode + 1, n_eval_episodes))

        episode_distance = 0.0
        previous_odom = None
        done = False

        observation = env.reset()
        state = torch.tensor(observation, device=device, dtype=torch.float)

        for t in count():
            # Greedy action selection (no exploration)
            with torch.no_grad():
                action = policy_net(state).max(1)[1].view(1, 1)

            observation, reward, done, info = env.step(action.item())

            # Track distance traveled
            try:
                current_odom = env.unwrapped.get_odom()
                if previous_odom is not None:
                    dx = current_odom.pose.pose.position.x - previous_odom.pose.pose.position.x
                    dy = current_odom.pose.pose.position.y - previous_odom.pose.pose.position.y
                    episode_distance += math.sqrt(dx ** 2 + dy ** 2)
                previous_odom = current_odom
            except (AttributeError, TypeError, RuntimeError):
                pass

            if done:
                # Retrieve goals reached from the environment
                goals = getattr(env.unwrapped, 'goals_reached_count', 0)
                episode_goals.append(goals)
                episode_distances.append(episode_distance)
                episode_steps_list.append(t + 1)

                if goals == 3:
                    rospy.loginfo("✓ Goal reached")
                else:
                    rospy.loginfo("✗ No goal reached")
                rospy.loginfo("Distance: %.2fm  |  Steps: %d" % (episode_distance, t + 1))
                break

            state = torch.tensor(observation, device=device, dtype=torch.float)

    goals_array = numpy.array(episode_goals)
    successful_episodes = int(numpy.sum(goals_array == 3))
    success_rate = (successful_episodes / n_eval_episodes) * 100.0
    avg_goals = numpy.mean(goals_array) if len(goals_array) > 0 else 0
    avg_distance = numpy.mean(episode_distances) if episode_distances else 0
    avg_steps = numpy.mean(episode_steps_list) if episode_steps_list else 0

    rospy.loginfo("\n" + "=" * 60)
    rospy.loginfo("EVALUATION COMPLETE")
    rospy.loginfo("=" * 60)
    rospy.loginfo("Episodes evaluated       : %d" % n_eval_episodes)
    rospy.loginfo("Success rate             : %.1f%% (%d/%d)" % (success_rate, successful_episodes, n_eval_episodes))
    rospy.loginfo("Average goals per episode: %.2f" % avg_goals)
    rospy.loginfo("Average distance         : %.2fm" % avg_distance)
    rospy.loginfo("Average steps            : %.1f" % avg_steps)
    if len(goals_array) > 0:
        rospy.loginfo("Goals distribution       : min=%d  median=%d  max=%d" %
                      (int(numpy.min(goals_array)), int(numpy.median(goals_array)), max_goals))
    rospy.loginfo("=" * 60)

    env.close()

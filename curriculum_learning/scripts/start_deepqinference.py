#!/usr/bin/env python3
"""
DQN Inference Script for TurtleBot3 Navigation

Loads a trained Dueling DQN model and runs evaluation episodes.
<<<<<<< HEAD
=======
In inference mode the episode does NOT reset when the robot reaches a goal;
instead a new goal is spawned and the robot keeps navigating until collision
or max_episode_steps is hit.
>>>>>>> origin/Nicola

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
<<<<<<< HEAD
from models.dqn import DQN
from models.dueling_dqn import DuelingDQN
from inference_reporter import InferenceReporter


def infer_checkpoint_dims(state_dict, model_type):
    """Infer input/output dimensions from checkpoint state_dict."""
    try:
        input_dim = state_dict['feature.0.weight'].shape[1]
        if model_type == 'dqn':
            output_dim = state_dict['fc.2.weight'].shape[0]
        else:
            output_dim = state_dict['advantage_stream.2.weight'].shape[0]
        return int(input_dim), int(output_dim)
    except KeyError as e:
        raise RuntimeError(f"Missing expected key in checkpoint state_dict: {e}")


def validate_observation(observation, expected_dim, context="observation"):
    """Validate observation format and fail fast on mismatch."""
    obs = numpy.asarray(observation, dtype=numpy.float32)

    if obs.ndim != 1:
        raise ValueError(
            f"Invalid {context} shape: expected 1D vector, got shape {obs.shape}"
        )

    if obs.shape[0] != expected_dim:
        raise ValueError(
            f"Invalid {context} size: expected {expected_dim}, got {obs.shape[0]}"
        )

    if not numpy.all(numpy.isfinite(obs)):
        raise ValueError(f"Invalid {context}: contains NaN or Inf values")

    return obs

=======


# ── Must match the architecture used during training ──────────────────────────

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
>>>>>>> origin/Nicola

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
<<<<<<< HEAD
    inference_reports_root = os.path.join(pkg_path, 'scripts', 'inference_reports')

    # Load inference parameters
    model_file = rospy.get_param(
        "/turtlebot3/checkpoint_file",
        rospy.get_param("/turtlebot3/best_model", "best_model_stage5.pth")
    )
    n_eval_episodes = rospy.get_param("/turtlebot3/n_episodes", 100)
    model_type = rospy.get_param("/turtlebot3/model_type", "dueling_dqn")
=======

    # Load inference parameters
    model_file = rospy.get_param("/turtlebot3/best_model", "best_model_stage1.pth")
    n_eval_episodes = rospy.get_param("/turtlebot3/n_episodes", 100)
>>>>>>> origin/Nicola

    rospy.loginfo("=== Inference Settings ===")
    rospy.loginfo("Model: %s" % model_file)
    rospy.loginfo("Evaluation episodes: %d" % n_eval_episodes)
    rospy.loginfo("==========================")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rospy.loginfo("Using device: %s" % device)
<<<<<<< HEAD

    reporter = InferenceReporter(inference_reports_root)
    reporter.write_header()
    reporter.write_configuration(model_file, n_eval_episodes, model_type, str(device))
    rospy.loginfo("Inference report: %s" % reporter.report_path)
=======
>>>>>>> origin/Nicola

    # Get environment dimensions
    n_actions = env.action_space.n
    initial_obs = env.reset()
    env_observations = len(initial_obs)

<<<<<<< HEAD
    if model_type not in ('dqn', 'dueling_dqn'):
        rospy.logerr(f"Unknown model type: {model_type}")
        env.close()
        exit(1)

    model_path = os.path.join(trained_models_root, model_file)
    if not os.path.isfile(model_path):
        rospy.logerr("Model file not found: %s" % model_path)
        env.close()
        exit(1)

    rospy.loginfo("Loading trained model from: %s" % model_path)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get('policy_net_state_dict', checkpoint)

    try:
        checkpoint_input_dim, checkpoint_output_dim = infer_checkpoint_dims(state_dict, model_type)
    except RuntimeError as e:
        rospy.logerr(str(e))
        env.close()
        exit(1)

    if checkpoint_output_dim != n_actions:
        rospy.logerr(
            "Action-space mismatch: checkpoint outputs %d actions, env expects %d. "
            "Use a checkpoint trained with the current action space.",
            checkpoint_output_dim, n_actions
        )
        env.close()
        exit(1)

    if model_type == 'dqn':
        policy_net = DQN(checkpoint_input_dim, checkpoint_output_dim).to(device)
    else:
        policy_net = DuelingDQN(checkpoint_input_dim, checkpoint_output_dim).to(device)

    policy_net.load_state_dict(state_dict)
    policy_net.eval()

    if checkpoint_input_dim != env_observations:
        rospy.logerr(
            "Observation-size mismatch: checkpoint expects %d, env provides %d.",
            checkpoint_input_dim, env_observations
        )
        env.close()
        raise RuntimeError(
            f"Observation format mismatch: model input={checkpoint_input_dim}, "
            f"env output={env_observations}. Use a compatible checkpoint/config."
        )

=======
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
>>>>>>> origin/Nicola
    if 'max_avg_reward' in checkpoint:
        rospy.loginfo("Model avg reward at save: %.2f" % checkpoint['max_avg_reward'])
    rospy.loginfo("Model loaded successfully!")

    rospy.loginfo("=" * 50)
<<<<<<< HEAD
    rospy.loginfo("RUNNING INFERENCE")
    rospy.loginfo("=" * 50)

    episode_goals = [] # goals reached per episode
=======
    rospy.loginfo("RUNNING INFERENCE – goals do NOT end the episode")
    rospy.loginfo("=" * 50)

    episode_goals = []       # goals reached per episode
    episode_distances = []   # distance traveled per episode
    episode_steps_list = []  # steps per episode
>>>>>>> origin/Nicola

    for i_episode in range(n_eval_episodes):
        rospy.loginfo("\n=== Evaluation Episode %d/%d ===" % (i_episode + 1, n_eval_episodes))

<<<<<<< HEAD
        done = False

        observation = env.reset()
        validated_observation = validate_observation(
            observation,
            checkpoint_input_dim,
            context=f"episode {i_episode + 1} initial observation",
        )
        state = torch.tensor(validated_observation, device=device, dtype=torch.float)
=======
        episode_distance = 0.0
        previous_odom = None
        done = False

        observation = env.reset()
        state = torch.tensor(observation, device=device, dtype=torch.float)
>>>>>>> origin/Nicola

        for t in count():
            # Greedy action selection (no exploration)
            with torch.no_grad():
                action = policy_net(state).max(1)[1].view(1, 1)

            observation, reward, done, info = env.step(action.item())

<<<<<<< HEAD
            if done:
                goals = getattr(env.unwrapped, 'goals_reached_count', 0)
                episode_goals.append(goals)
                reporter.append_episode_result(i_episode, n_eval_episodes, goals)
                break

            validated_observation = validate_observation(
                observation,
                checkpoint_input_dim,
                context=f"episode {i_episode + 1} step {t + 1} observation",
            )
            state = torch.tensor(validated_observation, device=device, dtype=torch.float)

    goals_array = numpy.array(episode_goals)
    successful_episodes = int(numpy.sum(goals_array == 3))
    success_rate = (successful_episodes / n_eval_episodes) * 100.0
    avg_goals = numpy.mean(goals_array) if len(goals_array) > 0 else 0

    rospy.loginfo("=" * 60)
    rospy.loginfo("EVALUATION COMPLETE")
    rospy.loginfo("Episodes evaluated       : %d" % n_eval_episodes)
    rospy.loginfo("Success rate             : %.1f%% (%d/%d)" % (success_rate, successful_episodes, n_eval_episodes))
    rospy.loginfo("Average goals per episode: %.2f" % avg_goals)

    reporter.write_summary(episode_goals, n_eval_episodes)
=======
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

                if goals > 0:
                    rospy.loginfo("✓ Goals reached: %d" % goals)
                else:
                    rospy.loginfo("✗ No goal reached")
                rospy.loginfo("Distance: %.2fm  |  Steps: %d" % (episode_distance, t + 1))
                break

            state = torch.tensor(observation, device=device, dtype=torch.float)

    goals_array = numpy.array(episode_goals)
    successful_episodes = int(numpy.sum(goals_array >= 1))
    success_rate = (successful_episodes / n_eval_episodes) * 100.0
    max_goals = int(numpy.max(goals_array)) if len(goals_array) > 0 else 0
    avg_goals = numpy.mean(goals_array) if len(goals_array) > 0 else 0
    avg_distance = numpy.mean(episode_distances) if episode_distances else 0
    avg_steps = numpy.mean(episode_steps_list) if episode_steps_list else 0

    rospy.loginfo("\n" + "=" * 60)
    rospy.loginfo("EVALUATION COMPLETE")
    rospy.loginfo("=" * 60)
    rospy.loginfo("Episodes evaluated       : %d" % n_eval_episodes)
    rospy.loginfo("Success rate (≥1 goal)   : %.1f%% (%d/%d)" % (success_rate, successful_episodes, n_eval_episodes))
    rospy.loginfo("Highest goals in 1 ep    : %d" % max_goals)
    rospy.loginfo("Average goals per episode: %.2f" % avg_goals)
    rospy.loginfo("Average distance         : %.2fm" % avg_distance)
    rospy.loginfo("Average steps            : %.1f" % avg_steps)
    if len(goals_array) > 0:
        rospy.loginfo("Goals distribution       : min=%d  median=%d  max=%d" %
                      (int(numpy.min(goals_array)), int(numpy.median(goals_array)), max_goals))
    rospy.loginfo("=" * 60)
>>>>>>> origin/Nicola

    env.close()

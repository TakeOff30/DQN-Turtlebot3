#!/usr/bin/env python3
"""
DQN Inference Script for TurtleBot3 Navigation

Loads a trained Dueling DQN model and runs evaluation episodes.

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


def adapt_observation(observation, target_dim):
    """Adapt observation vector to target model input dimension.
    Preserves the final 3 goal-related features when possible.
    """
    obs = numpy.asarray(observation, dtype=numpy.float32)
    source_dim = obs.shape[0]

    if source_dim == target_dim:
        return obs

    # Downsample when current observation is larger than model input
    if source_dim > target_dim:
        if source_dim >= 3 and target_dim >= 3:
            laser_source = source_dim - 3
            laser_target = target_dim - 3
            if laser_target > 0 and laser_source > 0:
                idx = numpy.linspace(0, laser_source - 1, num=laser_target, dtype=int)
                adapted = numpy.concatenate([obs[:laser_source][idx], obs[-3:]])
                return adapted.astype(numpy.float32)
        return obs[:target_dim].astype(numpy.float32)

    # Pad when current observation is smaller than model input
    padded = numpy.zeros((target_dim,), dtype=numpy.float32)
    padded[:source_dim] = obs
    return padded



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
    inference_reports_root = os.path.join(pkg_path, 'scripts', 'inference_reports')

    # Load inference parameters
    model_file = rospy.get_param(
        "/turtlebot3/checkpoint_file",
        rospy.get_param("/turtlebot3/best_model", "best_model_stage5.pth")
    )
    n_eval_episodes = rospy.get_param("/turtlebot3/n_episodes", 100)
    model_type = rospy.get_param("/turtlebot3/model_type", "dueling_dqn")

    rospy.loginfo("=== Inference Settings ===")
    rospy.loginfo("Model: %s" % model_file)
    rospy.loginfo("Evaluation episodes: %d" % n_eval_episodes)
    rospy.loginfo("==========================")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rospy.loginfo("Using device: %s" % device)

    reporter = InferenceReporter(inference_reports_root)
    reporter.write_header()
    reporter.write_configuration(model_file, n_eval_episodes, model_type, str(device))
    rospy.loginfo("Inference report: %s" % reporter.report_path)

    # Get environment dimensions
    n_actions = env.action_space.n
    initial_obs = env.reset()
    env_observations = len(initial_obs)

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
        rospy.logwarn(
            "Observation-size mismatch: checkpoint expects %d, env provides %d. "
            "Applying observation adaptation during inference.",
            checkpoint_input_dim, env_observations
        )

    if 'max_avg_reward' in checkpoint:
        rospy.loginfo("Model avg reward at save: %.2f" % checkpoint['max_avg_reward'])
    rospy.loginfo("Model loaded successfully!")

    rospy.loginfo("=" * 50)
    rospy.loginfo("RUNNING INFERENCE")
    rospy.loginfo("=" * 50)

    episode_goals = []       # goals reached per episode

    for i_episode in range(n_eval_episodes):
        rospy.loginfo("\n=== Evaluation Episode %d/%d ===" % (i_episode + 1, n_eval_episodes))

        done = False

        observation = env.reset()
        adapted_observation = adapt_observation(observation, checkpoint_input_dim)
        state = torch.tensor(adapted_observation, device=device, dtype=torch.float)

        for t in count():
            # Greedy action selection (no exploration)
            with torch.no_grad():
                action = policy_net(state).max(1)[1].view(1, 1)

            observation, reward, done, info = env.step(action.item())

            if done:
                # Retrieve goals reached from the environment
                goals = getattr(env.unwrapped, 'goals_reached_count', 0)
                episode_goals.append(goals)
                reporter.append_episode_result(i_episode, n_eval_episodes, goals)
                break

            adapted_observation = adapt_observation(observation, checkpoint_input_dim)
            state = torch.tensor(adapted_observation, device=device, dtype=torch.float)

    goals_array = numpy.array(episode_goals)
    successful_episodes = int(numpy.sum(goals_array == 3))
    success_rate = (successful_episodes / n_eval_episodes) * 100.0
    avg_goals = numpy.mean(goals_array) if len(goals_array) > 0 else 0

    rospy.loginfo("\n" + "=" * 60)
    rospy.loginfo("EVALUATION COMPLETE")
    rospy.loginfo("=" * 60)
    rospy.loginfo("Episodes evaluated       : %d" % n_eval_episodes)
    rospy.loginfo("Success rate             : %.1f%% (%d/%d)" % (success_rate, successful_episodes, n_eval_episodes))
    rospy.loginfo("Average goals per episode: %.2f" % avg_goals)
    rospy.loginfo("=" * 60)

    reporter.write_summary(episode_goals, n_eval_episodes)

    env.close()

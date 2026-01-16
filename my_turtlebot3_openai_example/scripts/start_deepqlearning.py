#!/usr/bin/env python3

import gym
import numpy
import math
import random
import matplotlib.pyplot as plt
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


def plot_training_metrics(rewards, durations, distances, epsilons, reward_breakdown, filename):
    plt.figure(figsize=(20, 15))

    # 1. Rewards
    plt.subplot(3, 2, 1)
    plt.plot(rewards, color='blue', alpha=0.3)
    # Calculate moving average
    if len(rewards) >= 50:
        moving_avg = [sum(rewards[i-50:i])/50 for i in range(50, len(rewards))]
        plt.plot(range(50, len(rewards)), moving_avg, color='red', label='50-Ep Avg')
    plt.title('Rewards per Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)

    # 2. Durations (Steps)
    plt.subplot(3, 2, 2)
    plt.plot(durations, color='green')
    plt.title('Duration (Steps) per Episode')
    plt.ylabel('Steps')
    plt.grid(True)

    # 3. Distances
    plt.subplot(3, 2, 3)
    plt.plot(distances, color='orange')
    plt.title('Distance Traveled (m)')
    plt.ylabel('Meters')
    plt.grid(True)

    # 4. Success Rate (Approximate based on Reward/Duration)
    plt.subplot(3, 2, 4)
    # Binary success (1 if survived 800 steps or high reward, 0 otherwise)
    # We use a simple heuristic here, but we could use the 'survival' component from breakdown
    successes = [1 if (d >= 799 or r > 400) else 0 for d, r in zip(durations, rewards)]
    if len(successes) >= 50:
        success_rate = [sum(successes[i-50:i])/50.0 for i in range(50, len(successes))]
        plt.plot(range(50, len(successes)), success_rate, color='purple')
    plt.title('Success Rate (Rolling 50-Ep)')
    plt.ylabel('Rate (0-1)')
    plt.ylim(0, 1.1)
    plt.grid(True)
    
    # 5. Reward Breakdown
    plt.subplot(3, 1, 3)
    for key, values in reward_breakdown.items():
        if len(values) > 0:
            # Plot moving average for clarity
            if len(values) >= 50:
                moving_avg = [sum(values[i-50:i])/50 for i in range(50, len(values))]
                plt.plot(range(50, len(values)), moving_avg, label=key)
            else:
                plt.plot(values, label=key, alpha=0.5)
    
    plt.title('Reward Components Contribution (Rolling 50-Ep)')
    plt.ylabel('Reward Value')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


class GridExplorer:
    def __init__(self, x_min=-10.0, x_max=10.0, y_min=-10.0, y_max=10.0, step_size=0.5):
        self.x_min = x_min
        self.y_min = y_min
        self.step_size = step_size
        self.x_bins = int((x_max - x_min) / step_size)
        self.y_bins = int((y_max - y_min) / step_size)
        self.visited_cells = set()
        self.current_cell = None

    def get_cell(self, x, y):
        # Convert continuous coordinates to grid indices
        x_idx = int((x - self.x_min) / self.step_size)
        y_idx = int((y - self.y_min) / self.step_size)
        return (x_idx, y_idx)

    def reset(self):
        self.visited_cells.clear()
        self.current_cell = None

    def get_reward(self, x, y):
        new_cell = self.get_cell(x, y)
        reward = 0.0

        if self.current_cell is None:
            # First step of episode
            self.visited_cells.add(new_cell)
            self.current_cell = new_cell
            return 0.0

        if new_cell == self.current_cell:
            # PENALTY: Stayed in same cell (stagnation)
            reward = -2 
        elif new_cell in self.visited_cells:
            # NEUTRAL: Moved to a cell we already visited (backtracking)
            reward = 0.0
            self.current_cell = new_cell
        else:
            # REWARD: Discovered a NEW cell!
            reward = 40.0
            self.visited_cells.add(new_cell)
            self.current_cell = new_cell

        return reward


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
        self.fc1 = nn.Linear(inputs, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.head = nn.Linear(128, outputs)

        self.apply(self._init_weights)
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Inizializzazione Xavier/Glorot (ottima per Tanh/ReLU)
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)
                
    def forward(self, x):
        if not x.is_cuda and device.type == 'cuda':
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
            return policy_net(state.unsqueeze(0)).max(1)[1].view(1, 1), eps_threshold
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
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch.next_state
                                                if s is not None]) if sum(non_final_mask) > 0 else None
    state_batch = torch.stack(batch.state)
    action_batch = torch.stack(batch.action)
    reward_batch = torch.stack(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, torch.squeeze(action_batch, 2))

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(batch_size, device=device)
    
    if non_final_next_states is not None:
        best_actions = policy_net(non_final_next_states).max(1)[1].unsqueeze(1)

        next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, best_actions).squeeze(1).detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    TAU = 0.005 

    # Aggiorna i parametri della target network lentamente verso la policy network
    for target_param, local_param in zip(target_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)

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
    pkg_path = rospack.get_path('my_turtlebot3_openai_example')
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
    
    # Curriculum Learning Parameters
    curriculum_enabled = rospy.get_param("/turtlebot3/curriculum_enabled", False)
    stage_0_threshold = rospy.get_param("/turtlebot3/stage_0_threshold", 150.0)
    stage_1_threshold = rospy.get_param("/turtlebot3/stage_1_threshold", 250.0)
    curriculum_eval_window = rospy.get_param("/turtlebot3/curriculum_evaluation_window", 50)
    recent_episode_rewards = deque(maxlen=curriculum_eval_window)
    current_curriculum_stage = 0
    
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

    # Initialises the algorithm that we are going to use for learning
    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward'))

    # Get number of actions from gym action space
    n_actions = env.action_space.n
    initial_obs = env.reset()
    n_observations = len(initial_obs)  # Dynamically get observation size

    rospy.loginfo("Initializing DQN with Input: {} observations, Output: {} actions".format(n_observations, n_actions))

    # initialize networks with input and output sizes
    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

  
    # ========== CHECKPOINT LOADING ==========
    if resume_training and os.path.isfile(model_path):
        rospy.loginfo("CARICAMENTO MODELLO ESISTENTE: " + model_path)
        checkpoint = torch.load(model_path)
        policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        target_net.load_state_dict(checkpoint['target_net_state_dict'])
        # Opzionale: caricare anche l'optimizer se vuoi continuità perfetta
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        rospy.loginfo("Modello caricato con successo! Si riparte da dove eravamo.")
        
    optimizer = optim.RMSprop(policy_net.parameters(), lr=lr)
    memory = ReplayMemory(10000)
    episode_durations = []
    steps_done = 0
    start_episode = 0

        
    start_time = time.time()
    last_s = 0
    last_m = 0 
    last_h = 0
    highest_reward = 0
    best_model_reward = -float('inf')
    
    # Training metrics tracking
    max_episode_duration = 0
    max_distance_traveled = 0.0
    total_crashes = 0
    episode_rewards_history = []
    episode_durations_history = []
    episode_distances_history = []
    episode_epsilon_history = []
    
    # Reward Breakdown History
    reward_breakdown_history = {
        'base': [],
        'velocity': [],
        'distance_clearance': [],
        'step_milestone': [],
        'survival': [],
        'distance_traveled_milestone': [],
        'exploration': []
    }
    
  
    # Initialize Grid Explorer (0.5m grid cells)
    grid_explorer = GridExplorer(step_size=0.5)

    # Starts the main training loop: the one about the episodes to do
    for i_episode in range(start_episode, n_episodes):
        rospy.loginfo("############### START EPISODE=>" + str(i_episode) + " [STAGE " + str(current_curriculum_stage) + "]")

        cumulated_reward = 0
        episode_distance = 0.0
        last_distance_check = 0.0
        previous_odom = None
        done = False
        
        # Reset grid for new episode
        grid_explorer.reset()

        # Initialize the environment and get first state of the robot
        observation = env.reset()
        state = torch.tensor(observation, device=device, dtype=torch.float)
        
        # Breakdown accumulator
        episode_breakdown = {k: 0.0 for k in reward_breakdown_history.keys()}

        for t in count():
            rospy.logwarn("############### Start Step=>" + str(t))
            # Select and perform an action
            action, epsilon = select_action(state, epsilon_start, epsilon_end, epsilon_decay)
            rospy.logdebug("Next action is:%d", action)

            observation, reward, done, info = env.step(action.item())
            rospy.logdebug(f"=== observation: {observation}===")
            # Accumulate breakdown
            if isinstance(info, dict):
                for key, val in info.items():
                    if key in episode_breakdown:
                        episode_breakdown[key] += float(val)

            rospy.logdebug(str(observation) + " " + str(reward))
            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward
            
            # Track distance traveled
            try:
                current_odom = env.unwrapped.get_odom()
                if previous_odom is not None:
                    dx = current_odom.pose.pose.position.x - previous_odom.pose.pose.position.x
                    dy = current_odom.pose.pose.position.y - previous_odom.pose.pose.position.y
                    episode_distance += numpy.sqrt(dx*dx + dy*dy)
                previous_odom = current_odom

                # GRID EXPLORATION REWARD
                # Get current position
                robot_x = current_odom.pose.pose.position.x
                robot_y = current_odom.pose.pose.position.y
                
                # Calculate bonus based on grid logic
                exploration_bonus = grid_explorer.get_reward(robot_x, robot_y)
                
                # Add to total reward
                reward += exploration_bonus
                episode_breakdown['exploration'] += exploration_bonus
                
                if exploration_bonus > 5.0:
                    rospy.loginfo("Exploration Bonus! New cell visited.")

            except:
                pass

            # Check if likely stuck (e.g. not moving effectively)
            if t > 0 and t % 15 == 0:
                # Check if distance traveled in last 25 steps is significant (> 0.1m)
                if (episode_distance - last_distance_check) < 0.1:
                    rospy.loginfo("Robot appears STUCK (moved < 0.1m in 25 steps). Ending episode.")
                    reward = -350
                    cumulated_reward += reward
                    done = True
                last_distance_check = episode_distance

            # Manual Max Steps check REMOVED - handled by Environment now
                
            reward = torch.tensor([reward], device=device)

            #next_state = ''.join(map(str, observation))
            next_state = torch.tensor(observation, device=device, dtype=torch.float)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Perform one step of the optimization (on the policy network)
            rospy.logdebug("# state we were=>" + str(state))
            rospy.logdebug("# action that we took=>" + str(action))
            rospy.logdebug("# reward that action gave=>" + str(reward))
            rospy.logdebug("# episode cumulated_reward=>" + str(cumulated_reward))
            rospy.logdebug("# State in which we will start next step=>" + str(next_state))
            optimize_model(batch_size, gamma)
            
            if done:
                episode_durations.append(t + 1)
                rospy.logdebug("DONE")
                last_time_steps = numpy.append(last_time_steps, [int(t + 1)])
                
                # Track metrics
                episode_rewards_history.append(cumulated_reward)
                episode_durations_history.append(t + 1)
                episode_distances_history.append(episode_distance)
                
                # Update breakdown history
                for key in reward_breakdown_history:
                    reward_breakdown_history[key].append(episode_breakdown.get(key, 0.0))

                # Track current epsilon
                current_eps = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1. * steps_done / epsilon_decay)
                episode_epsilon_history.append(current_eps)
                
                # Update max records
                if t + 1 > max_episode_duration:
                    max_episode_duration = t + 1
                    rospy.loginfo("New record! Max episode duration: " + str(max_episode_duration) + " steps")
                
                if episode_distance > max_distance_traveled:
                    max_distance_traveled = episode_distance
                    rospy.loginfo("New record! Max distance traveled: " + str(round(max_distance_traveled, 2)) + " meters")
                
                # Count crashes (done with negative reward)
                if cumulated_reward < 0:
                    total_crashes += 1
                
                # Curriculum Learning: Track rewards and advance stages
                if curriculum_enabled:
                    recent_episode_rewards.append(cumulated_reward)
                    
                    # Check if we should advance curriculum stage
                    if len(recent_episode_rewards) == curriculum_eval_window:
                        avg_reward = sum(recent_episode_rewards) / len(recent_episode_rewards)
                        
                        # Advance from Stage 0 to Stage 1 (collision avoidance -> fast movement)
                        if current_curriculum_stage == 0 and avg_reward >= stage_0_threshold:
                            current_curriculum_stage = 1
                            env.unwrapped.update_curriculum_stage(1, new_velocity_weight=1.0)
                            rospy.logerr("="*50)
                            rospy.logerr("CURRICULUM ADVANCED TO STAGE 1!")
                            rospy.logerr("Average reward over last " + str(curriculum_eval_window) + 
                                        " episodes: " + str(avg_reward))
                            rospy.logerr("Focus: Fast continuous movement")
                            rospy.logerr("="*50)
                        
                        # Advance from Stage 1 to Stage 2 (fast movement -> maximize distance)
                        elif current_curriculum_stage == 1 and avg_reward >= stage_1_threshold:
                            current_curriculum_stage = 2
                            env.unwrapped.update_curriculum_stage(2, new_velocity_weight=1.0, new_distance_weight=1.0)
                            rospy.logwerr("="*50)
                            rospy.logwerr("CURRICULUM ADVANCED TO STAGE 2!")
                            rospy.logwerr("Average reward over last " + str(curriculum_eval_window) + 
                                        " episodes: " + str(avg_reward))
                            rospy.logwerr("Focus: Maximize distance from obstacles")
                            rospy.logwerr("="*50)
                
                break
            else:
                rospy.logdebug("NOT DONE")
                state = next_state

            #rospy.logwarn("############### END Step=>" + str(t))
            # Update the target network, copying all weights and biases in DQN
        if i_episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        
        rospy.logerr(("EP: " + str(i_episode + 1) + " - gamma: " + str(
            round(gamma, 2)) + " - epsilon: " + str(round(epsilon, 2)) + "] - Reward: " + str(
            cumulated_reward) + " - Distance: " + str(round(episode_distance, 2)) + "m - Time: %d:%02d:%02d" % (h-last_h, m-last_m, s-last_s)))
        
        last_m = m
        last_s = s
        last_h = h
        
        # Save best model
        if cumulated_reward > best_model_reward:
            best_model_reward = cumulated_reward
            torch.save({
                'episode': i_episode,
                'policy_net_state_dict': policy_net.state_dict(),
                'target_net_state_dict': target_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'reward': cumulated_reward,
                'distance': episode_distance,
                'duration': t + 1,
                'curriculum_stage': current_curriculum_stage if curriculum_enabled else 0,
                'steps_done': steps_done,
            }, model_path + '/best_model.pth')
            rospy.loginfo("Saved new best model with reward: " + str(cumulated_reward))
        
        # Save periodic checkpoints every 500 episodes
        if (i_episode + 1) % 500 == 0:
            # 1. Generate and Save Plot
            plot_filename = outdir + '/training_metrics_ep' + str(i_episode + 1) + '.png'
            plot_training_metrics(episode_rewards_history, episode_durations_history, 
                                  episode_distances_history, episode_epsilon_history, 
                                  reward_breakdown_history, plot_filename)
            rospy.loginfo("Saved metrics plot to " + plot_filename)

            # 2. Save Checkpoint with FULL metrics history
            checkpoint_data = {
                'episode': i_episode,
                'policy_net_state_dict': policy_net.state_dict(),
                'target_net_state_dict': target_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'reward': cumulated_reward,
                'curriculum_stage': current_curriculum_stage if curriculum_enabled else 0,
                'steps_done': steps_done,
                'reward_history': episode_rewards_history,
                'reward_breakdown_history': reward_breakdown_history,
                'duration_history': episode_durations_history,
                'distance_history': episode_distances_history,
                'epsilon_history': episode_epsilon_history,
                'success_rate': (float(sum([1 for r in recent_episode_rewards if r > 400])) / float(len(recent_episode_rewards))) if len(recent_episode_rewards) > 0 else 0.0,
            }
            
            # Save periodic checkpoint
            torch.save(checkpoint_data, model_path + '/checkpoint_ep' + str(i_episode + 1) + '.pth')
            
            # Also save as latest for easy resuming
            torch.save(checkpoint_data, model_path + '/checkpoint_latest.pth')
            
            rospy.loginfo("Saved checkpoint at episode " + str(i_episode + 1))

    rospy.loginfo(("\n|" + str(n_episodes) + "|" + str(gamma) + "|" + str(epsilon_start) + "*" +
                   str(epsilon_decay) + "|" + str(highest_reward) + "| PICTURE |"))

    l = last_time_steps.tolist()
    l.sort()

    # print("Parameters: a="+str)
    rospy.loginfo("Overall score: {:0.2f}".format(last_time_steps.mean()))
    rospy.loginfo("Best 100 score: {:0.2f}".format(sum(episode_rewards_history[-100:]) / len(episode_rewards_history[-100:])))

    # Save final model
    final_training_time = time.time() - start_time
    torch.save({
        'episode': n_episodes,
        'policy_net_state_dict': policy_net.state_dict(),
        'target_net_state_dict': target_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_reward': cumulated_reward,
        'highest_reward': highest_reward,
        'best_model_reward': best_model_reward,
        'curriculum_stage': current_curriculum_stage if curriculum_enabled else 0,
        'max_episode_duration': max_episode_duration,
        'max_distance_traveled': max_distance_traveled,
        'total_crashes': total_crashes,
        'training_time': final_training_time,
        'gamma': gamma,
        'epsilon_start': epsilon_start,
        'epsilon_end': epsilon_end,
        'epsilon_decay': epsilon_decay,
    }, model_path + '/final_model.pth')
    rospy.loginfo("Final model saved to: " + model_path + '/final_model.pth')
    rospy.loginfo("Best model saved to: " + model_path + '/best_model.pth')
    
    # Generate comprehensive training report
    report_path = reports_dir + '/training_report.txt'
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TURTLEBOT3 DQN TRAINING REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("TRAINING CONFIGURATION\n")
        f.write("-"*80 + "\n")
        f.write("Number of Episodes: {}\n".format(n_episodes))
        f.write("Gamma: {}\n".format(gamma))
        f.write("Epsilon Start: {}\n".format(epsilon_start))
        f.write("Epsilon End: {}\n".format(epsilon_end))
        f.write("Epsilon Decay: {}\n".format(epsilon_decay))
        f.write("Batch Size: {}\n".format(batch_size))
        f.write("Target Update Frequency: {}\n".format(target_update))
        f.write("Curriculum Learning Enabled: {}\n".format(curriculum_enabled))
        if curriculum_enabled:
            f.write("Final Curriculum Stage: {}\n".format(current_curriculum_stage))
            f.write("Stage 0 Threshold: {}\n".format(stage_0_threshold))
            f.write("Stage 1 Threshold: {}\n".format(stage_1_threshold))
        f.write("\n")
        
        f.write("TRAINING RESULTS\n")
        f.write("-"*80 + "\n")
        hours, remainder = divmod(int(final_training_time), 3600)
        minutes, seconds = divmod(remainder, 60)
        f.write("Total Training Time: {:02d}:{:02d}:{:02d}\n".format(hours, minutes, seconds))
        f.write("Highest Reward Achieved: {:.2f}\n".format(highest_reward))
        f.write("Best Model Reward: {:.2f}\n".format(best_model_reward))
        f.write("Average Episode Duration: {:.2f} steps\n".format(numpy.mean(episode_durations_history)))
        f.write("Overall Score (mean last time steps): {:.2f}\n".format(last_time_steps.mean()))
        f.write("Best 100 Episodes Score: {:.2f}\n".format(sum(episode_rewards_history[-100:]) / len(episode_rewards_history[-100:])))
        f.write("\n")
        
        f.write("PERFORMANCE RECORDS\n")
        f.write("-"*80 + "\n")
        f.write("Maximum Episode Duration (without crash): {} steps\n".format(max_episode_duration))
        f.write("Maximum Distance Traveled: {:.2f} meters\n".format(max_distance_traveled))
        f.write("Total Crashes: {} out of {} episodes ({:.2f}%)\n".format(
            total_crashes, n_episodes, (total_crashes / n_episodes) * 100))
        f.write("Success Rate: {:.2f}%\n".format(((n_episodes - total_crashes) / n_episodes) * 100))
        f.write("\n")
        
        f.write("EPISODE STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write("Reward Statistics:\n")
        f.write("  Mean: {:.2f}\n".format(numpy.mean(episode_rewards_history)))
        f.write("  Std Dev: {:.2f}\n".format(numpy.std(episode_rewards_history)))
        f.write("  Min: {:.2f}\n".format(numpy.min(episode_rewards_history)))
        f.write("  Max: {:.2f}\n".format(numpy.max(episode_rewards_history)))
        f.write("\n")
        f.write("Duration Statistics:\n")
        f.write("  Mean: {:.2f} steps\n".format(numpy.mean(episode_durations_history)))
        f.write("  Std Dev: {:.2f} steps\n".format(numpy.std(episode_durations_history)))
        f.write("  Min: {} steps\n".format(numpy.min(episode_durations_history)))
        f.write("  Max: {} steps\n".format(numpy.max(episode_durations_history)))
        f.write("\n")
        f.write("Distance Statistics:\n")
        f.write("  Mean: {:.2f} meters\n".format(numpy.mean(episode_distances_history)))
        f.write("  Std Dev: {:.2f} meters\n".format(numpy.std(episode_distances_history)))
        f.write("  Min: {:.2f} meters\n".format(numpy.min(episode_distances_history)))
        f.write("  Max: {:.2f} meters\n".format(numpy.max(episode_distances_history)))
        f.write("\n")
        
        f.write("MODEL FILES\n")
        f.write("-"*80 + "\n")
        f.write("Final Model: {}\n".format(model_path + '/final_model.pth'))
        f.write("Best Model: {}\n".format(model_path + '/best_model.pth'))
        f.write("Checkpoints: {} files (every 500 episodes)\n".format(n_episodes // 500))
        f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("Report generated at: {}\n".format(time.strftime("%Y-%m-%d %H:%M:%S")))
        f.write("="*80 + "\n")
    
    rospy.loginfo("Training report saved to: " + report_path)
    rospy.loginfo("="*50)
    rospy.loginfo("TRAINING COMPLETE!")
    rospy.loginfo("Max Episode Duration: {} steps".format(max_episode_duration))
    rospy.loginfo("Max Distance Traveled: {:.2f} meters".format(max_distance_traveled))
    rospy.loginfo("Total Crashes: {} ({:.2f}%)".format(total_crashes, (total_crashes / n_episodes) * 100))
    rospy.loginfo("="*50)

    env.close()

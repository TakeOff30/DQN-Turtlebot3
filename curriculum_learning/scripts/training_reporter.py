#!/usr/bin/env python3

import numpy
import time
from utils import plot_training_metrics

class TrainingReporter:
    def __init__(self, reports_dir, model_path):
        self.model_path = model_path
        self.report_path = f"{reports_dir}/training_report.txt"
    
    def generate_plot(self, episode, outdir, episode_rewards_history, 
                     episode_durations_history, episode_distances_history,
                     episode_epsilon_history, reward_breakdown_history):
        """Generate and save training metrics plot."""
        plot_filename = f"{outdir}/training_metrics_ep{episode}.png"
        plot_training_metrics(episode_rewards_history, episode_durations_history,
                            episode_distances_history, episode_epsilon_history,
                            reward_breakdown_history, plot_filename)
        return plot_filename
    
    def write_header(self):
        with open(self.report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("TURTLEBOT3 DQN TRAINING REPORT\n")
            f.write("="*80 + "\n\n")
    
    def write_configuration(self, n_episodes, gamma, epsilon_start, epsilon_end,
                            epsilon_decay, batch_size, target_update):
        with open(self.report_path, 'w') as f:
            f.write("TRAINING CONFIGURATION\n")
            f.write("-"*80 + "\n")
            f.write(f"Number of Episodes: {n_episodes}\n")
            f.write(f"Gamma: {gamma}\n")
            f.write(f"Epsilon Start: {epsilon_start}\n")
            f.write(f"Epsilon End: {epsilon_end}\n")
            f.write(f"Epsilon Decay: {epsilon_decay}\n")
            f.write(f"Batch Size: {batch_size}\n")
            f.write(f"Target Update Frequency: {target_update}\n")
            f.write("\n")
    
    def write_training_results(self, training_time, highest_reward, last_time_steps):
        with open(self.report_path, 'w') as f:
            f.write("TRAINING RESULTS\n")
            f.write("-"*80 + "\n")
            hours, remainder = divmod(int(training_time), 3600)
            minutes, seconds = divmod(remainder, 60)
            f.write(f"Total Training Time: {hours:02d}:{minutes:02d}:{seconds:02d}\n")
            f.write(f"Highest Reward Achieved: {highest_reward:.2f}\n")
            f.write(f"Overall Score (mean last time steps): {last_time_steps.mean():.2f}\n")
            f.write("\n")
    
    def write_episode_statistics(self, episode_rewards_history,
                                  episode_durations_history, episode_distances_history):
        with open(self.report_path, 'w') as f:
            f.write("EPISODE STATISTICS\n")
            f.write("-"*80 + "\n")
            
            # Reward stats
            f.write("Reward Statistics:\n")
            f.write(f"  Mean: {numpy.mean(episode_rewards_history):.2f}\n")
            f.write(f"  Std Dev: {numpy.std(episode_rewards_history):.2f}\n")
            f.write(f"  Min: {numpy.min(episode_rewards_history):.2f}\n")
            f.write(f"  Max: {numpy.max(episode_rewards_history):.2f}\n")
            f.write("\n")
            
            # Duration stats
            f.write("Duration Statistics:\n")
            f.write(f"  Mean: {numpy.mean(episode_durations_history):.2f} steps\n")
            f.write(f"  Std Dev: {numpy.std(episode_durations_history):.2f} steps\n")
            f.write(f"  Min: {numpy.min(episode_durations_history)} steps\n")
            f.write(f"  Max: {numpy.max(episode_durations_history)} steps\n")
            f.write("\n")
            
            # Distance stats
            f.write("Distance Statistics:\n")
            f.write(f"  Mean: {numpy.mean(episode_distances_history):.2f} meters\n")
            f.write(f"  Std Dev: {numpy.std(episode_distances_history):.2f} meters\n")
            f.write(f"  Min: {numpy.min(episode_distances_history):.2f} meters\n")
            f.write(f"  Max: {numpy.max(episode_distances_history):.2f} meters\n")
            f.write("\n")

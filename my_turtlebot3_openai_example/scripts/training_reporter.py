#!/usr/bin/env python3

import numpy
import time
from utils import plot_training_metrics

class TrainingReporter:
    def __init__(self, reports_dir, model_path):
        self.reports_dir = reports_dir
        self.model_path = model_path
    
    def generate_plot(self, episode, outdir, episode_rewards_history, 
                     episode_durations_history, episode_distances_history,
                     episode_epsilon_history, reward_breakdown_history):
        """Generate and save training metrics plot."""
        plot_filename = f"{outdir}/training_metrics_ep{episode}.png"
        plot_training_metrics(episode_rewards_history, episode_durations_history,
                            episode_distances_history, episode_epsilon_history,
                            reward_breakdown_history, plot_filename)
        return plot_filename
    
    def generate_text_report(self, n_episodes, gamma, epsilon_start, epsilon_end,
                           epsilon_decay, batch_size, target_update, training_time,
                           highest_reward, episode_rewards_history, episode_durations_history,
                           episode_distances_history, max_episode_duration, 
                           max_distance_traveled, last_time_steps):
        """Generate comprehensive text training report."""
        report_path = f"{self.reports_dir}/training_report.txt"
        
        with open(report_path, 'w') as f:
            self._write_header(f)
            self._write_configuration(f, n_episodes, gamma, epsilon_start, epsilon_end,
                                     epsilon_decay, batch_size, target_update)
            self._write_training_results(f, training_time, highest_reward,
                                        episode_rewards_history, episode_durations_history,
                                        last_time_steps)
            self._write_performance_records(f, max_episode_duration, max_distance_traveled)
            self._write_episode_statistics(f, episode_rewards_history, 
                                          episode_durations_history, episode_distances_history)
            self._write_model_files(f, n_episodes)
            self._write_footer(f)
        
        return report_path
    
    def _write_header(self, f):
        f.write("="*80 + "\n")
        f.write("TURTLEBOT3 DQN TRAINING REPORT\n")
        f.write("="*80 + "\n\n")
    
    def _write_configuration(self, f, n_episodes, gamma, epsilon_start, epsilon_end,
                            epsilon_decay, batch_size, target_update):
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
    
    def _write_training_results(self, f, training_time, highest_reward,
                               episode_rewards_history, episode_durations_history,
                               last_time_steps):
        f.write("TRAINING RESULTS\n")
        f.write("-"*80 + "\n")
        hours, remainder = divmod(int(training_time), 3600)
        minutes, seconds = divmod(remainder, 60)
        f.write(f"Total Training Time: {hours:02d}:{minutes:02d}:{seconds:02d}\n")
        f.write(f"Highest Reward Achieved: {highest_reward:.2f}\n")
        f.write(f"Average Episode Duration: {numpy.mean(episode_durations_history):.2f} steps\n")
        f.write(f"Overall Score (mean last time steps): {last_time_steps.mean():.2f}\n")
        best_100 = sum(episode_rewards_history[-100:]) / len(episode_rewards_history[-100:])
        f.write(f"Best 100 Episodes Score: {best_100:.2f}\n")
        f.write("\n")
    
    def _write_performance_records(self, f, max_episode_duration, max_distance_traveled):
        f.write("PERFORMANCE RECORDS\n")
        f.write("-"*80 + "\n")
        f.write(f"Maximum Episode Duration: {max_episode_duration} steps\n")
        f.write(f"Maximum Distance Traveled: {max_distance_traveled:.2f} meters\n")
        f.write("\n")
    
    def _write_episode_statistics(self, f, episode_rewards_history,
                                  episode_durations_history, episode_distances_history):
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
    
    def _write_model_files(self, f, n_episodes):
        f.write("MODEL FILES\n")
        f.write("-"*80 + "\n")
        f.write(f"Final Model: {self.model_path}/final_model.pth\n")
        f.write(f"Checkpoints: {n_episodes // 500} files (every 500 episodes)\n")
        f.write("\n")
    
    def _write_footer(self, f):
        f.write("="*80 + "\n")
        f.write(f"Report generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n")

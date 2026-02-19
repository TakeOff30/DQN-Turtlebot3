#!/usr/bin/env python3

import os
from datetime import datetime
import numpy


class InferenceReporter:
    def __init__(self, reports_root):
        os.makedirs(reports_root, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_path = os.path.join(reports_root, f"inference_report_{timestamp}.txt")

    def write_header(self):
        with open(self.report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("TURTLEBOT3 DQN INFERENCE REPORT\n")
            f.write("=" * 80 + "\n\n")

    def write_configuration(self, model_file, n_eval_episodes, model_type, device):
        with open(self.report_path, 'a') as f:
            f.write("INFERENCE CONFIGURATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Model File: {model_file}\n")
            f.write(f"Model Type: {model_type}\n")
            f.write(f"Evaluation Episodes: {n_eval_episodes}\n")
            f.write(f"Device: {device}\n")
            f.write("\n")
            f.write("EPISODE GOALS\n")
            f.write("-" * 80 + "\n")

    def append_episode_result(self, episode_idx, total_episodes, goals):
        with open(self.report_path, 'a') as f:
            f.write(
                f"Episode {episode_idx + 1}/{total_episodes} | "
                f"Goals: {goals}\n"
            )

    def write_summary(self, episode_goals, n_eval_episodes):
        goals_array = numpy.array(episode_goals)
        successful_episodes = int(numpy.sum(goals_array == 3))
        success_rate = (successful_episodes / n_eval_episodes) * 100.0 if n_eval_episodes > 0 else 0.0
        avg_goals = float(numpy.mean(goals_array)) if len(goals_array) > 0 else 0.0

        with open(self.report_path, 'a') as f:
            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write("INFERENCE SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"Episodes Evaluated: {n_eval_episodes}\n")
            f.write(f"Success Rate: {success_rate:.1f}% ({successful_episodes}/{n_eval_episodes})\n")
            f.write(f"Average Goals Per Episode: {avg_goals:.2f}\n")
            f.write("=" * 80 + "\n")

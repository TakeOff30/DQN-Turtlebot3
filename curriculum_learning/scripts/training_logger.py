#!/usr/bin/env python3

import rospy
import time

class TrainingLogger:
    def __init__(self):
        self.start_time = time.time()
        self.last_h = 0
        self.last_m = 0
        self.last_s = 0
    
    def log_episode_start(self, episode, stage=None):
        stage_info = f" [STAGE {stage}]" if stage is not None else ""
        rospy.loginfo(f"############### START EPISODE=>{episode}{stage_info}")
    
    def log_step_start(self, step):
        rospy.logwarn(f"############### Start Step=>{step}")
    
    def log_episode_end(self, episode, gamma, epsilon, reward, distance):
        m, s = divmod(int(time.time() - self.start_time), 60)
        h, m = divmod(m, 60)
        
        rospy.logerr(f"EP: {episode + 1} - gamma: {round(gamma, 2)} ] - "
                    f"Reward: {reward} - Distance: {round(distance, 2)}m - "
                    f"Time: {h-self.last_h:d}:{m-self.last_m:02d}:{s-self.last_s:02d}")
        
        self.last_m = m
        self.last_s = s
        self.last_h = h
    
    def log_new_record(self, metric_name, value, unit=""):
        rospy.loginfo(f"New record! {metric_name}: {value}{unit}")
    
    def log_checkpoint_saved(self, episode, plot_path):
        rospy.loginfo(f"Saved metrics plot to {plot_path}")
        rospy.loginfo(f"Saved checkpoint at episode {episode}")
    
    def log_robot_stuck(self):
        rospy.loginfo("Robot appears STUCK (moved < 0.1m in 15 steps). Ending episode.")
    
    def log_final_scores(self, n_episodes, gamma, epsilon_start, epsilon_decay, highest_reward, 
                        last_time_steps, episode_rewards_history):
        rospy.loginfo(f"\n|{n_episodes}|{gamma}|{epsilon_start}*{epsilon_decay}|{highest_reward}| PICTURE |")
        rospy.loginfo(f"Overall score: {last_time_steps.mean():.2f}")
        best_100 = sum(episode_rewards_history[-100:]) / len(episode_rewards_history[-100:])
        rospy.loginfo(f"Best 100 score: {best_100:.2f}")
    
    def log_training_complete(self, max_duration, max_distance, final_model_path, report_path):
        rospy.loginfo(f"Final model saved to: {final_model_path}")
        rospy.loginfo(f"Training report saved to: {report_path}")
        rospy.loginfo("="*50)
        rospy.loginfo("TRAINING COMPLETE!")
        rospy.loginfo(f"Max Episode Duration: {max_duration} steps")
        rospy.loginfo(f"Max Distance Traveled: {max_distance:.2f} meters")
        rospy.loginfo("="*50)

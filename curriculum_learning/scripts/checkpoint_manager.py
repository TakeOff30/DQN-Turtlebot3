#!/usr/bin/env python3

import torch
import rospy
import os

class CheckpointManager:
    def __init__(self, model_path):
        self.model_path = model_path
        if not os.path.exists(model_path):
            os.makedirs(model_path)
    
    def save_checkpoint(self, episode, policy_net, target_net, optimizer,
                       reward, steps_done, episode_rewards_history,
                       reward_breakdown_history, episode_durations_history,
                       episode_distances_history, episode_epsilon_history):
        """Save periodic training checkpoint."""
        checkpoint_data = {
            'episode': episode,
            'policy_net_state_dict': policy_net.state_dict(),
            'target_net_state_dict': target_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'reward': reward,
            'steps_done': steps_done,
            'reward_history': episode_rewards_history,
            'reward_breakdown_history': reward_breakdown_history,
            'duration_history': episode_durations_history,
            'distance_history': episode_distances_history,
            'epsilon_history': episode_epsilon_history,
        }
        
        # Save numbered checkpoint
        checkpoint_path = f"{self.model_path}/checkpoint_ep{episode}.pth"
        torch.save(checkpoint_data, checkpoint_path)
        
        # Save as latest
        latest_path = f"{self.model_path}/checkpoint_latest.pth"
        torch.save(checkpoint_data, latest_path)
        
        return checkpoint_path
    
    def save_final_model(self, n_episodes, policy_net, target_net, optimizer,
                        cumulated_reward, highest_reward, max_episode_duration,
                        max_distance_traveled, training_time, gamma,
                        epsilon_start, epsilon_end, epsilon_decay):
        """Save final model at end of training."""
        final_model_data = {
            'episode': n_episodes,
            'policy_net_state_dict': policy_net.state_dict(),
            'target_net_state_dict': target_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'final_reward': cumulated_reward,
            'highest_reward': highest_reward,
            'max_episode_duration': max_episode_duration,
            'max_distance_traveled': max_distance_traveled,
            'training_time': training_time,
            'gamma': gamma,
            'epsilon_start': epsilon_start,
            'epsilon_end': epsilon_end,
            'epsilon_decay': epsilon_decay,
        }
        
        final_path = f"{self.model_path}/final_model.pth"
        torch.save(final_model_data, final_path)
        return final_path
    
    def load_checkpoint(self, checkpoint_path, policy_net, target_net, optimizer=None):
        """Load checkpoint from file."""
        if os.path.isfile(checkpoint_path):
            rospy.logwarn(f"Loading checkpoint model: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            target_net.load_state_dict(checkpoint['target_net_state_dict'])
            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            rospy.loginfo("Checkpoint loaded successfully!")
            return checkpoint
        return None

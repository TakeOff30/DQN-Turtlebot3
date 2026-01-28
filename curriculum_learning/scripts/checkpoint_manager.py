#!/usr/bin/env python3

import torch
import rospy
import os
from datetime import datetime

class CheckpointManager:
    def __init__(self, model_path):
        self.model_path = model_path
        if not os.path.exists(model_path):
            os.makedirs(model_path)
    
    def save_final_model(self, policy_net, max_avg_reward, filename="final_model", timestamp=True):
        """Save final model at end of training with timestamp.        """
        final_model_data = {
            'policy_net_state_dict': policy_net.state_dict(),
            'max_avg_reward': max_avg_reward
        }
        timestamp_s = datetime.now().strftime("%Y%m%d_%H%M%S")
        if timestamp:
            final_filename = f"{filename}_{timestamp_s}.pth"
        else:
            final_filename = f"{filename}.pth"
        final_path = os.path.join(self.model_path, final_filename)
        torch.save(final_model_data, final_path)
        return final_path
    
    def load_checkpoint(self, checkpoint_path, policy_net, target_net):
        """Load checkpoint from file."""
        rospy.logwarn(f"Loading checkpoint model: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        target_net.load_state_dict(policy_net)
        target_net.eval()
        max_avg_reward = checkpoint['max_avg_reward'] if 'max_avg_reward' in checkpoint.keys() else 0
        rospy.loginfo("Checkpoint loaded successfully!")
        return max_avg_reward

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
    
    def save_final_model(self, policy_net, target_net, optimizer):
        """Save final model at end of training with timestamp.        """
        final_model_data = {
            'policy_net_state_dict': policy_net.state_dict(),
            'target_net_state_dict': target_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_filename = f"final_model_{timestamp}.pth"
        final_path = os.path.join(self.model_path, final_filename)
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

import torch
import torch.nn as nn

class DQN(nn.Module):
    """
    Standard DQN architecture.
    """
    def __init__(self, inputs, outputs):
        super(DQN, self).__init__()
        
        # Shared feature extraction
        self.feature = nn.Sequential(
            nn.Linear(inputs, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, outputs)
        )
        
        self.apply(self._init_weights)
    
    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            nn.init.constant_(module.bias, 0)

    def forward(self, x):
        device = next(self.parameters()).device
        x = x.to(device)
        
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        features = self.feature(x)
        return features

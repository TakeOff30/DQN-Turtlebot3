import torch.nn as nn

class DuelingDQN(nn.Module):
    """Dueling DQN: separates Value and Advantage streams for better learning.
    Reference: Wang et al. 2016, 'Dueling Network Architectures for Deep RL'
    """

    def __init__(self, inputs, outputs):
        super(DuelingDQN, self).__init__()
        
        # Shared feature extraction
        self.feature = nn.Sequential(
            nn.Linear(inputs, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        
        # Value stream: V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Advantage stream: A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, outputs)
        )
        
        # He (Kaiming) initialization for ReLU networks
        self.apply(self._init_weights)
    
    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = x.to(device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values
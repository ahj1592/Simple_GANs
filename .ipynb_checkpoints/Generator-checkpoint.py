import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=10, im_dim=28*28, hidden_dim=128):
        super(Generator, self).__init__()
        
        self.gen = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.BatchNorm1d(hidden_dim*2),
            nn.ReLU(inplace=True),
            
            nn.Linear(hidden_dim*2, hidden_dim*4),
            nn.BatchNorm1d(hidden_dim*4),
            nn.ReLU(inplace=True),
            
            nn.Linear(hidden_dim*4, hidden_dim*8),
            nn.BatchNorm1d(hidden_dim*8),
            nn.ReLU(inplace=True),
            
            nn.Linear(hidden_dim*8, im_dim),
            nn.Sigmoid()
        )
    
    def forward(self, noise):
        return self.gen(noise)

import torch
import torch.nn as nn
import math
import numpy as np

class STGSelector(nn.Module):
    def __init__(self, input_dim, args):
        super(STGSelector, self).__init__()
        self.args = args
        self.mu = torch.nn.Parameter(0.01*torch.randn(input_dim, ).to(self.args.device), requires_grad=True)
        self.eps = torch.randn(self.mu.size()).to(self.args.device)
        self.sigma = 0.5
    
    def hard_sigmoid(self, x):
        return torch.clamp(x+0.5, 0.0, 1.0)

    def forward(self, prev_x, permuted_x):
        z = self.mu + self.sigma*self.eps.normal_()
        stochastic_gate = self.hard_sigmoid(z)
        if permuted_x == None:
            new_x = prev_x * stochastic_gate
        else:
            new_x = prev_x * stochastic_gate + permuted_x * (1-stochastic_gate)
        return new_x
    
    def regularizer(self): 
        x = ( self.mu + 0.5 ) / self.sigma
        return 0.5 * (1 + torch.erf(x / math.sqrt(2))) 

    def get_gates(self):
        mu = self.mu.detach().cpu().numpy()
        return mu, np.minimum(1.0, np.maximum(0.0, mu + 0.5)) 

import torch
import torch.nn as nn
import math
from .poisson import PoiBin
import numpy as np

class HCSelector(nn.Module):
    def __init__(self, input_dim, group_sizes, args):
        super(HCSelector, self).__init__()
        '''
        log_logit = normal(0,0.01)
        eps = uniform(0,1)
        '''
        self.group_sizes = group_sizes
        self.args = args
        self.log_logit = nn.Parameter(0.01*torch.randn(input_dim, device=self.args.device), requires_grad=True)
        self.eps = torch.randn(self.log_logit.size()).to(self.args.device)
        
        self.temperature = 0.5
        self.limit_a = torch.Tensor([-.1]).to(self.args.device)
        self.limit_b = torch.Tensor([1.1]).to(self.args.device)
        self.epsilon = 1e-6

    def forward(self, prev_x, permuted_x):
        u = self.eps.uniform_(self.epsilon, 1-self.epsilon)                                     # uniform sampling
        s = nn.Sigmoid()((torch.log(u) - torch.log(1-u) + self.log_logit) / self.temperature)   # prob
        z = s * (self.limit_b - self.limit_a) + self.limit_a                                    # prob streching
        stochastic_gate = nn.Hardtanh(min_val=0, max_val=1)(z)
        if permuted_x == None:
            new_x = prev_x * stochastic_gate
        else:
            new_x = prev_x * stochastic_gate + permuted_x * (1-stochastic_gate)

        return new_x
    
    def regularizer_all(self): 
        penalty = nn.Sigmoid()(self.log_logit - self.temperature * (torch.log(-self.limit_a / self.limit_b)))
        return penalty
    
    def regularizer_group(self):
        z_group = []
        mus = 0
        st = 0
        for i in range(len(self.group_sizes)):
            ed = self.group_sizes[i]
            mus = self.log_logit[st:st+ed]

            # -- approximate -- #
            real_prob = nn.Sigmoid()(mus)*(self.limit_b - self.limit_a) + self.limit_a
            prob = nn.Hardtanh(min_val=0, max_val=1)(real_prob)

            '''normalize'''
            if self.args.norm == 0:
                norm_term = 1
            else:
                norm_term = ed * self.args.norm

            mu = prob.sum()
            var = torch.sqrt((prob*(1-prob)).sum())
            gam = var**(-3)*((prob*(1-prob)*(1-2*prob)).sum())
            m_shift = (0+0.5-mu)/(var*math.sqrt(2)) / norm_term

            '''
            poisson_cdf : poisson approximate
            normal_cdf : normal approximate
            r_normal_cdf : refined normal approximate
            '''

            if self.args.distribution == 'poi':
                pb = PoiBin(prob, self.args)
                pb_cdf = pb.cdf_0.reshape(1)
                cdf_zero = pb_cdf
            elif self.args.distribution == 'norm':
                normal_cdf = 1/2 * (1 + torch.erf(m_shift)).reshape(1)
                cdf_zero = normal_cdf
            elif self.args.distribution == 'rnom':
                normal_cdf = 1/2 * (1 + torch.erf(m_shift)).reshape(1)
                r_normal_cdf = normal_cdf + gam * (1-m_shift**2) * normal_cdf / 6
                cdf_zero = r_normal_cdf
            elif self.args.distribution == 'pos':
                cdf_zero = torch.exp(-mu/np.sqrt(norm_term)).reshape(1)
            elif self.args.distribution == 'origin':
                cdf_zero = torch.cumprod(1-prob, dim=0)[-1].reshape(1)

            z_group.append(1-cdf_zero)
            st += self.group_sizes[i]

        z_groups = torch.cat(z_group, dim=0).reshape(-1)
        return z_groups.mean()

    def get_gates(self):
        log_logit = self.log_logit.detach().cpu().numpy()
        real_prob = nn.Sigmoid()(self.log_logit)*(self.limit_b - self.limit_a) + self.limit_a
        return log_logit, nn.Hardtanh(min_val=0, max_val=1)(real_prob).detach().cpu().numpy()

import torch.nn as nn
from .HCselector import HCSelector
from .STGselector import STGSelector

class Model(nn.Module):
    def __init__(self, num_coeffs, group_sizes, args):
        super().__init__()
        self.first = nn.Linear(num_coeffs, int(num_coeffs/2))
        self.NN =nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(int(num_coeffs/2),int(num_coeffs/4)),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(int(num_coeffs/2),1)
        )

        if args.selector == 'HC':
            self.feature_selector = HCSelector(num_coeffs, group_sizes, args)
            print(num_coeffs, group_sizes)
            self.regularizer_all = self.feature_selector.regularizer_all
            self.regularizer_group = self.feature_selector.regularizer_group
        else:
            self.feature_selector = STGSelector(num_coeffs, args)
            self.regularizer = self.feature_selector.regularizer
        self.get_gates = self.feature_selector.get_gates
        
    def forward(self, x, per_x):
        x = self.feature_selector(x, per_x)
        x = self.first(x)
        x = self.NN(x)
        return x
    
# class Model(nn.Module):
#     def __init__(self, num_coeffs, group_sizes, args):
#         super().__init__()
#         self.NN =nn.Sequential(
#             nn.Linear(num_coeffs, 256),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(256,128),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(128,1)
#         )

#         if args.selector == 'HC':
#             self.feature_selector = HCSelector(num_coeffs, group_sizes, args)
#             print(num_coeffs, group_sizes)
#             self.regularizer_all = self.feature_selector.regularizer_all
#             self.regularizer_group = self.feature_selector.regularizer_group
#         else:
#             self.feature_selector = STGSelector(num_coeffs, args)
#             self.regularizer = self.feature_selector.regularizer
#         self.get_gates = self.feature_selector.get_gates
        
#     def forward(self,x):
#         x = self.feature_selector(x)
#         x = self.NN(x)
#         return x
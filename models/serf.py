import math
import torch
import torch.nn as nn
from torch.nn import functional as F


#x * erf(ln(1+e^{x}))
class SERF(nn.Module):
    def __init__(self, thresh = 50):
        super().__init__()

        self.thresh = thresh
        print('SERF using log trick', self.thresh)
    
    def forward(self,x):
        return self.serf_log1pexp(x)

    # naive serf
    def serf(self,x):
        return x * torch.erf(torch.log(1+ torch.exp(x)))


    def serf_log1pexp(self,x):
        return x * torch.erf(torch.log1p(torch.exp(torch.clamp(x, max=self.thresh)))) #thanks @t-mesk on github
        
    # #logsumexp trickk
    # def serf_log(self,x):
    #     c = torch.max(x)
    #     print('torch max', c)
    #     ins = torch.exp(-c) + torch.exp(x-c)
    #     print('torch.min', torch.min(ins))
    #     print('torch.isnan', ins.isnan().any())
    #     print('torch.isinf', ins.isinf().any())
    #     s= c + torch.log(ins)
    #     print('sum.isnan', s.isnan().any())
    #     print('sum.isinf', s.isinf().any())
    #     return x * torch.erf(s)
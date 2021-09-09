import math
import torch
import torch.nn as nn
from torch.nn import functional as F


#x * erf(ln(1+e^{x}))
class SERF(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        return x * torch.erf(torch.log(1+ torch.exp(x)))
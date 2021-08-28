"""
Original Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
"""
Adapted to use from 
https://github.com/HobbitLong/SupContrast
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

model_dict = {
    'resnet18':  512,
    'resnet34':  512,
    'resnet50':  2048,
    'resnet101': 2048,
    'resnet152': 2048,
    'efficientnet-b3': 1536,
    'efficientnet-b5': 2048
}

class SupConEncoder(nn.Module):
    """backbone + projection head"""
    def __init__(self, encoder, name='resnet50', head='mlp', feat_dim=128):
        super(SupConEncoder, self).__init__()
        dim_in = model_dict[name]
        self.encoder = encoder
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat
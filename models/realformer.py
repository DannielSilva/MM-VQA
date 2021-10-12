# implementation from https://github.com/cloneofsimo/RealFormer-pytorch/blob/main/models.py
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from models.serf import SERF

class ResEncoderBlock(nn.Module):
    def __init__(self, emb_s = 32, head_cnt = 8, dp1 = 0.1, dp2 = 0.1):
        super().__init__()
        emb = emb_s * head_cnt
        self.kqv = nn.Linear(emb_s, 3*emb_s, bias = False)
        self.dp = nn.Dropout(dp1)     
        self.proj = nn.Linear(emb, emb,bias = False)
        self.head_cnt = head_cnt
        self.emb_s = emb_s
        self.ln1 = nn.LayerNorm(emb)
        self.ln2 = nn.LayerNorm(emb)
        
        self.ff = nn.Sequential(
            nn.Linear(emb, 4 * emb),
            #nn.GELU(),
            SERF(),
            nn.Linear(4 * emb, emb),
            nn.Dropout(dp2),
        )
        print('Using SERF')

    def resmha(self, x, prev = None):
        B, T, _ = x.shape
        x = x.reshape(B, T, self.head_cnt, self.emb_s)
        k, q, v = torch.split(self.kqv(x), self.emb_s, dim = -1) # B, T, h, emb_s
        if prev is not None : 
            att_score = torch.einsum('bihk,bjhk->bijh', q, k)/self.emb_s**0.5 + prev
        else:
            att_score = torch.einsum('bihk,bjhk->bijh', q, k)/self.emb_s**0.5

        prev = att_score
        att = F.softmax(prev, dim = 2) #B, T, T, h sum on dim 1 = 1
        res = torch.einsum('btih,bihs->bths', att, v).reshape(B, T, -1) #B, T, h * emb_s
        return self.dp(self.proj(res)), prev
    
    def forward(self, x, prev = None, last=False): ## add & norm later.
        rmha, prev =  self.resmha(x, prev = prev)
        x = self.ln1(x + rmha)
        x = self.ln2(x + self.ff(x))

        if last:
            return x
        return x, prev
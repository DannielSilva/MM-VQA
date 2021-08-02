import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torchvision import models
import math
import numpy as np
import torch.nn.functional as F
import timm

class A:
  def __init__(self,a):
    self.a=a
  
  def method(self):
    print("aa",self.a)

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

models_dict = {'resnet152':[models.resnet152, [2048,1024,512,256,64]],
               'tf_efficientnetv2_m':[timm.create_model,[1280,512,160,48,24]]}
def get_image_encoder(name):
    m, channel_size = models_dict[name]
    if 'resnet' in name:
        return m(pretrained=True), channel_size
    elif 'efficientnetv2' in name:
        return m(name, pretrained=True), channel_size

def get_transfer(args):
    if 'resnet' in args.cnn_encoder:
        print('Using resnet', args.cnn_encoder)
        return ResNetTransfer(args)
    elif 'efficientnetv2' in args.cnn_encoder:
        print('Using efficientnetv2', args.cnn_encoder)
        return EffNetV2Transfer(args)
    else:
        raise NotImplementedError

class Transfer(nn.Module):
    def __init__(self,args):
        super(Transfer, self).__init__()

        self.args = args
        self.model, self.channel_size = get_image_encoder(args.cnn_encoder)
        #self.model = models.resnet152(pretrained=True)
        # for p in self.parameters():
        #     p.requires_grad=False
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(self.channel_size[0], args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.gap2 = nn.AdaptiveAvgPool2d((1,1))
        self.conv3 = nn.Conv2d(self.channel_size[1], args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.gap3 = nn.AdaptiveAvgPool2d((1,1))
        self.conv4 = nn.Conv2d(self.channel_size[2], args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.gap4 = nn.AdaptiveAvgPool2d((1,1))
        self.conv5 = nn.Conv2d(self.channel_size[3], args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.gap5 = nn.AdaptiveAvgPool2d((1,1))
        self.conv7 = nn.Conv2d(self.channel_size[4], args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.gap7 = nn.AdaptiveAvgPool2d((1,1))

class ResNetTransfer(Transfer):
    def __init__(self, args):
        super().__init__(args)
    
    def forward(self, img):
        modules2 = list(self.model.children())[:-2]
        fix2 = nn.Sequential(*modules2)
        v_2 = self.gap2(self.relu(self.conv2(fix2(img)))).view(-1,self.args.hidden_size)
        modules3 = list(self.model.children())[:-3]
        fix3 = nn.Sequential(*modules3)
        v_3 = self.gap3(self.relu(self.conv3(fix3(img)))).view(-1,self.args.hidden_size)
        modules4 = list(self.model.children())[:-4]
        fix4 = nn.Sequential(*modules4)
        v_4 = self.gap4(self.relu(self.conv4(fix4(img)))).view(-1,self.args.hidden_size)
        modules5 = list(self.model.children())[:-5]
        fix5 = nn.Sequential(*modules5)
        v_5 = self.gap5(self.relu(self.conv5(fix5(img)))).view(-1,self.args.hidden_size)
        modules7 = list(self.model.children())[:-7]
        fix7 = nn.Sequential(*modules7)
        v_7 = self.gap7(self.relu(self.conv7(fix7(img)))).view(-1,self.args.hidden_size)
        return v_2, v_3, v_4, v_5, v_7

class EffNetV2Transfer(Transfer):
    def __init__(self, args):
        super().__init__(args)
    
    def forward(self, img):
        #5 viz

        #1st block
        first = list(self.model.children())[:3]
        first_nn = nn.Sequential(*first)
        v_7 = self.gap7(self.relu(self.conv7(first_nn(img)))).view(-1,self.args.hidden_size)

        #blocks is a Sequential that contains the individual blocks
        blocks = list(self.model.children())[3]

        #2nd block (1st sub block in blocks)
        first_b = list(blocks[:2])
        second = first + first_b
        second_nn  = nn.Sequential(*second)
        v_5 = self.gap5(self.relu(self.conv5(second_nn(img)))).view(-1,self.args.hidden_size)

        #3rd block (2nd sub block in blocks)
        second_b = list(blocks[:4])
        third = first + second_b
        third_nn = nn.Sequential(*third)
        v_4 = self.gap4(self.relu(self.conv4(third_nn(img)))).view(-1,self.args.hidden_size)

        #4th block
        forth = list(self.model.children())[:-5]
        forth_b_nn = nn.Sequential(*forth)
        v_3 = self.gap3(self.relu(self.conv3(forth_b_nn(img)))).view(-1,self.args.hidden_size)

        #5th block
        fifth = list(self.model.children())[:-2]
        fifth_b_nn = nn.Sequential(*fifth )
        v_2 = self.gap2(self.relu(self.conv2(fifth_b_nn(img)))).view(-1,self.args.hidden_size)

        return v_2, v_3, v_4, v_5, v_7

class MultiHeadedSelfAttention(nn.Module):
    def __init__(self,args):
        super(MultiHeadedSelfAttention,self).__init__()
        self.proj_q = nn.Linear(args.hidden_size, args.hidden_size)
        self.proj_k = nn.Linear(args.hidden_size, args.hidden_size)
        self.proj_v = nn.Linear(args.hidden_size, args.hidden_size)
        self.drop = nn.Dropout(args.hidden_dropout_prob)
        self.scores = None
        self.n_heads = args.heads
    def forward(self, x, mask):
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (self.split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))
        h = (scores @ v).transpose(1, 2).contiguous()
        h = self.merge_last(h, 2)
        self.scores = scores
        return h
    def split_last(self, x, shape):
        shape = list(shape)
        assert shape.count(-1) <= 1
        if -1 in shape:
            shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
        return x.view(*x.size()[:-1], *shape)
    def merge_last(self, x, n_dims):
        s = x.size()
        assert n_dims > 1 and n_dims < len(s)
        return x.view(*s[:-n_dims], -1)

class PositionWiseFeedForward(nn.Module):
    def __init__(self,args):
        super(PositionWiseFeedForward,self).__init__()
        self.fc1 = nn.Linear(args.hidden_size, args.hidden_size*4)
        self.fc2 = nn.Linear(args.hidden_size*4, args.hidden_size)
    def forward(self, x):
        return self.fc2(gelu(self.fc1(x)))

class BertLayer(nn.Module):
    def __init__(self,args, share='all', norm='pre'):
        super(BertLayer, self).__init__()
        self.share = share
        self.norm_pos = norm
        self.norm1 = nn.LayerNorm(args.hidden_size, eps=1e-12)
        self.norm2 = nn.LayerNorm(args.hidden_size, eps=1e-12)
        self.drop1 = nn.Dropout(args.hidden_dropout_prob)
        self.drop2 = nn.Dropout(args.hidden_dropout_prob)
        if self.share == 'ffn':
            self.attention = nn.ModuleList([MultiHeadedSelfAttention(args) for _ in range(args.n_layers)])
            self.proj = nn.ModuleList([nn.Linear(args.hidden_size, args.hidden_size) for _ in range(args.n_layers)])
            self.feedforward = PositionWiseFeedForward(args)
        elif self.share == 'att':
            self.attention = MultiHeadedSelfAttention(args)
            self.proj = nn.Linear(args.hidden_size, args.hidden_size)
            self.feedforward = nn.ModuleList([PositionWiseFeedForward(args) for _ in range(args.n_layers)])
        elif self.share == 'all':
            self.attention = MultiHeadedSelfAttention(args)
            self.proj = nn.Linear(args.hidden_size, args.hidden_size)
            self.feedforward = PositionWiseFeedForward(args)
        elif self.share == 'none':
            self.attention = nn.ModuleList([MultiHeadedSelfAttention(args) for _ in range(args.n_layers)])
            self.proj = nn.ModuleList([nn.Linear(args.hidden_size, args.hidden_size) for _ in range(args.n_layers)])
            self.feedforward = nn.ModuleList([PositionWiseFeedForward(args) for _ in range(args.n_layers)])
    def forward(self, hidden_states, attention_mask, layer_num):
        if self.norm_pos == 'pre':
            if isinstance(self.attention, nn.ModuleList):
                h = self.proj[layer_num](self.attention[layer_num](self.norm1(hidden_states), attention_mask))
            else:
                h = self.proj(self.attention(self.norm1(hidden_states), attention_mask))
            out = hidden_states + self.drop1(h)
            if isinstance(self.feedforward, nn.ModuleList):
                h = self.feedforward[layer_num](self.norm1(out))
            else:
                h = self.feedforward(self.norm1(out))
            out = out + self.drop2(h)
        if self.norm_pos == 'post':
            if isinstance(self.attention, nn.ModuleList):
                h = self.proj[layer_num](self.attention[layer_num](hidden_states, attention_mask))
            else:
                h = self.proj(self.attention(hidden_states, attention_mask))
            out = self.norm1(hidden_states + self.drop1(h))
            if isinstance(self.feedforward, nn.ModuleList):
                h = self.feedforward[layer_num](out)
            else:
                h = self.feedforward(out)
            out = self.norm2(out + self.drop2(h))
        return out

def get_bert_model(args):
    if args.task == 'distillation':
            bert_name = args.clinicalbert
    else:
        bert_name = 'bert-base-uncased'
    return bert_name

class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer,self).__init__()
        bert_name = get_bert_model(args)

        base_model = AutoModel.from_pretrained(bert_name)
        bert_model = nn.Sequential(*list(base_model.children())[0:])
        self.bert_embedding = bert_model[0]
#         self.embed = Embeddings(args)
        self.trans = get_transfer(args)
        self.blocks = BertLayer(args,share='none', norm='pre')
        self.n_layers = args.n_layers
    def forward(self, img, input_ids, token_type_ids, mask):
        v_2, v_3, v_4, v_5, v_7 = self.trans(img)
#         h = self.embed(input_ids, token_type_ids)
        #print('shape input_ids',input_ids.shape)
        h = self.bert_embedding(input_ids=input_ids, token_type_ids=token_type_ids, position_ids=None)
        #print('h', h.shape)
        for i in range(len(h)):
            h[i][1] = v_2[i]
        for i in range(len(h)):
            h[i][2] = v_3[i]
        for i in range(len(h)):
            h[i][3] = v_4[i]
        for i in range(len(h)):
            h[i][4] = v_5[i]
        for i in range(len(h)):
            h[i][5] = v_7[i]
        for i in range(self.n_layers):
            h = self.blocks(h, mask, i)
        return h

class Model(nn.Module):
    def __init__(self,args):
        super(Model,self).__init__()
        self.transformer = get_transformer_model(args)
        self.fc1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.activ1 = nn.Tanh()
        self.classifier = nn.Sequential(nn.Linear(args.hidden_size, args.hidden_size),
                                        nn.LayerNorm(args.hidden_size, eps=1e-12, elementwise_affine=True),
                                        nn.Linear(args.hidden_size, args.vocab_size))
        self.task = args.task
        self.dataset = args.dataset

    def forward(self, img, input_ids, segment_ids, input_mask):
        if self.dataset == 'roco':
            h = self.transformer(img, input_ids, segment_ids, input_mask)
            if self.task == 'MLM':
                pooled_h = self.activ1(self.fc1(h))
                logits = self.classifier(pooled_h)
            elif self.task == 'distillation':
                logits = h
            return logits

        elif self.dataset == 'VQA-Med':
            h = self.transformer(img, input_ids, segment_ids, input_mask)
            pooled_h = self.activ1(self.fc1(h.mean(1)))
            logits = self.classifier(pooled_h)
            return logits, 0,0



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
            nn.GELU(),
            nn.Linear(4 * emb, emb),
            nn.Dropout(dp2),
        )

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
    
    def forward(self, x, prev = None): ## add & norm later.
        rmha, prev =  self.resmha(x, prev = prev)
        x = self.ln1(x + rmha)
        x = self.ln2(x + self.ff(x))

        return x, prev

def get_transformer_model(args):
    if 'transformer' in args.transformer_model:
        print('Using regular transformer')
        return Transformer(args)
    elif 'realformer' in args.transformer_model:
        print('Using RealFormer')
        return RealFormer(args)
    else:
        raise NotImplementedError

class RealFormer(nn.Module):
    def __init__(self,args):
        super().__init__()
        bert_name = get_bert_model(args)

        base_model = AutoModel.from_pretrained(bert_name)
        bert_model = nn.Sequential(*list(base_model.children())[0:])
        self.bert_embedding = bert_model[0]
#         self.embed = Embeddings(args)
        self.trans = get_transfer(args)
        head_cnt = 8
        self.mains = nn.Sequential(*[ResEncoderBlock(emb_s = args.hidden_size // head_cnt, head_cnt = head_cnt, dp1 = 0.1, dp2 = 0.1) for _ in range(args.n_layers)])
    def forward(self, img, input_ids, token_type_ids, mask):
        v_2, v_3, v_4, v_5, v_7 = self.trans(img)
        h = self.bert_embedding(input_ids=input_ids, token_type_ids=token_type_ids, position_ids=None)
        for i in range(len(h)):
            h[i][1] = v_2[i]
        for i in range(len(h)):
            h[i][2] = v_3[i]
        for i in range(len(h)):
            h[i][3] = v_4[i]
        for i in range(len(h)):
            h[i][4] = v_5[i]
        for i in range(len(h)):
            h[i][5] = v_7[i]

        prev = None
        for resencoder in self.mains:
            h, prev = resencoder(h, prev = prev)
        return h
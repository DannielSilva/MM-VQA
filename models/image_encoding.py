import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torchvision import models
import timm
from models.serf import SERF

# dictionary storing the models to use and the channel_size to be used to retrieve the image tokens
# first key: num_viz 
# second key: image encoder name

models_dict = {5:   {'resnet152':[models.resnet152, [2048,1024,512,256,64]],
                     #'tf_efficientnetv2_m':[timm.create_model,[1280,512,160,48,24]]
                     'tf_efficientnetv2_m':[timm.create_model,[24, 48, 80, 176, 512]]
               },

               7:   {'tf_efficientnetv2_m':[timm.create_model,[24,48,80,160,176,304,512]]}
              }
def get_image_encoder(args):
    m, channel_size = models_dict[args.num_vis][args.cnn_encoder]
    if 'resnet' in args.cnn_encoder:
        return m(pretrained=True), channel_size
    elif 'efficientnetv2' in args.cnn_encoder:
        #return m(args.cnn_encoder, pretrained=True), channel_size
        return m(args.cnn_encoder, features_only=True,pretrained=True), channel_size

def get_transfer(args):
    if 'resnet' in args.cnn_encoder:
        print('Using resnet', args.cnn_encoder)
        return ResNetTransfer(args)
    elif 'efficientnetv2' in args.cnn_encoder:
        print('Using efficientnetv2', args.cnn_encoder)
        if args.num_vis == 5:
            print('Using timm feature extraction')
            return Timm_EFfNetV2(args)
            #return EffNetV2Transfer(args)
        elif args.num_vis == 7:
            return EffNetV2Transfer7Tokens(args)
    else:
        raise NotImplementedError

class Transfer(nn.Module):
    def __init__(self,args):
        super(Transfer, self).__init__()

        self.args = args
        self.model, self.channel_size = get_image_encoder(args)
        #self.model = models.resnet152(pretrained=True)
        # for p in self.parameters():
        #     p.requires_grad=False
        self.serf = SERF()
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
        v_2 = self.gap2(self.serf(self.conv2(fix2(img)))).view(-1,self.args.hidden_size)
        modules3 = list(self.model.children())[:-3]
        fix3 = nn.Sequential(*modules3)
        v_3 = self.gap3(self.serf(self.conv3(fix3(img)))).view(-1,self.args.hidden_size)
        modules4 = list(self.model.children())[:-4]
        fix4 = nn.Sequential(*modules4)
        v_4 = self.gap4(self.serf(self.conv4(fix4(img)))).view(-1,self.args.hidden_size)
        modules5 = list(self.model.children())[:-5]
        fix5 = nn.Sequential(*modules5)
        v_5 = self.gap5(self.serf(self.conv5(fix5(img)))).view(-1,self.args.hidden_size)
        modules7 = list(self.model.children())[:-7]
        fix7 = nn.Sequential(*modules7)
        v_7 = self.gap7(self.serf(self.conv7(fix7(img)))).view(-1,self.args.hidden_size)
        return v_2, v_3, v_4, v_5, v_7

#has problems to use after because it does not have .forward_features() method
class Timm_EFfNetV2(Transfer):
    def __init__(self, args):
        super().__init__(args)
        print('timm serf')
        self.relu = nn.ReLU()

    def forward(self, img):
        o = self.model(img)
        #v_0 = self.gap2(self.serf(F.normalize(self.conv2(o[0]),dim=1))).view(-1,self.args.hidden_size)
        v_0 = self.gap2(self.serf(self.conv2(o[0]))).view(-1,self.args.hidden_size)
        v_1 = self.gap3(self.serf(self.conv3(o[1]))).view(-1,self.args.hidden_size)
        v_2 = self.gap4(self.serf(self.conv4(o[2]))).view(-1,self.args.hidden_size)
        v_3 = self.gap5(self.serf(self.conv5(o[3]))).view(-1,self.args.hidden_size)
        v_4 = self.gap7(self.serf(self.conv7(o[4]))).view(-1,self.args.hidden_size)
        #import IPython; IPython.embed(); import sys; sys.exit(0)
        return v_0, v_1, v_2, v_3, v_4
        #return F.normalize(v_0, dim=1),  F.normalize(v_1, dim=1), F.normalize(v_2, dim=1), F.normalize(v_3, dim=1), F.normalize(v_4, dim=1)



class EffNetV2Transfer(Transfer):
    def __init__(self, args):
        super().__init__(args)
        print('Effnetv2 using SERF')
    
    def forward(self, img):
        #5 viz

        #1st block
        first = list(self.model.children())[:3]
        first_nn = nn.Sequential(*first)
        v_7 = self.gap7(self.serf(self.conv7(first_nn(img)))).view(-1,self.args.hidden_size)

        #blocks is a Sequential that contains the individual blocks
        blocks = list(self.model.children())[3]

        #2nd block (1st sub block in blocks)
        first_b = list(blocks[:2])
        second = first + first_b
        second_nn  = nn.Sequential(*second)
        v_5 = self.gap5(self.serf(self.conv5(second_nn(img)))).view(-1,self.args.hidden_size)

        #3rd block (2nd sub block in blocks)
        second_b = list(blocks[:4])
        third = first + second_b
        third_nn = nn.Sequential(*third)
        v_4 = self.gap4(self.serf(self.conv4(third_nn(img)))).view(-1,self.args.hidden_size)

        #4th block
        forth = list(self.model.children())[:-5]
        forth_b_nn = nn.Sequential(*forth)
        v_3 = self.gap3(self.serf(self.conv3(forth_b_nn(img)))).view(-1,self.args.hidden_size)

        #5th block
        fifth = list(self.model.children())[:-2]
        fifth_b_nn = nn.Sequential(*fifth )
        v_2 = self.gap2(self.serf(self.conv2(fifth_b_nn(img)))).view(-1,self.args.hidden_size)

        return v_2, v_3, v_4, v_5, v_7

class EffNetV2Transfer7Tokens(nn.Module):
    def __init__(self, args):
        super().__init__()
        print('effnet 7 tokens with serf')
        self.args = args
        
        self.model, self.channel_size = get_image_encoder(args)
        self.serf = SERF()

        self.conv = nn.ModuleList([])
        self.gap = nn.ModuleList([])
        for i, channel_size in enumerate(self.channel_size):
            self.conv.append(nn.Conv2d(channel_size, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False))
            self.gap.append(nn.AdaptiveAvgPool2d((1,1)))
        
    def forward(self, img):
        first = list(self.model.children())[:3]
        blocks= list(self.model.children())[3]

        # block_0 = first + list(blocks[:1])
        # fix_0 = nn.Sequential(*block_0)

        block_0 = first + list(blocks[:(0+1)])
        block_0_nn = nn.Sequential(*block_0)
        viz_0 = self.gap[0](self.serf(self.conv[0](block_0_nn(img)))).view(-1,self.args.hidden_size)

        block_1 = first + list(blocks[:(1+1)])
        block_1_nn = nn.Sequential(*block_1)
        viz_1 = self.gap[1](self.serf(self.conv[1](block_1_nn(img)))).view(-1,self.args.hidden_size)

        block_2 = first + list(blocks[:(2+1)])
        block_2_nn = nn.Sequential(*block_2)
        viz_2 = self.gap[2](self.serf(self.conv[2](block_2_nn(img)))).view(-1,self.args.hidden_size)

        block_3 = first + list(blocks[:(3+1)])
        block_3_nn = nn.Sequential(*block_3)
        viz_3 = self.gap[3](self.serf(self.conv[3](block_3_nn(img)))).view(-1,self.args.hidden_size)

        block_4 = first + list(blocks[:(4+1)])
        block_4_nn = nn.Sequential(*block_4)
        viz_4 = self.gap[4](self.serf(self.conv[4](block_4_nn(img)))).view(-1,self.args.hidden_size)

        block_5 = first + list(blocks[:(5+1)])
        block_5_nn = nn.Sequential(*block_5)
        viz_5 = self.gap[5](self.serf(self.conv[5](block_5_nn(img)))).view(-1,self.args.hidden_size)

        block_6 = first + list(blocks[:(6+1)])
        block_6_nn = nn.Sequential(*block_6)
        viz_6 = self.gap[6](self.serf(self.conv[6](block_6_nn(img)))).view(-1,self.args.hidden_size)
        return viz_0 + viz_1 + viz_2 + viz_3 + viz_4 + viz_5 + viz_6
        # viz = []
        # for b in range(len(blocks)):
        #     block_b = first + list(blocks[:(b+1)])
        #     block_b_nn = nn.Sequential(*block_b)
        #     viz.append(self.gap[b](self.serf(self.conv[b](block_b_nn(img)))).view(-1,self.args.hidden_size))
            
        # return viz[0], viz[1], viz[2], viz[3], viz[4], viz[5], viz[6]
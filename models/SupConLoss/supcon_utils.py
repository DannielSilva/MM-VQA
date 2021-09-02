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
from tqdm import tqdm
import numpy as np
import random
from transformers import BertTokenizer, BertModel
from googletrans import Translator
import os
from PIL import Image
from roco_utils import encode_text
from torch.utils.data import Dataset, DataLoader


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

model_dict = {
    'resnet18':  512,
    'resnet34':  512,
    'resnet50':  2048,
    'resnet101': 2048,
    'resnet152': 2048,
    'efficientnet-b3': 1536,
    'efficientnet-b5': 2048,
    'tf_efficientnetv2_m': 1280
}

class SupConEncoder(nn.Module):
    """backbone + projection head"""
    def __init__(self, encoder, name='resnet152', head='mlp', feat_dim=128):
        super(SupConEncoder, self).__init__()
        dim_in = model_dict[name]
        self.encoder = encoder
        self.name = name
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        if 'resnet' in name:
            self.encoder.fc = nn.Sequential()
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
        feat = self.features(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat

    def features(self,x):
        if 'resnet' in self.name:
            return self.encoder(x)
        elif 'efficientnetv2' in self.name:
            return self.gap(self.encoder.forward_features(x)).squeeze()
        elif 'efficientnet' in self.name:
            return self.encoder.extract_features(x)

def get_supcon_model(model, args):
    return SupConEncoder(model.transformer.trans.model, name=args.cnn_encoder)

def jaccard_similarity(doc1, doc2): 
    
    # List the unique words in a document
    words_doc1 = set(doc1.lower().split()) 
    words_doc2 = set(doc2.lower().split())
    
    # Find the intersection of words list of doc1 & doc2
    intersection = words_doc1.intersection(words_doc2)

    # Find the union of words list of doc1 & doc2
    union = words_doc1.union(words_doc2)
        
    # Calculate Jaccard similarity score 
    # using length of intersection set divided by length of union set
    if len(union) != 0:
        return float(len(intersection)) / len(union)
    else:
        print('union == 0\n1st: ', doc1,'\n2nd: ',doc2)
        return 0.0

def buildMask(bsz,caption, aug):
    mask = torch.zeros(bsz, bsz, dtype=torch.float)
    for c1 in range(len(caption)):
        for c2 in range(len(aug)):
            mask[c1,c2] = jaccard_similarity(caption[c1],aug[c2])
    
    return mask

class ROCO_SupCon(Dataset):
    def __init__(self, args, df, tfm, keys, mode):
        self.df = df.values
        self.args = args
        self.path = args.data_dir
        self.tfm = tfm
        self.keys = keys
        self.mode = mode

        self.translator = Translator()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.clinicalbert = None
        
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name = self.df[idx,1] 
        path = os.path.join(self.path, self.mode, 'radiology', 'images',name)

        img = Image.open(path).convert('RGB')
        
        if self.tfm:
            img = self.tfm(img)
    
        caption = self.df[idx, 2].strip()       
        tokens, segment_ids, input_mask, targets = encode_text(caption, self.tokenizer, self.keys, self.args, self.clinicalbert)

        aug_caption = self.translate_caption(caption)
        aug_tokens, _, _, aug_targets = encode_text(aug_caption, self.tokenizer, self.keys, self.args, self.clinicalbert)
        

        return img, tokens, aug_tokens, segment_ids, input_mask, targets, aug_targets, caption, aug_caption

    def translate_caption(self, caption):
        langs = ['fr','de','pt']
        l = random.choice(langs)
        result = self.translator.translate(caption, src='en', dest=l)
        final = self.translator.translate(result.text, src=l, dest='en')
        return final.text

def process_tensors(img,caption_token,aug_tokens,segment_ids,attention_mask,target,aug_targets):
    def cat_tensors(a,b):
        return torch.cat([a,b], dim=0)
    return cat_tensors(img[0],img[1]),cat_tensors(caption_token,aug_tokens),cat_tensors(segment_ids,segment_ids),cat_tensors(attention_mask,attention_mask),cat_tensors(target,aug_targets)

def train_one_epoch(loader, model, supcon_model, criterion, supcon_loss, optimizer, device, args, epoch):

    model.train()
    train_loss = []
    PREDS = []
    TARGETS = []
    bar = tqdm(loader, leave=False)
    for i, (img, caption_token,aug_tokens,segment_ids,attention_mask,target,aug_targets,caption_text,aug_text) in enumerate(bar):
        img,caption_token,segment_ids,attention_mask,target = process_tensors(img,caption_token,aug_tokens,segment_ids,attention_mask,target,aug_targets)
        img, caption_token,segment_ids,attention_mask,target = img.to(device), caption_token.to(device), segment_ids.to(device), attention_mask.to(device), target.to(device)
        
        caption_token = caption_token.squeeze(1)
        attention_mask = attention_mask.squeeze(1)
    
        loss_func = criterion
        optimizer.zero_grad()

        
        logits = model(img, caption_token, segment_ids, attention_mask)
        logits = logits.log_softmax(-1)  # (bs x seq_len x vocab_size)
        loss = loss_func(logits.permute(0,2,1), target)  
        
        bsz = img.shape[0] //2
        features = supcon_model(img) # (bs x feat_dim)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0) # (bs//2 x feat_dim), (bs//2 x feat_dim)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1) # (bs//2, 2, feat_dim)

        mask = buildMask(bsz,caption_text, aug_text)
        loss_supcon = supcon_loss(features, mask=mask)

        loss = loss + loss_supcon

        loss.backward()
        optimizer.step()    

        # print('e')
        # import IPython; IPython.embed(); import sys; sys.exit(0)
           
    
        bool_label = target > 0

        pred = logits[bool_label, :].argmax(1)
        valid_labels = target[bool_label]   
        
        PREDS.append(pred)
        TARGETS.append(valid_labels)
        
        acc = (pred == valid_labels).type(torch.float).mean() * 100.

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        
        
        bar.set_description('train_loss: %.5f, train_acc: %.2f' % (loss_np, acc))
        

        
    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()

#     # Calculate total accuracy
    total_acc = (PREDS == TARGETS).mean() * 100.


    return np.mean(train_loss), total_acc

'''delete this vvvv'''
def validate(loader, model, supcon_model,criterion, supcon_loss, scaler, device, args, epoch):

    model.eval()
    val_loss = []

    PREDS = []
    TARGETS = []
    bar = tqdm(loader, leave=False)

    with torch.no_grad():
        for i, (img, caption_token,segment_ids,attention_mask,target) in enumerate(bar):

            img, caption_token,segment_ids,attention_mask,target = img.to(device), caption_token.to(device), segment_ids.to(device), attention_mask.to(device), target.to(device)
            caption_token = caption_token.squeeze(1)
            attention_mask = attention_mask.squeeze(1)
            
            loss_func = criterion

            
            logits = model(img, caption_token, segment_ids, attention_mask)
            
            logits = logits.log_softmax(-1)  # (bs x seq_len x vocab_size)
            loss = loss_func(logits.permute(0,2,1), target)
                    

            
            bool_label = target > 0
            pred = logits[bool_label, :].argmax(1)
            valid_labels = target[bool_label]   
        
            PREDS.append(pred)
            TARGETS.append(valid_labels)
            
            acc = (pred == valid_labels).type(torch.float).mean() * 100.

            loss_np = loss.detach().cpu().numpy()

            val_loss.append(loss_np)

            bar.set_description('val_loss: %.5f, val_acc: %.5f' % (loss_np, acc))
           

        val_loss = np.mean(val_loss)

    
    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()

    # Calculate total accuracy
    total_acc = (PREDS == TARGETS).mean() * 100.

    return val_loss, PREDS, total_acc
'''delete this ^^^^'''
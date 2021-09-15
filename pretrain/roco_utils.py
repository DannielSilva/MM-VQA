import os
import numpy as np
import pandas as pd
import math
import torch
import random
import wandb
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
import pickle

import pytorch_lightning as pl

import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from transformers import BertTokenizer, BertModel
import os
from PIL import Image
from transformers import AutoTokenizer, AutoModel

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_permutation(n):
    index_list = list(np.arange(n))
    perms = []
    for i in range(n):
        lst = index_list[:i] + index_list[i+1:]
        perms.append(np.random.choice(lst))
    return perms


def get_keywords(args):
    with open(os.path.join(args.data_dir, 'vocab', 'med_vocab.pkl'), 'rb') as f:
        key = pickle.load(f)

    keywords = []

    for k,v in key.items():
        keywords.extend(v)
    
    keywords_ = list(set(keywords))

    for word in keywords_:
        keywords.extend(word + '.')
    
    keywords = list(set(keywords))

    return keywords

def load_image_names(data_dir, split):
    with open(os.path.join(data_dir, split + '_image_names.pickle'), 'rb') as f:
      image_names = pickle.load(f)

    return image_names

def load_mlm_data(args):
    train_path = os.path.join(args.data_dir,'train','radiology')
    val_path = os.path.join(args.data_dir,'validation','radiology')
    test_path = os.path.join(args.data_dir,'test','radiology')

    train_image_names = os.listdir(os.path.join(train_path,'images'))
    val_image_names = os.listdir(os.path.join(val_path,'images'))
    # train_image_names = load_image_names(train_path,'train')
    # val_image_names = load_image_names(val_path,'val')
    # test_image_names = os.listdir(os.path.join(test_path,'images'))

    train_data = pd.read_csv(os.path.join(train_path,'traindata.csv'))
    train_data = train_data[train_data['name'].isin(train_image_names)]

    val_data = pd.read_csv(os.path.join(val_path, 'valdata.csv'))
    val_data = val_data[val_data['name'].isin(val_image_names)]

    # test_data = pd.read_csv(os.path.join(test_path, 'testdata.csv'))
    # test_data = test_data[test_data['name'].isin(test_image_names)]
    
    if args.train_pct != 1.0:
        train_data = train_data.sample(frac = args.train_pct)
    if args.valid_pct != 1.0:
        val_data = val_data.sample(frac = args.valid_pct)
    # test_data = test_data.sample(frac = args.test_pct)
        
    return train_data, val_data

def shuffle_list(some_list):
    length = len(some_list)
    for i in range(length):
        j = i + np.floor(np.random.uniform()*(length - i - 1))
        j = int(j)
        some_list[i], some_list[j] = some_list[j], some_list[i]

    return some_list

#Utils
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def distillation(caption, tokenizer, clinicalbert, args):
    output_label = []
    new_tokens = []

    t = tokenizer.tokenize(caption,truncation=True, max_length=(args.max_token_length-2)) #max length without [CLS] and [SEP]

    new_tokens.extend(t)

    item = tokenizer(caption, truncation=True) 
    with torch.no_grad():
        input_ids = torch.tensor(item['input_ids'], dtype=torch.long).unsqueeze(dim=0)
        attention_mask = torch.tensor(item['attention_mask'], dtype=torch.long).unsqueeze(dim=0)
        out = clinicalbert(input_ids, attention_mask, output_hidden_states=True)
        last = out[0].squeeze()

    length = len(item['input_ids']) - 1 # the output of the MMBERT only expects corresponding tokens
                                        # of the caption without CLS and SEP tokens // [1:length]
    output_label = last[1:length] #remove CLS
    assert (len(new_tokens)==output_label.shape[0]), "Token len must be equal to label len"
    
    return  new_tokens, output_label


def mask_word(sentence, tokenizer, keywords, args):
    tokens = sentence.split()
    output_label = []
    new_tokens = []

    for i, char in enumerate(tokens):
        if char in keywords:
            t = tokenizer.tokenize(char)
            for j in range(len(t)):
                prob = random.random()
                if prob < args.mlm_prob:
                    
                    output_label.extend([tokenizer.encode(t[j])[1]])
                    t[j] = '[MASK]'

                else:
                    output_label.extend([0])
            new_tokens.extend(t)
        else:
            t = tokenizer.tokenize(char)
            new_tokens.extend(t)
            output_label.extend([0]*len(t))
            
    assert (len(new_tokens)==len(output_label)), "Token len must be equal to label len"
    
    return  new_tokens, output_label

def encode_text(caption,tokenizer, keywords, args, clinicalbert):
    TOTAL_SPECIAL_TOKENS = args.num_vis + 3 #at least the visual tokens and [CLS] and two [SEP] will be used
    part1 = [0 for _ in range(args.num_vis)]
    
    #get token ids and remove [CLS] and [SEP] token id
    if args.task == 'MLM':
        caption, labels = mask_word(caption, tokenizer, keywords, args)
    elif args.task == 'distillation':
        #part1 = [torch.zeros(768, dtype=torch.float) for _ in range(5)]
        caption, labels = distillation(caption, tokenizer, clinicalbert, args)

    
    part2 = tokenizer.convert_tokens_to_ids(caption)
    part2 = part2[:args.max_position_embeddings-TOTAL_SPECIAL_TOKENS]
    labels = labels[:args.max_position_embeddings-TOTAL_SPECIAL_TOKENS]
    
    tokens = [tokenizer.cls_token_id] + part1 + [tokenizer.sep_token_id] + part2 + [tokenizer.sep_token_id]
    #labels = [0]*(2+len(part1)) + labels + [0]
    
    segment_ids = [0]*(len(part1)+2) + [1]*(len(part2[:args.max_position_embeddings-TOTAL_SPECIAL_TOKENS])+1)
    input_mask = [1]*len(tokens)
    n_pad = args.max_position_embeddings - len(tokens)
    tokens.extend([0]*n_pad)
    segment_ids.extend([0]*n_pad)
    input_mask.extend([0]*n_pad)
    if args.task == 'MLM':
        labels = [0]*(2+len(part1)) + labels + [0]
        labels.extend([0]*(n_pad))
        labels = torch.tensor(labels)
    elif args.task == 'distillation':
        # labels = [torch.zeros(768, dtype=torch.float)]*(2+len(part1)) + labels + [torch.zeros(768, dtype=torch.float)]
        # labels.extend([torch.zeros(768, dtype=torch.float)]*(n_pad))
        # labels = torch.stack(labels,dim=0)

        labels = torch.cat([torch.zeros((2+len(part1)),768, dtype=torch.float), labels, torch.zeros(1,768, dtype=torch.float)], dim=0)
        labels = torch.cat([labels, torch.zeros(n_pad, 768, dtype=torch.float)], dim=0)

    return torch.tensor(tokens,dtype=torch.long), torch.tensor(segment_ids,dtype=torch.long), torch.tensor(input_mask,dtype=torch.long), labels#torch.tensor(labels)


def calculate_bleu_score(preds,targets):
  bleu_per_answer = np.asarray([sentence_bleu([idx2ans[target].split()],idx2ans[pred].split()) for pred,target in zip(preds,targets)])
  return np.mean(bleu_per_answer)


def train_one_epoch(loader, model, criterion, optimizer, scaler, device, args, epoch):

    model.train()
    train_loss = []
    PREDS = []
    TARGETS = []
    bar = tqdm(loader, leave=False)
    for i, (img, caption_token,segment_ids,attention_mask,target) in enumerate(bar):

        img, caption_token,segment_ids,attention_mask,target = img.to(device), caption_token.to(device), segment_ids.to(device), attention_mask.to(device), target.to(device)
        
        caption_token = caption_token.squeeze(1)
        attention_mask = attention_mask.squeeze(1)
    
        loss_func = criterion
        optimizer.zero_grad()

        if args.mixed_precision:
            with torch.cuda.amp.autocast():
                logits = model(img, caption_token, segment_ids, attention_mask)
                if args.task == 'MLM':
                    logits = logits.log_softmax(-1)  # (bs x seq_len x vocab_size)
                    loss = loss_func(logits.permute(0,2,1), target)
                elif args.task == 'distillation':
                    loss = loss_func(logits, target)
        else:
            logits = model(img, caption_token, segment_ids, attention_mask)
            if args.task == 'MLM':
                    logits = logits.log_softmax(-1)  # (bs x seq_len x vocab_size)
                    loss = loss_func(logits.permute(0,2,1), target)
            elif args.task == 'distillation':
                loss = loss_func(logits, target)   


        if args.mixed_precision:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()    

        # logits = model(img, caption_token, segment_ids, attention_mask)
        # logits = logits.log_softmax(-1)  # (bs x seq_len x vocab_size)
        # loss = loss_func(logits.permute(0,2,1), target)

        # loss.backward()
        # optimizer.step()
           
        if args.task == 'MLM':
            bool_label = target > 0

            pred = logits[bool_label, :].argmax(1)
            valid_labels = target[bool_label]   
            
            PREDS.append(pred)
            TARGETS.append(valid_labels)
            
            acc = (pred == valid_labels).type(torch.float).mean() * 100.

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        #print('train_loss: %.5f' % (loss_np))
        if args.task == 'MLM':
            bar.set_description('train_loss: %.5f, train_acc: %.2f' % (loss_np, acc))
        
        elif args.task == 'distillation':
            bar.set_description('train_loss: %.5f' % (loss_np))

        '''wandb.log({'step_train_loss': loss_np,
            'step_train_acc': acc,
            'train_batch': epoch*len(loader) + i})'''
        
    if args.task == 'MLM':
        PREDS = torch.cat(PREDS).cpu().numpy()
        TARGETS = torch.cat(TARGETS).cpu().numpy()

#     # Calculate total accuracy
        total_acc = (PREDS == TARGETS).mean() * 100.

    elif args.task == 'distillation':
        total_acc = None

    return np.mean(train_loss), total_acc

def validate(loader, model, criterion, scaler, device, args, epoch, rec=True):

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

            if args.mixed_precision:
                with torch.cuda.amp.autocast():
                    logits = model(img, caption_token, segment_ids, attention_mask)
                    if args.task == 'MLM':
                        logits = logits.log_softmax(-1)  # (bs x seq_len x vocab_size)
                        loss = loss_func(logits.permute(0,2,1), target)
                    elif args.task == 'distillation':
                        loss = loss_func(logits, target)
            else:
                logits = model(img, caption_token, segment_ids, attention_mask)
                if args.task == 'MLM':
                    logits = logits.log_softmax(-1)  # (bs x seq_len x vocab_size)
                    loss = loss_func(logits.permute(0,2,1), target)
                elif args.task == 'distillation':
                    loss = loss_func(logits, target)
                    #import IPython; IPython.embed(); exit(0)       


            # logits = model(img, caption_token, segment_ids, attention_mask)
            # logits = logits.log_softmax(-1)  # (bs x seq_len x vocab_size)
            # loss = loss_func(logits.permute(0,2,1), target)

            if args.task == 'MLM':
                bool_label = target > 0
                pred = logits[bool_label, :].argmax(1)
                valid_labels = target[bool_label]   
            
                PREDS.append(pred)
                TARGETS.append(valid_labels)
                
                acc = (pred == valid_labels).type(torch.float).mean() * 100.

            loss_np = loss.detach().cpu().numpy()

            val_loss.append(loss_np)

            if args.task == 'MLM':
                bar.set_description('val_loss: %.5f, val_acc: %.5f' % (loss_np, acc))
            elif args.task == 'distillation':
                bar.set_description('val_loss: %.5f' % (loss_np))


            # if rec:
            #   wandb.log({'step_val_loss': loss_np,
            #       'step_val_acc': acc,
            #       'val_batch': epoch*len(loader) + i})

        val_loss = np.mean(val_loss)

    if args.task == 'MLM':
        PREDS = torch.cat(PREDS).cpu().numpy()
        TARGETS = torch.cat(TARGETS).cpu().numpy()

        # Calculate total accuracy
        total_acc = (PREDS == TARGETS).mean() * 100.

    elif args.task == 'distillation':
        total_acc = None

    if not rec:
      print('epoch_val_loss', val_loss,
                  'val_batch', total_acc)
    return val_loss, PREDS, total_acc
    
def test(loader):

    model.eval()

    PREDS = []
    TARGETS = []

    with torch.no_grad():
        for (img,caption_token,attention_mask,target) in tqdm(loader, leave=False):

            img, caption_token,segment_ids,attention_mask,target = img.to(device), caption_token.to(device), segment_ids.to(device), attention_mask.to(device), target.to(device)
            caption_token = caption_token.squeeze(1)
            attention_mask = attention_mask.squeeze(1)
            
            logits = model(img, caption_token, segment_ids, attention_mask)
        
            bool_label = target > 0
            pred = logits[bool_label, :].argmax(1)
            valid_labels = target[bool_label]   
        
            PREDS.append(pred)
            TARGETS.append(valid_labels)

    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()

    total_acc = (PREDS == TARGETS).mean() * 100.

    return PREDS, total_acc



'''section of code to test if the weights of the encoder model are being updated
despite the encoder in the forward method building nn.Sequential from the layers of the model'''

'''two ways, first by testing if the model stores a grad after processing, which means the weights will change'''
input_test_train={}
input_test_val = {}
def get_same_batch(loader, mode):
    if mode == 'train':
        dict_batch = input_test_train
    elif mode =='eval':
        dict_batch = input_test_val
    
    if not dict_batch:
        for i, (img, caption_token,segment_ids,attention_mask,target) in enumerate(loader):
            break
        dict_batch['i'] = i
        dict_batch['img'] = img
        dict_batch['caption_token'] = caption_token
        dict_batch['segment_ids'] = segment_ids
        dict_batch['attention_mask'] = attention_mask
        dict_batch['target'] = target
    return dict_batch['i'], dict_batch['img'], dict_batch['caption_token'], dict_batch['segment_ids'], dict_batch['attention_mask'],dict_batch['target']

counter_grads = {}
def initialize_counter_grads(model):
    for name, param in model.named_parameters():
        if 'fc' in name:
            continue
        if 'weight' in name:
            counter_grads[name] = torch.zeros(param.grad.shape)

def update_countner_grads(model):
    for name, param in model.named_parameters():
        if 'fc' in name:
            continue
        if 'weight' in name:
            temp = torch.zeros(param.grad.shape)
            temp[param.grad != 0] += 1
            counter_grads[name] += temp

'''check directly if the weights are different from the first and last epochs '''
weights_params = []
def initialize_params(model):
    a = list(model.parameters())[0].clone().detach()
    weights_params.append(a)

    b = list(model.parameters())[201].clone().detach()
    weights_params.append(b)

    c = list(model.parameters())[450].clone().detach()
    weights_params.append(c)
    
def compare_params(model):
    a = list(model.parameters())[0].clone().detach()

    b = list(model.parameters())[201].clone().detach()

    c = list(model.parameters())[450].clone().detach()

    print('comparing params 0', torch.equal(a, weights_params[0]))
    print('comparing params 201', torch.equal(b, weights_params[1]))
    print('comparing params 450', torch.equal(c, weights_params[2]))

def train_one_epoch_test_parameters(loader, model, criterion, optimizer, scaler, device, args, epoch):
    model.train()
    train_loss = []
    
    i,img, caption_token,segment_ids,attention_mask,target = get_same_batch(loader,'train')
    img, caption_token,segment_ids,attention_mask,target = img.to(device), caption_token.to(device), segment_ids.to(device), attention_mask.to(device), target.to(device)
    
    caption_token = caption_token.squeeze(1)
    attention_mask = attention_mask.squeeze(1)

    loss_func = criterion
    optimizer.zero_grad()


    logits = model(img, caption_token, segment_ids, attention_mask)
    logits = logits.log_softmax(-1)  # (bs x seq_len x vocab_size)
    loss = loss_func(logits.permute(0,2,1), target)
    

    loss.backward()
    print('eeek')
    #compare the weights of the params
    if epoch == 0:
        initialize_params(model.transformer.trans.model)
    if epoch == 9:
        compare_params(model.transformer.trans.model)

    "compare the grads"
    '''if epoch == 0:
        initialize_counter_grads(model.transformer.trans.model)
    update_countner_grads(model.transformer.trans.model)'''
    #import IPython;IPython.embed(); import sys;sys.exit(0)
    optimizer.step()    


    loss_np = loss.detach().cpu().numpy()
    train_loss.append(loss_np)
    #print('train_loss: %.5f' % (loss_np))
            
    total_acc = 0.1 #random

    return np.mean(train_loss), total_acc

def validate_test_parameters(loader, model, criterion, scaler, device, args, epoch):
    model.eval()
    val_loss = []
    
    with torch.no_grad():
        i,img, caption_token,segment_ids,attention_mask,target = get_same_batch(loader,'eval')
        #for i, (img, caption_token,segment_ids,attention_mask,target) in enumerate(loader):
        img, caption_token,segment_ids,attention_mask,target = img.to(device), caption_token.to(device), segment_ids.to(device), attention_mask.to(device), target.to(device)
        
        caption_token = caption_token.squeeze(1)
        attention_mask = attention_mask.squeeze(1)

        loss_func = criterion
        #optimizer.zero_grad()


        logits = model(img, caption_token, segment_ids, attention_mask)
        logits = logits.log_softmax(-1)  # (bs x seq_len x vocab_size)
        loss = loss_func(logits.permute(0,2,1), target)
        
        
        # loss.backward()
        # optimizer.step()    


        loss_np = loss.detach().cpu().numpy()
        val_loss.append(loss_np)
        #print('val_loss: %.5f' % (loss_np))
            
    total_acc = 0.1 #random

    return np.mean(val_loss), None, total_acc

class ROCO(Dataset):
    def __init__(self, args, df, tfm, keys, mode):
        self.df = df.values
        self.args = args
        self.path = args.data_dir
        self.tfm = tfm
        self.keys = keys
        self.mode = mode

        if args.task == 'distillation':
            self.tokenizer = AutoTokenizer.from_pretrained(args.clinicalbert, model_max_length=args.max_token_length)
        elif args.task == 'MLM':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.clinicalbert = None

        if args.task == 'distillation':
            self.clinicalbert = AutoModel.from_pretrained(args.clinicalbert)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        #info = self.df.iloc[idx]
        name = self.df[idx,1] 
        #name = info['PMC_ID']
                 
        path = os.path.join(self.path, self.mode, 'radiology', 'images',name)


        img = Image.open(path).convert('RGB')
        
        if self.tfm:
            img = self.tfm(img)
    
        caption = self.df[idx, 2].strip()
        #caption = info['caption'].strip()
        #caption = self.df.iloc[idx]['caption'].strip()
        
        
        tokens, segment_ids, input_mask, targets = encode_text(caption, self.tokenizer, self.keys, self.args, self.clinicalbert)

        return img, tokens, segment_ids, input_mask, targets




class ROCOModule(pl.LightningDataModule):
    def __init__(self, args):
        super(ROCOModule, self).__init__()

        self.args = args

    def setup(self, stage=None):

        train, val, test = load_mlm_data(self.args)

        train = train[train['name']!='PMC4240561_MA-68-291-g002.jpg'].reset_index(drop=True)

        train_tfm = transforms.Compose([transforms.Resize((224,224)), 
                                    transforms.RandomResizedCrop(224,scale=(0.95,1.05),ratio=(0.95,1.05)),
                                    transforms.RandomRotation(5),
                                    transforms.ColorJitter(brightness=0.05,contrast=0.05,saturation=0.05,hue=0.05),
                                    transforms.ToTensor(), 
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        val_tfm = transforms.Compose([transforms.Resize((224,224)), 
                                    transforms.ToTensor(), 
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.train = ROCO(self.args, train, train_tfm, 'train')
        self.val = ROCO(self.args, val, val_tfm, 'validation')
        self.test = ROCO(self.args, test, val_tfm, 'test')

    def train_dataloader(self):
        return DataLoader(self.train, batch_size = self.args.batch_size, shuffle = True, num_workers = self.args.num_workers, pin_memory = True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size = self.args.batch_size, shuffle = False, num_workers = self.args.num_workers, pin_memory = True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size = self.args.batch_size, shuffle = False, num_workers = self.args.num_workers, pin_memory = True)



class ROCOModel(pl.LightningModule):
    def __init__(self, args, model):
        super(ROCOModel,self).__init__()

        self.args = args
        self.model = model

    def training_step(self, batch, batch_idx):

        loss, acc = self.shared_step(batch, batch_idx)
        result = pl.TrainResult(loss)

        container = {'train_loss': loss, 'train_acc': acc}

        result.log_dict(container, on_step = True, on_epoch = True, prog_bar = True, logger = True)

        return result

    def validation_step(self, batch, batch_idx):

        loss, acc = self.shared_step(batch, batch_idx)
        result = pl.EvalResult(checkpoint_on = loss)

        container = {'val_loss': loss, 'val_acc': acc}        
        result.log_dict(container, on_step = True, on_epoch = True, prog_bar = True, logger = True)

        return result

    def shared_step(self, batch, batch_idx):

        img, caption_token, segment_ids, attention_mask, target = batch
        caption_token = caption_token.squeeze(1)
        attention_mask = attention_mask.squeeze(1)

        logits = self.model(img, caption_token, segment_ids, attention_mask)
        
        bool_label = target > 0
        pred = logits[bool_label, :].argmax(1)
        valid_labels = target[bool_label]  

        logits = logits.log_softmax(-1)  # (bs x seq_len x vocab_size)
        
        loss = self.loss_func(logits.permute(0,2,1), target)
        acc = (pred == valid_labels).type(torch.float).mean() * 100.

        return loss, acc

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr = self.args.lr)

        return [optimizer]

    def loss_func(self, pred, target):
        return nn.NLLLoss()(pred, target)


def calculate_bleu_score(preds,targets):
  bleu_per_answer = np.asarray([sentence_bleu([idx2ans[target].split()],idx2ans[pred].split()) for pred,target in zip(preds,targets)])
  return np.mean(bleu_per_answer)



class Embeddings(nn.Module):
    def __init__(self, args):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(args.vocab_size, 128, padding_idx=0)
        self.word_embeddings_2 = nn.Linear(128, args.hidden_size, bias=False)
        self.position_embeddings = nn.Embedding(args.max_position_embeddings, args.hidden_size)
        self.type_embeddings = nn.Embedding(3, args.hidden_size)
        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.len = args.max_position_embeddings
    def forward(self, input_ids, segment_ids, position_ids=None):
        if position_ids is None:
            if torch.cuda.is_available():
                position_ids = torch.arange(self.len, dtype=torch.long).cuda()
            else:
                position_ids = torch.arange(self.len, dtype=torch.long)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        words_embeddings = self.word_embeddings(input_ids)
        words_embeddings = self.word_embeddings_2(words_embeddings)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.type_embeddings(segment_ids)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class Transfer(nn.Module):
    def __init__(self,args):
        super(Transfer, self).__init__()

        self.args = args
        self.model = models.resnet152(pretrained=True)
        # for p in self.parameters():
        #     p.requires_grad=False
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(2048, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.gap2 = nn.AdaptiveAvgPool2d((1,1))
        self.conv3 = nn.Conv2d(1024, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.gap3 = nn.AdaptiveAvgPool2d((1,1))
        self.conv4 = nn.Conv2d(512, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.gap4 = nn.AdaptiveAvgPool2d((1,1))
        self.conv5 = nn.Conv2d(256, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.gap5 = nn.AdaptiveAvgPool2d((1,1))
        self.conv7 = nn.Conv2d(64, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.gap7 = nn.AdaptiveAvgPool2d((1,1))
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

class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer,self).__init__()
        if args.task == 'distillation':
            bert_name = args.clinicalbert
        else:
            bert_name = 'bert-base-uncased'

        base_model = AutoModel.from_pretrained(bert_name)
        bert_model = nn.Sequential(*list(base_model.children())[0:])
        self.bert_embedding = bert_model[0]
#         self.embed = Embeddings(args)
        self.trans = Transfer(args)
        self.blocks = BertLayer(args,share='none', norm='pre')
        self.n_layers = args.n_layers
    def forward(self, img, input_ids, token_type_ids, mask):
        v_2, v_3, v_4, v_5, v_7 = self.trans(img)
#         h = self.embed(input_ids, token_type_ids)
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
        for i in range(self.n_layers):
            h = self.blocks(h, mask, i)
        return h

class Model(nn.Module):
    def __init__(self,args):
        super(Model,self).__init__()
        self.transformer = Transformer(args)
        self.fc1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.activ1 = nn.Tanh()
        self.classifier = nn.Sequential(nn.Linear(args.hidden_size, args.hidden_size),
                                        nn.LayerNorm(args.hidden_size, eps=1e-12, elementwise_affine=True),
                                        nn.Linear(args.hidden_size, args.vocab_size))
        self.task = args.task
    def forward(self, img, input_ids, segment_ids, input_mask):
        h = self.transformer(img, input_ids, segment_ids, input_mask)
        if self.task == 'MLM':
            pooled_h = self.activ1(self.fc1(h))
            logits = self.classifier(pooled_h)
        elif self.task == 'distillation':
            logits = h
        return logits


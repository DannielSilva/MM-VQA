from transformers import MarianTokenizer, MarianMTModel
import os
import pandas as pd
import torch
import torch.nn as nn
import argparse
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class TransformerBackTranslation(nn.Module):
    def __init__(self,src, trg,device):
        super(TransformerBackTranslation, self).__init__()
        self.src = src
        self.trg = trg
        self.device = device

        model_name_forward = f'Helsinki-NLP/opus-mt-{src}-{trg}'
        self.model_forward = MarianMTModel.from_pretrained(model_name_forward)
        self.tokenizer_forward = MarianTokenizer.from_pretrained(model_name_forward)

        model_name_backwards = f'Helsinki-NLP/opus-mt-{trg}-{src}'
        self.model_backwards = MarianMTModel.from_pretrained(model_name_backwards)
        self.tokenizer_backwards = MarianTokenizer.from_pretrained(model_name_backwards)

    def translate(self,text):
        self.model_forward.eval()
        self.model_backwards.eval()
        with torch.no_grad():
            batch = self.tokenizer_forward(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
            gen = self.model_forward.generate(**batch)
            intermediate = self.tokenizer_forward.batch_decode(gen, skip_special_tokens=True) #list of string
            intermediate = intermediate#[0] #string

            batch = self.tokenizer_backwards(intermediate, return_tensors="pt", truncation=True, padding=True).to(self.device)
            gen = self.model_backwards.generate(**batch)
            res = self.tokenizer_backwards.batch_decode(gen, skip_special_tokens=True)
        return res#[0]

    def forward(self,text):
        return self.translate(text)

class Captions_Dataset(Dataset):
    def __init__(self, args, df):
        self.df = df.values
        self.args = args
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        id = self.df[idx,0] 
    
        caption = self.df[idx, 2].strip()

        return id,caption

parser = argparse.ArgumentParser(description="translation")
parser.add_argument('--roco_dir', type=str, default = '~/roco/train/radiology', help='path to dataset', required = False)
parser.add_argument('--language', type=str, default = 'fr', help='language to translate to for back translation', required = True)
parser.add_argument('--batch_size', type=int, default=16, help='batch_size.')
parser.add_argument('--num_workers', type=int, default=16, help='num works to generate data.')
parser.add_argument('--save_freq', type=int, default=2500, help='saving freq')
args = parser.parse_args()
train_path = args.roco_dir

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_data = pd.read_csv(os.path.join(train_path, 'traindata.csv'))


translator = TransformerBackTranslation(src='en',trg=args.language,device=device)
translator.to(device)
print('translator on device', translator.device)

#example
# txt=train_data.iloc[0]['caption']
# print('orig :', txt)
# print('after:',translator.translate(txt))

dataset = Captions_Dataset(args, train_data)
loader = DataLoader(dataset, batch_size = args.batch_size, shuffle=False, num_workers = args.num_workers)
print('len(loader), len(dataset)',len(loader), len(dataset))
res = []
bar = tqdm(loader, leave=False)
save_freq = len(loader) * args.save_freq // len(dataset)
print(f'saving every {save_freq}th batch')
for i, (id, caption) in enumerate(bar):
    out = translator(list(caption))

    info = pd.DataFrame()
    info['id'] = id
    info['caption_'+args.language] = out
    res.append(info)
    #import IPython; IPython.embed(); import sys; sys.exit(0)
    if i % save_freq == 0:
        print('saving file')
        final = pd.concat(res)
        final.to_csv(os.path.join(train_path,'caption_' + args.language +'.csv'), index=False, header=final.columns)
    #batch = tokenizer_forward(caption, return_tensors="pt", truncation=True, padding=True).to(self.device)

print('saving file')
final = pd.concat(res)
final.to_csv(os.path.join(train_path,'caption_' + args.language +'.csv'), index=False, header=final.columns)

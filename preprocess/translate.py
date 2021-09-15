import pandas as pd
import numpy as np
import threading
import math
import os
from googletrans import Translator
import argparse
from tqdm import tqdm
tqdm.pandas()
from preprocess.translate_transformers import TransformerBackTranslation
import torch

def split_dataframe(df, chunk_size = 1000): 
    chunks = list()
    num_chunks = math.ceil(df.shape[0] / chunk_size)
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks

def check_progress():
    before = train_data[(args.resume_pos-(args.num_threads * args.chunk_size)):args.resume_pos]
    before['result'] = np.where(before['caption'] == before['caption_fr'], 0, 1)
    print("number of not changed == number processed??",before['result'].value_counts()[0]==args.resume_pos)
    print('caption equals caption_fr',before['caption'].equals(before['caption_fr']))

def translate_googletrans(x,dest):
    translator = Translator()
    try:
        result = translator.translate(x['caption'], src='en', dest=dest)
        final = translator.translate(result.text, src=dest, dest='en')
    except Exception as e:
        print(str(e))
        return DEFAULT
    return final.text

def translate_transformer(x, dest):
    return transformer.translate(x['caption'])

def dummy(x):
    for i in range(len(x)):
        a = math.sqrt(i)
func = {'googletrans':translate_googletrans, 'transformer':translate_transformer }
def do_translate(t_num, dest):
    print('my num', t_num)
    df = chunks[args.resume_pos // args.chunk_size + t_num]
    print('t_num, chunk_size', t_num, len(df))
    f = func[args.method]
    print('func', args.method)
    df['caption_'+ dest] = df.progress_apply(lambda x: f(x,dest), axis=1)  #df.progress_apply(lambda x: dummy(x), axis=1)



parser = argparse.ArgumentParser(description="translation")
parser.add_argument('--roco_dir', type=str, default = '~/roco/train/radiology', help='path to dataset', required = False)
parser.add_argument('--language', type=str, default = 'fr', help='language to translate to for back translation', required = True)
parser.add_argument('--num_threads', type=int, default=25, help='number of threads.', required = False)
parser.add_argument('--chunk_size', type=int, default=1000, help='chunk size of dataframe for each threaad.', required = False)
parser.add_argument('--default', type=str, default = 'not yet', help='default to put in column', required = False)
parser.add_argument('--resume', action='store_true', default = False,  help='resume from a position', required = False)
parser.add_argument('--resume_pos', type=int, default=0, help='chunk size of dataframe for each threaad.')
parser.add_argument('--method', type=str, default='googletrans',
                        choices=['googletrans','transformer'], help='method of translation')



args = parser.parse_args()


train_path = args.roco_dir

DEFAULT =  args.default
if args.resume and args.resume_pos == 0:
    print('Cant resume in pos 0')
    import sys; sys.exit(0)
if args.resume:
    train_data = pd.read_csv(os.path.join(train_path, 'traindata_'+args.language+'.csv'))
    check_progress()

if not args.resume:
    train_data = pd.read_csv(os.path.join(train_path, 'traindata.csv'))
    train_data['caption_' + args.language] = DEFAULT

chunks = split_dataframe(train_data, chunk_size=args.chunk_size)
print('len chunks', len(chunks))
print('chunk size',len(chunks[0]))

threads = []
NUM_THREADS = args.num_threads #len(chunks) #

if args.method == 'transformer':
    print('It doesnt work this way with threads, use googletrans method')
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # transformer = TransformerBackTranslation(src='en',trg=args.language,device=device)
    # transformer.to(device)
    # print('transformer on device',device)

#import IPython; IPython.embed(); import sys; sys.exit(0)
for i in range(NUM_THREADS):
    t = threading.Thread(target=do_translate, args=(i,args.language,))
    t.daemon = True
    threads.append(t)

for i in range(NUM_THREADS):
    threads[i].start()

for i in range(NUM_THREADS):
    threads[i].join()

res = pd.concat(chunks)
res.to_csv(os.path.join(train_path,'traindata_' + args.language +'.csv'), index=False, header=res.columns)
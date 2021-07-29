import pandas as pd
import os
from tqdm import tqdm
import pickle
import argparse


#print(os.listdir(os.path.join(ROCO_PATH,"train")))

''' build dataframes for roco'''

def build_dataframe(split):

    licences = pd.read_csv(os.path.join(ROCO_PATH,split,"licences.txt"))
    #print(licences)

    captions = pd.read_csv(os.path.join(ROCO_PATH,split,"captions.txt"), sep='\t', names=['ROCO_ID','caption'],header=None)

    #print(captions)

    result = pd.merge(licences, captions, on="ROCO_ID")
    #print(result.iloc[0])
    columns = ['name','caption']
    df = result[columns]
    filename = 'traindata.csv' if split == 'train' else 'valdata.csv'
    print(filename)
    df.to_csv(os.path.join(ROCO_PATH,split,filename), index=False, header=columns)
    return df


# import IPython; IPython.embed(); exit(1)

''' retrieve vocabulary from the captions'''

def count_keywords(split, keywords):
    num_lines = sum(1 for line in open(os.path.join(ROCO_PATH,split,"keywords.txt"),'r'))
    print('num_lines',num_lines)
    with open(os.path.join(ROCO_PATH,split,"keywords.txt"), "r") as f:
    #     for i,line in enumerate(tqdm(f, total=num_lines)):
    #         keys = line.split('\t')[1:]
    #         for k in keys:
    #             if k != '':
    #                 k = k.strip()
    #                 if k not in keywords:
    #                     keywords.append(k)
    # print('len',len(keywords))
    # return keywords

        for i,line in enumerate(tqdm(f, total=num_lines)):
            roco_id = line.split('\t')[0]
            keys = line.split('\t')[1:]

            words = []
            for k in keys:
                if k != '':
                    words.append(k.strip())
            

            keywords[roco_id] = words
            
        return keywords



''' sort keywords '''

def sort_keywords():
    KEYWORDS_PATH = os.path.join(ROCO_PATH,'train')
    columns = ('id','keys')
    keywords_df = pd.read_csv( os.path.join(KEYWORDS_PATH,"keywords.txt"), sep='\t\t', names=columns, engine='python') #read and parse
    rows = [{'id':x, 'keys':y} for x, y in zip(keywords_df['id'], keywords_df['keys'])]

    traindata_df = pd.read_csv(os.path.join(KEYWORDS_PATH,"traindata.csv"))
    
    del keywords_df
    rows_list = []
    for row in rows:

            dict1 = {}
            if row['keys'] is None:
                continue
            keys = row['keys'].split('\t')
            
            # get input row in dictionary format
            # key = col_name
            dict1['id'] = row['id']
            dict1['keys'] = " ".join(keys)
            keys.sort()
            dict1['sorted_keys'] = "".join(keys)
            

            rows_list.append(dict1)

    df = pd.DataFrame(rows_list)
    res = pd.merge(df, traindata_df, on="id")
    res.drop('caption', axis=1, inplace=True)
    df = res.sort_values('sorted_keys')
    df.to_csv(os.path.join(KEYWORDS_PATH,'keywords_sorted_name.csv'), index=False, header=('id','keys','sorted_keys','name'))
    #import IPython; IPython.embed(); exit(0)
    return df

parser = argparse.ArgumentParser(description="preprocess roco tasks")
parser.add_argument('--task', type=str, choices=['dataframe', 'vocab', 'sort_keywords'], help="name for wandb run", required=True)
parser.add_argument('--roco_dir', type=str, default = 'D:\ist\Tese\ROCO\\', help='path to dataset', required = False)

args = parser.parse_args()
print(args)

ROCO_PATH = args.roco_dir

if args.task == 'dataframe':
    train_df = build_dataframe('train')
    val_df = build_dataframe('validation')


if args.task == 'vocab':
    keywords = {}
    keywords = count_keywords('train', keywords)
    keywords = count_keywords('validation', keywords)
    print('len',len(keywords))
    with open(os.path.join(ROCO_PATH,'vocab','med_vocab.pkl'), 'wb') as fp:
        pickle.dump(keywords, fp, protocol=pickle.DEFAULT_PROTOCOL)

if args.task == 'sort_keywords':
    df = sort_keywords()

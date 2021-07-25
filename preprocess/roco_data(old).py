import pandas as pd
import os
from tqdm import tqdm
import pickle

ROCO_PATH = 'D:\ist\Tese\ROCO\\'
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

DATA_DF = False
if DATA_DF:
    train_df = build_dataframe('train')
    val_df = build_dataframe('validation')
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

KEYWORDS = False
if KEYWORDS:
    keywords = {}
    keywords = count_keywords('train', keywords)
    keywords = count_keywords('validation', keywords)
    print('len',len(keywords))
    with open(os.path.join(ROCO_PATH,'vocab','med_vocab.pkl'), 'wb') as fp:
        pickle.dump(keywords, fp, protocol=pickle.DEFAULT_PROTOCOL)

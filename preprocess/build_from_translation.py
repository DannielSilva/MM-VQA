import argparse
import pandas as pd
import os

parser = argparse.ArgumentParser(description="building dataframe")
parser.add_argument('--roco_dir', type=str, default = '~/roco/train/radiology', help='path to dataset', required = False)
parser.add_argument('--languages', nargs='+')
args = parser.parse_args()
print(args)
train_path = args.roco_dir

train_data = pd.read_csv(os.path.join(train_path, 'traindata.csv'))


for l in args.languages:
    caption_data = pd.read_csv(os.path.join(train_path, 'caption_' + l + '.csv'))
    train_data['caption_' + l] = caption_data['caption_' + l]
    
train_data.to_csv(os.path.join(train_path,'traindata_translated.csv'), index=False, header=train_data.columns)
#import IPython; IPython.embed(); import sys; sys.exit(0)

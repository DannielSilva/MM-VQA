import pandas as pd
import os

#ImageCLEF 2019 - MED VQA
data_dir = os.path.join('..', 'ImageClef-2019-VQA-Med')
train_dir = os.path.join(data_dir, 'Train')
val_dir = os.path.join(data_dir, 'Val')
test_dir = os.path.join(data_dir, 'Test')
def create_df(d_dir, mode):
    res = pd.DataFrame()
    category_files = os.listdir(os.path.join(d_dir, 'QAPairsByCategory'))
    print(category_files)
    for f in category_files:
        category = f.split('_')[1].lower()
        print(category)
        file_path = os.path.join(d_dir, 'QAPairsByCategory', f)
        print(file_path)
        df = pd.read_csv(file_path, sep='|', names = ['img_id', 'question', 'answer'])

        df['mode'] = [mode] * df.shape[0]
        df['category'] = [category] * df.shape[0]
        df.loc[df.answer == 'no', 'category'] = 'binary'
        df.loc[df.answer == 'yes', 'category'] = 'binary'

        res = pd.concat([res,df])
        
    return res

train_df = create_df(train_dir, 'train')
val_df = create_df(val_dir, 'val')

test_df = pd.read_csv(os.path.join(test_dir,'test_questions&answers.txt'), sep='|', names = ['img_id', 'category', 'question', 'answer'])
test_df.loc[test_df.answer == 'no', 'category'] = 'binary'
test_df.loc[test_df.answer == 'yes', 'category'] = 'binary'
test_df['mode'] = ['test'] * test_df.shape[0]

cols = train_df.columns.tolist()
test_df = test_df[cols]

train_df.to_csv(os.path.join(train_dir, 'traindf.csv'), index=False, columns=cols)
val_df.to_csv(os.path.join(val_dir, 'valdf.csv'), index=False, columns=cols)
test_df.to_csv(os.path.join(test_dir, 'testdf.csv'), index=False, columns=cols)
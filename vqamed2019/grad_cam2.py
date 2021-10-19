# grad-cam implementation adapted from this article
# https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82

import argparse
from utils import seed_everything, load_data,encode_text #,Model
import wandb
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms

from models.mmbert import Model, get_transformer_model

from transformers import BertTokenizer
from PIL import Image

import cv2
import timm
import os
import matplotlib.pyplot as plt

#image resize function from https://stackoverflow.com/a/44659589
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pretrain on ROCO with MLM")
    parser.add_argument('--data_dir', type = str, required = False, default = "../ImageClef-2019-VQA-Med", help = "path for data")
    parser.add_argument('--model_dir', type = str, required = False, default = "../ImageClef-2019-VQA-Med/mmbert/MLM/vqa-sentence_transformers-allmpnet48-2.pt", help = "path to load weights")
    
    parser.add_argument('--cnn_encoder', type=str, default='tf_efficientnetv2_m', help='name of the cnn encoder')
    parser.add_argument('--use_relu', action = 'store_true', default = False, help = "use ReLu")
    parser.add_argument('--transformer_model', type=str, default='realformer',choices=['transformer', 'realformer', 'feedback-transformer'], help='name of the transformer model')
    parser.add_argument('--dataset', type=str, default='VQA-Med', help='roco or vqamed2019')
    parser.add_argument('--num_vis', type = int, default=5, help = "num of visual embeddings")
    parser.add_argument('--hidden_size', type=int, default=768, help='embedding size')
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.3, help='dropout')
    parser.add_argument('--n_layers', type=int, default=4, help='num of heads in multihead attenion')
    parser.add_argument('--heads', type=int, default=8, help='num of bertlayers')
    parser.add_argument('--vocab_size', type=int, default=30522, help='vocabulary size')

    parser.add_argument('--task', type=str, default='MLM',
                        choices=['MLM', 'distillation'], help='pretrain task for the model to be trained on')
    
    parser.add_argument('--seed', type = int, required = False, default = 42, help = "set seed for reproducibility")
    parser.add_argument('--img', type=str, default = '../dog_cat.jfif', help='path to img', required = False)
    parser.add_argument('--output', type=str, default = '../grad_cam', help='output img', required = False)

    parser.add_argument('--train_pct', type = float, required = False, default = 1.0, help = "fraction of train samples to select")
    parser.add_argument('--valid_pct', type = float, required = False, default = 1.0, help = "fraction of validation samples to select")
    parser.add_argument('--test_pct', type = float, required = False, default = 1.0, help = "fraction of test samples to select")
    parser.add_argument('--max_position_embeddings', type = int, required = False, default = 28, help = "max length of sequence")
    
    parser.add_argument('--vqa_img', type=str, default = 'synpic371.jpg', help="path to vqa img", required = False)
    parser.add_argument('--category', type=str, default = 'organ', choices=['organ','modality','plane','abnormality','binary'], help="question category", required = False)
    parser.add_argument('--mode', type=str, default = 'Train', choices=['Train', 'Val', 'Test'], help="data split", required = False)
    parser.add_argument('--grad_cam', action='store_false', required = False, default = True,  help='flag to save model input_tensor')
    parser.add_argument('--save_dir', type = str, required = False, default = "./gradcam-images", help = "path to save gradcam images")
    

    args = parser.parse_args()

    seed_everything(args.seed)


    train_df, val_df, test_df = load_data(args)
    df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)

    ans2idx = {ans:idx for idx,ans in enumerate(df['answer'].unique())}
    idx2ans = {idx:ans for ans,idx in ans2idx.items()}
    df['answer'] = df['answer'].map(ans2idx).astype(int)
    train_df = df[df['mode']=='train'].reset_index(drop=True)
    val_df = df[df['mode']=='val'].reset_index(drop=True)
    test_df = df[df['mode']=='test'].reset_index(drop=True)

    num_classes = len(ans2idx)

    args.num_classes = num_classes
    print('numclasses',num_classes)


    img_path = os.path.join(args.data_dir,args.mode,'images',args.vqa_img)
    info_df=df.loc[df['img_id'] == img_path]

    category_df=info_df.loc[info_df['category'] == args.category]
    if category_df['question'].empty:
        raise ValueError('Image does not exist in data split.')
    question = category_df['question'].item()
    answer = category_df['answer'].item()

    model = Model(args)
    model.classifier[2] = nn.Linear(args.hidden_size, num_classes)
    model.load_state_dict(torch.load(args.model_dir,map_location=torch.device('cpu')))
    model.eval()

    img = Image.open(img_path).convert('RGB')
    tfm = transforms.Compose([transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(), 
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img = tfm(img)
    img = img.unsqueeze(0)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokens, segment_ids, input_mask= encode_text(question, tokenizer, args)
    tokens, segment_ids, input_mask = torch.tensor(tokens, dtype = torch.long).unsqueeze(dim=0), torch.tensor(segment_ids, dtype = torch.long).unsqueeze(dim=0), torch.tensor(input_mask, dtype = torch.long).unsqueeze(dim=0)
    logits, _, _ = model(img, tokens, segment_ids, input_mask)

    logits[:, answer].backward()
    gradients = model.transformer.trans.get_activations_gradient()

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = model.transformer.trans.get_activations().detach()

    # weight the channels by corresponding gradients
    for i in range(512):
        activations[:, i, :, :] *= pooled_gradients[i]
        
    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()

    # relu on top of the heatmap
    heatmap = np.maximum(heatmap, 0)

    # normalize the heatmap
    heatmap /= torch.max(heatmap)

    # draw the heatmap
    plt.imsave('test.png', heatmap.squeeze())


    # img = img.squeeze()
    # img = img.numpy()
    # img = img.reshape(img.shape[1],img.shape[2],img.shape[0])
    # img = np.uint8(255 * img)

    img = cv2.imread(img_path)
    print('imgread', img.shape)
    w=224;h=224
    (h_orig, w_orig) = img.shape[:2]
    if h_orig < w_orig:
        img=image_resize(img, height = h)
    else:
        img=image_resize(img, width= w)
    center = (img.shape[0]/2,img.shape[1]/2)
    x = center[1] - w/2
    y = center[0] - h/2
    img = img[int(y):int(y+h), int(x):int(x+w)]
    print('final img shape',img.shape)

    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    cv2.imwrite(os.path.join(args.save_dir,args.category+"_"+args.vqa_img), superimposed_img)

    res = logits.softmax(1).argmax(1).detach()
    print('question: ', question)
    print('answer: ', answer, idx2ans[answer])
    print('preds:', res, idx2ans[res.item()])
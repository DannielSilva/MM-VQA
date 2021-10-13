import argparse
from utils import seed_everything, load_data, LabelSmoothing,encode_text #,Model
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


from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
#from torchvision.models import resnet50

from transformers import BertTokenizer

import cv2
import timm
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pretrain on ROCO with MLM")
    parser.add_argument('--data_dir', type = str, required = False, default = "../ImageClef-2019-VQA-Med", help = "path for data")
    parser.add_argument('--model_dir', type = str, required = False, default = "../ImageClef-2019-VQA-Med/mmbert/MLM/vqa-sentence_transformers-allmpnet48-2.pt", help = "path to load weights")
    parser.add_argument('--method', type=str, default='gradcam',
                            choices=["gradcam","scorecam","gradcam++","ablationcam","xgradcam","eigencam"], help='gradcam method')
    
    parser.add_argument('--cnn_encoder', type=str, default='tf_efficientnetv2_m', help='name of the cnn encoder')
    parser.add_argument('--use_relu', action = 'store_true', default = False, help = "use ReLu")
    parser.add_argument('--transformer_model', type=str, default='realformer',choices=['transformer', 'realformer', 'feedback-transformer'], help='name of the transformer model')
    parser.add_argument('--dataset', type=str, default='VQA-Med', help='roco or vqamed2019')
    parser.add_argument('--num_vis', type = int, default=5, help = "num of visual embeddings")
    parser.add_argument('--hidden_size', type=int, default=768, help='embedding size')
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.3, help='dropout')
    parser.add_argument('--n_layers', type=int, default=4, help='num of heads in multihead attenion')
    parser.add_argument('--heads', type=int, default=12, help='num of bertlayers')
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
    parser.add_argument('--category', type=str, default = 'organ', choices=['organ','modality','plane','abnormality'], help="question category", required = False)
    parser.add_argument('--mode', type=str, default = 'Train', choices=['Train', 'Val', 'Test'], help="data split", required = False)
    parser.add_argument('--grad_cam', action='store_false', required = False, default = True,  help='flag to save model input_tensor')
    

    args = parser.parse_args()


    methods = \
    {"gradcam": GradCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
    }


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
    df=train_df.loc[train_df['img_id'] == img_path]

    category_df=df.loc[df['category'] == args.category]
    question = category_df['question'].item()
    answer = category_df['answer'].item()
    

    model = Model(args)
    model.classifier[2] = nn.Linear(args.hidden_size, num_classes)
    target_layers = model.transformer.mains[3].ff[3]#model.fc1#model.transformer.mains[-3]#model.transformer.trans.model.blocks[6][4]#
    #model = timm.create_model('tf_efficientnetv2_m', pretrained=True)
    #model.classifier = torch.nn.Sequential()
    #target_layers = model.blocks[6][4]
    print( target_layers)
    

    def preprocess_image(img: np.ndarray, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) -> torch.Tensor:
        preprocessing = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        return preprocessing(img.copy()).unsqueeze(0)
    
    rgb_img = cv2.imread(args.img, 1)[:, :, ::-1] #dog_cat.jfif - synpic371.jpg'
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255
    img_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokens, segment_ids, input_mask= encode_text(question, tokenizer, args)
    tokens, segment_ids, input_mask = torch.tensor(tokens, dtype = torch.long).unsqueeze(dim=0), torch.tensor(segment_ids, dtype = torch.long).unsqueeze(dim=0), torch.tensor(input_mask, dtype = torch.long).unsqueeze(dim=0)

    h = model.transformer.prepare_input(img_tensor, tokens, segment_ids, input_mask)

    # out = model(h)


    # Construct the CAM object once, and then re-use it on many images:
    def reshape_transform(tensor, height=16, width=16,channels=3):
        print('tensor shape',tensor.shape)
        result = tensor[:, 5  , :].reshape(tensor.size(0),
        3, height, width)

        # result = tensor[:, 1 : 6  , :].reshape(tensor.size(0),
        # 5*3, height, width)

        #result = tensor.unsqueeze(dim=1).unsqueeze(dim=1)
        
        # result = tensor.reshape(tensor.size(0),
        # channels, height, width)
        return result
    print('Using ' + args.method)
    print('reshape', reshape_transform(torch.rand(2,28,768)).shape)
    cam = methods[args.method](model=model, target_layer=target_layers, use_cuda=False,reshape_transform=reshape_transform)


    # If target_category is None, the highest scoring category
    # will be used for every image in the batch.
    # target_category can also be an integer, or a list of different integers
    # for every image in the batch.
    target_category = answer#None#281

    #import IPython; IPython.embed(); import sys; sys.exit(0)
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=h, target_category=target_category)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=False)

    print('Writing output file to ',args.output)
    cv2.imwrite(args.output + args.model_dir.split('/')[-1] + ".jpg", visualization)
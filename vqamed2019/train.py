#import cv2
import argparse
from utils import seed_everything, VQAMed, train_one_epoch, validate, test, load_data, LabelSmoothing, train_img_only, val_img_only, test_img_only, LabelSmoothByCategory #,Model
import wandb
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms, models
from torch.cuda.amp import GradScaler
import os
import warnings
#import albumentations as A
#import pretrainedmodels
#from albumentations.core.composition import OneOf
#from albumentations.pytorch.transforms import ToTensorV2

from models.mmbert import Model
from models.asl_singlelabel import ASLSingleLabel

warnings.simplefilter("ignore", UserWarning)



if __name__ == '__main__':
    __spec__ = None
    parser = argparse.ArgumentParser(description = "Finetune on ImageClef 2019")

    parser.add_argument('--run_name', type = str, required = True, help = "run name for wandb")
    parser.add_argument('--data_dir', type = str, required = False, default = "ImageClef-2019-VQA-Med", help = "path for data")
    parser.add_argument('--model_dir', type = str, required = False, default = "MMBERT/pretrain/val_loss_3.pt", help = "path to load weights")
    parser.add_argument('--resume_dir', type = str, required = False, default = "ImageClef-2019-VQA-Med/mmbert/MLM/model.pt", help = "path to load weights")
    parser.add_argument('--save_dir', type = str, required = False, default = "ImageClef-2019-VQA-Med/mmbert", help = "path to save weights")
    parser.add_argument('--category', type = str, required = False, default = None,  help = "choose specific category if you want")
    parser.add_argument('--use_pretrained', action = 'store_true', default = False, help = "use pretrained weights or not")
    parser.add_argument('--resume_training', action = 'store_true', default = False, help = "resume")
    parser.add_argument('--mixed_precision', action = 'store_true', default = False, help = "use mixed precision or not")
    parser.add_argument('--clip', action = 'store_true', default = False, help = "clip the gradients or not")

    parser.add_argument('--seed', type = int, required = False, default = 42, help = "set seed for reproducibility")
    parser.add_argument('--num_workers', type = int, required = False, default = 4, help = "number of workers")
    parser.add_argument('--epochs', type = int, required = False, default = 100, help = "num epochs to train")
    parser.add_argument('--train_pct', type = float, required = False, default = 1.0, help = "fraction of train samples to select")
    parser.add_argument('--valid_pct', type = float, required = False, default = 1.0, help = "fraction of validation samples to select")
    parser.add_argument('--test_pct', type = float, required = False, default = 1.0, help = "fraction of test samples to select")

    parser.add_argument('--max_position_embeddings', type = int, required = False, default = 28, help = "max length of sequence")
    parser.add_argument('--batch_size', type = int, required = False, default = 16, help = "batch size")
    parser.add_argument('--lr', type = float, required = False, default = 1e-4, help = "learning rate'")
    # parser.add_argument('--weight_decay', type = float, required = False, default = 1e-2, help = " weight decay for gradients")
    parser.add_argument('--factor', type = float, required = False, default = 0.1, help = "factor for rlp")
    parser.add_argument('--patience', type = int, required = False, default = 10, help = "patience for rlp")
    parser.add_argument('--counter', type = int, required = False, default = 20, help = "patience for remaining training")
    # parser.add_argument('--lr_min', type = float, required = False, default = 1e-6, help = "minimum lr for Cosine Annealing")
    parser.add_argument('--hidden_dropout_prob', type = float, required = False, default = 0.3, help = "hidden dropout probability")
    parser.add_argument('--smoothing', type = float, required = False, default = None, help = "label smoothing")

    parser.add_argument('--image_size', type = int, required = False, default = 224, help = "image size")
    parser.add_argument('--hidden_size', type = int, required = False, default = 312, help = "hidden size")
    parser.add_argument('--vocab_size', type = int, required = False, default = 30522, help = "vocab size")
    parser.add_argument('--type_vocab_size', type = int, required = False, default = 2, help = "type vocab size")
    parser.add_argument('--heads', type = int, required = False, default = 12, help = "heads")
    parser.add_argument('--n_layers', type = int, required = False, default = 4, help = "num of layers")
    parser.add_argument('--num_vis', type = int, required = True, help = "num of visual embeddings")
    
    parser.add_argument('--wandb', action = 'store_false', default = True, help = "record in wandb or not")
    parser.add_argument('--save_model_epoch', type = int, required = False, default = 4, help = "periodically save the model")

    parser.add_argument('--task', type=str, default='MLM',
                        choices=['MLM', 'distillation'], help='task which the model was pre-trained on')
    parser.add_argument('--clinicalbert', type=str, default='emilyalsentzer/Bio_ClinicalBERT')
    parser.add_argument('--dataset', type=str, default='VQA-Med', help='roco or vqamed2019')
    parser.add_argument('--cnn_encoder', type=str, default='resnet152', help='name of the cnn encoder')
    parser.add_argument('--use_relu', action = 'store_true', default = False, help = "use ReLu")
    parser.add_argument('--transformer_model', type=str, default='transformer',choices=['transformer', 'realformer', 'feedback-transformer'], help='name of the transformer model')
    parser.add_argument('--loss', type=str, default='CrossEntropyLoss', choices=['CrossEntropyLoss', 'ASLSingleLabel'], help='loss to evaluate model on')

    args = parser.parse_args()
    print('Using wandb',args.wandb)
    if args.wandb:
        wandb.init(project='medvqa', name = args.run_name, config = args) #settings=wandb.Settings(start_method='fork'),

    seed_everything(args.seed)


    train_df, val_df, test_df = load_data(args)

    if args.category:
            
        train_df = train_df[train_df['category']==args.category].reset_index(drop=True)
        val_df = val_df[val_df['category']==args.category].reset_index(drop=True)
        test_df = test_df[test_df['category']==args.category].reset_index(drop=True)

        train_df = train_df[~train_df['answer'].isin(['yes', 'no'])].reset_index(drop = True)
        val_df = val_df[~val_df['answer'].isin(['yes', 'no'])].reset_index(drop = True)
        test_df = test_df[~test_df['answer'].isin(['yes', 'no'])].reset_index(drop = True)

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

    

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if torch.cuda.is_available():
        print('cleaning cuda cache')
        torch.cuda.empty_cache()

    model = Model(args)

    if args.use_pretrained:
        print('loading model from roco')
        print(args.model_dir)
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args.model_dir)
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        model.load_state_dict(model_dict)
        # change classifier final layer for the current problem
        model.classifier[2] = nn.Linear(args.hidden_size, num_classes)

    if args.resume_training:
        print('resume training', args.resume_dir)
        model.classifier[2] = nn.Linear(args.hidden_size, num_classes)
        # before = args.run_name.split('-')[-1]
        # before = args.run_name.split(('-'))[0] + '-' + str(int(before)-1) +"_acc.pt"
        # path = os.path.join(args.save_dir, before)
        model.load_state_dict(torch.load(args.resume_dir))

    if not args.use_pretrained and not args.resume_training:
        print('from scratch')
        model.classifier[2] = nn.Linear(args.hidden_size, num_classes)
    


        
    model.to(device)

    if args.wandb:
        wandb.watch(model, log='all')


    optimizer = optim.Adam(model.parameters(),lr=args.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience = args.patience, factor = args.factor, verbose = True)


    if args.smoothing:
        print('Using label smoothing')
        #criterion = LabelSmoothing(smoothing=args.smoothing)
        criterion = LabelSmoothByCategory(train_df=train_df,num_classes=args.num_classes,device=device)
        print(criterion.plane_tensor)
    elif args.loss == 'CrossEntropyLoss':
        print('Using CrossEntropyLoss')
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'ASLSingleLabel':
        print('Using ASLSingleLabel')
        criterion = ASLSingleLabel()

    scaler = GradScaler()


    train_tfm = transforms.Compose([
                                    
                                    #transforms.ToPILImage(),
                                    #transforms.Resize([224,224]),
                                    transforms.Resize(224), 
                                    transforms.CenterCrop(224), 
                                    transforms.RandomResizedCrop(224,scale=(0.75,1.25),ratio=(0.75,1.25)),
                                    transforms.RandomRotation(10),
                                    # Cutout(),
                                    transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4,hue=0.4),
                                    transforms.ToTensor(), 
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    val_tfm = transforms.Compose([#transforms.ToPILImage(),
                                #transforms.Resize([224,224]),
                                transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(), 
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    test_tfm = transforms.Compose([#transforms.ToPILImage(),
                                #transforms.Resize([224,224]),
                                transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(), 
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])




    traindataset = VQAMed(train_df, imgsize = args.image_size, tfm = train_tfm, args = args, mode='train')
    valdataset = VQAMed(val_df, imgsize = args.image_size, tfm = val_tfm, args = args, mode='eval')
    testdataset = VQAMed(test_df, imgsize = args.image_size, tfm = test_tfm, args = args, mode='test')

    trainloader = DataLoader(traindataset, batch_size = args.batch_size, shuffle=True, num_workers = args.num_workers)
    valloader = DataLoader(valdataset, batch_size = args.batch_size, shuffle=False, num_workers = args.num_workers)
    testloader = DataLoader(testdataset, batch_size = args.batch_size, shuffle=False, num_workers = args.num_workers)

    best_acc1 = 0
    best_acc2 = 0
    best_loss = np.inf
    counter = 0


    for epoch in range(args.epochs):

        print(f'Epoch {epoch+1}/{args.epochs}')


        train_loss, _, _, _, _ = train_one_epoch(trainloader, model, optimizer, criterion, device, scaler, args, idx2ans)
        val_loss, predictions, val_acc, val_bleu = validate(valloader, model, criterion, device, scaler, args, val_df,idx2ans)
        test_loss, predictions, acc, bleu = test(testloader, model, criterion, device, scaler, args, test_df,idx2ans)

        scheduler.step(val_loss)
     

        if not args.category:

            log_dict = acc
            
            for k,v in bleu.items():
                log_dict[k] = v

            if args.wandb:
                log_dict['train_loss'] = train_loss
                log_dict['val_loss'] = val_loss
                log_dict['test_loss'] = test_loss
                log_dict['learning_rate'] = optimizer.param_groups[0]["lr"]
                log_dict['val_total_acc'] = val_acc['val_total_acc']

                wandb.log(log_dict)

        else:

            if args.wandb:
                wandb.log({'train_loss': train_loss,
                        'val_loss': val_loss,
                        'test_loss': test_loss,
                        'learning_rate': optimizer.param_groups[0]["lr"],
                        f'val_{args.category}_acc': val_acc,
                        f'val_{args.category}_bleu': val_bleu,
                        f'{args.category}_acc': acc,
                        f'{args.category}_bleu': bleu}) 

        #save by val loss
        if val_loss < best_loss:
            print('Saving model by loss')
            torch.save(model.state_dict(), os.path.join(args.save_dir, args.task , args.run_name + "_loss" + '.pt'))
            best_loss = val_loss

        #save by accuracy in val
        if not args.category:

            if val_acc['val_total_acc'] > best_acc1:
                print('Saving model')
                torch.save(model.state_dict(), os.path.join(args.save_dir, args.task , args.run_name + '.pt'))
                best_acc1=val_acc['val_total_acc']

        else:

            if val_acc['val_' + args.category + '_acc'] > best_acc1:
                print('Saving model')
                torch.save(model.state_dict(), os.path.join(args.save_dir, args.task , args.run_name + '.pt'))
                best_acc1 = val_acc['val_' + args.category + '_acc'] 

        # if (epoch + 1) % args.save_model_epoch == 0:
        #     torch.save(model.state_dict(),os.path.join(args.save_dir, f'{args.run_name}_acc_epoch_{epoch}.pt'))

        if best_acc1 > best_acc2:
            counter = 0
            best_acc2 = best_acc1
        else:
            counter+=1
            print(f'Counter {counter}/{args.counter}')
            if counter > args.counter:
                print('Counter expired, finishing.')
                break      
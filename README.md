# ECIR Submission - MMBERT Reproducibiliy Study

## Extending the MMBERT Model for Medical Visual Question Answering

Abstract: *Models for Visual Question Answering (VQA) on medical images should answer diagnostically relevant natural language questions with basis on visual contents. A recent study in the area proposed MMBERT, a multi-modal encoder model that combines a ResNet backbone to represent images at multiple resolutions, together with a Transformer encoder. By pre-training the model over the Radiology Objects in COntext (ROCO) dataset of images+captions, the authors achieved state-of-the-art performance on the VQA-Med dataset of questions over radiology images, used in ImageCLEF 2019. Taking the source code provided by the authors, we first attempted to reproduce the results for MMBERT, afterwards extending the model in several directions: (a) using a stronger image encoder based on EfficientNetV2, (b) using a multi-modal encoder based on the RealFormer architecture, (c) extending the pre-training task with a contrastive objective, and (d) using a novel loss function for fine-tuning the model to the VQA task, that specifically considers class imbalance. Exactly reproducing the results published for MMBERT was met with some difficulties, and the default hyper-parameters given in the original source code resulted in a lower performance. Our experiments showed that aspects such as the size of the training batches can significantly affect the performance. Moreover, starting from baseline results corresponding to our reproduction of MMBERT, we also show that the proposed extensions can lead to improvements.*

## Model pre-training on the ROCO dataset

Model pre-training can be done in two settings, with Masked Language Modeling objective in [pretrain/roco_train.py](https://github.com/DannielSilva/MMBERT/blob/main/pretrain/roco_train.py) or Masked Language Modeling + Contrastive Learning in [pretrain/roco_supcon_train.py](https://github.com/DannielSilva/MMBERT/blob/main/pretrain/roco_supcon_train.py).

Example showing how to do model pre-training on ROCO, with the supervised contrastive loss leveraging sentence-bert similarity scores.
```
python pretrain/roco_supcon_train.py -r='contrastive_roco_run_name' --con_task='supcon' --similarity='sentence_transformers' --num_vis=5 --save_dir='save_dir' --cnn_encoder='tf_efficientnetv2_m' --transformer_model='realformer' --data_dir='roco_dir'  --num_workers=16 --batch_size=16 --mlm_prob=0.15 --task='MLM'
```

Example showing how to do model pre-training on ROCO, only with the MLM objective
```
python -u pretrain/roco_train.py -r='mlm-only_roco_run_name' --num_vis=5 --save_dir='save_dir' --cnn_encoder='tf_efficientnetv2_m' --transformer_model='realformer' --data_dir='roco_dir'  --num_workers=16 --batch_size=16 --mlm_prob=0.15 --task='MLM'
```

## Model training and evaluation on the VQA-Med 2019 dataset

Example showing how to do model training with the EfficientNetV2+RealFormer encoder.
```
python vqamed2019/train.py --run_name='vqa_run_name' --cnn_encoder='tf_efficientnetv2_m' --transformer_model='realformer' --data_dir="ImageClef-2019-VQA-Med_dir" --use_pretrained --model_dir='path_to_pretrained_model' --batch_size=16 --num_vis=5 --hidden_size=768 --num_workers=16 --save_dir="../ImageClef-2019-VQA-Med/mmbert" --loss='ASLSingleLabel' --epochs=100
```

Example showing how to do model evaluation.
```
python vqamed2019/eval.py --run_name='eval-model-name' --num_vis=5 --model_dir='model_dir' --transformer='realformer' --heads=8 --cnn_encoder='tf_efficientnetv2_m'
```

## Command line arguments

| Parameter                 | Default       | Training/Testing       | Description   |	
| :------------------------ |:-------------:|:----------------------:| :-------------|
| --run_name        	      |	              | both                   |  run name here for [wandb](https://wandb.ai) analysis
| --lr              	      |	2e-5  / 1e-4         | pre-train/fine-tuning              | learning rate
| --batch_size     		      |   16     | training                 | batch size 
| --epochs     		      |   100     | fine-tuning                 | number of epochs 
| --counter     		          |   20     | fine-tuning                 | number of epochs to wait for early stop
| --use_pretrained        	          |      | fine-tuning                      | flag to load model in fine-tuning and testing
| --mlm_prob        	      |  0.15  | pre-train                     | prob for MLM objective
| --model_dir        	      |   	          | fine-tuning and testing                     | path to an already saved model
| --data_dir        	      |   	          | both                     | path to dataset (ROCO or VQA-MED ImageCLEF2019)
| --save_dir        	      |   	          | both                     | path to save model
| --con_task                | ```supcon``` | pre-train                | contrastive learn task (```simclr``` or ```supcon```)
| --similarity                | ```jaccard_similarity```        | pre-train                | similarity measure between captions for SupCon (```jaccard```,```sentence_transformers```)
| --num_vis        		      |  5  | both                     | number of visual tokens 
| --hidden_size        		  | 768   | both                     | dimensionality for the transformer/realformer hidden states 
| --transformer_model       |  ```transformer```  | both                     | Transformer or RealFormer architecture
| --cnn_encoder             |   ```resnet152```	 | both                     | ResNet152 (```resnet152```) or EfficientNetV2 (```tf_efficientnetv2_m```)
| --use_relu             |   ```False```	 | both                     | flag if set replaces SERF acivation function with ReLU
| --loss             |   ```CrossEntropyLoss```	 | fine-tuning                     | Cross Entropy loss (```CrossEntropyLoss```) or Asymmetric Loss (```ASLSingleLabel```)

<!--| --category      		      |    	          | both                   | category of questions to consider -->
<!--| --mixed_precision         |               | both                   | use mixed-precision operations -->


## Datasets and trained models

1) The Radiology Objects in COntext (ROCO) dataset: https://www.kaggle.com/virajbagal/roco-dataset
    
    a) Download the already processed vocabulary file with medical keywords for the MLM objective [med_vocab.pkl](https://drive.google.com/file/d/1Crd6cYfurb82FOFBcTcehFpmidOfHGfl/view?usp=sharing) - code used in preprocess/roco_data.py

    b) Replace the file traindata.csv in roco/train/radiology with the following one, in order to consider back-translation also for SupCon: [traindata.csv](https://drive.google.com/file/d/1hXcIzB56Re7xCKjAOQ_bB8pgeu_BLiuh/view?usp=sharing) - code used in preprocess/translate_transformers.py
 
2) The VQA-Med 2019 dataset: https://github.com/abachaa/VQA-Med-2019

3) Pretrained models are available here: 

  <!--  a) [Model](https://drive.google.com/file/d/1lqWkLqTv9AdLg1hlDzT77I3wj7rfA0W1/view?usp=sharing) pre-trained with supervised contrastive loss leveraging sentence-bert similarity scores + batch 48 + patience 80 - achieves 62.80% accuracy and 64.32% BLEU.

    b) ... -->

| Image Encoder  | Architecture | Activation | Loss | Pretraining task | Accuracy | BLEU | Link |
| :------------------------ |:-------------:|:----------------------:| :-------------:|  :-------------:| :-------------:| :-------------:| :-------------|
|ResNet152 | Transformer | ReLU | CE | MLM | 58.80 | 60.74 | [Here](https://drive.google.com/file/d/1FMLh8LJICTVcHkKKNUfeWTkAY90HetZ7/view?usp=sharing) | |
|Effic.NetV2      | Transformer  | ReLU                | CE | MLM        | 59.40    | 61.36 | [Here](https://drive.google.com/file/d/1v9XK1Bw3QrJvHlUUOpWok8weCOxL2ELv/view?usp=sharing) | |
|Effic.NetV2      | RealFormer  | ReLU                | CE  | MLM         | 59.20    | 61.52 | [Here](https://drive.google.com/file/d/1AOSlTy7LVid7OCUQ5mpuSYcvMJI2BnG1/view?usp=sharing) | |
|Effic.NetV2      | RealFormer  | SERF                | CE   | MLM      | 60.00   | 62.39  | [Here](https://drive.google.com/file/d/1GBXytRhaljDYZytRz8A1l2vn_hkpoDTP/view?usp=sharing) | |
|Effic.NetV2 | RealFormer   | SERF            | ASL     | MLM         | 59.80    | 61.55 | [Here](https://drive.google.com/file/d/1UtRw8ox0HY36JCu4JRnDL8Wu4_QWM6rh/view?usp=sharing) | |
|Effic.NetV2 | RealFormer   | SERF      | ASL   | MLM + SimCLR       | 59.80    | 61.50 | [Here](https://drive.google.com/file/d/1S7iIe-iEn0l14zRmkCNn7_MiKv5BbpF3/view?usp=sharing) | |
|Effic.NetV2 | RealFormer   | SERF       | ASL  | MLM + SupCon-J      | 60.20    | 62.50 | [Here](https://drive.google.com/file/d/1V8LUYB66gPihbIVeXlU1Nee47ON29gfF/view?usp=sharing) | |
|Effic.NetV2 | RealFormer   | SERF      | ASL | MLM + SupCon-SB      | 60.60    | 62.98 | [Here](https://drive.google.com/file/d/15ldq2Gn-EyoJUj3gO8SiMgYhI2aZ_oeG/view?usp=sharing) | |
|Effic.NetV2 | RealFormer   | SERF      | ASL  | MLM + SupCon-SB            | 61.60†    | 63.72† | [Here](https://drive.google.com/file/d/15STLuQ4cwcNiPIb2VilP4hvQiPn5Az9f/view?usp=sharing) | |
|Effic.NetV2 | RealFormer   | SERF           | ASL | MLM + SupCon-SB            | 62.80†*    | 64.32†* | [Here](https://drive.google.com/file/d/1WerXfF5ve9T9Bt309Fal5QHeaV7jpSq_/view?usp=sharing) | |

Notation: † represents a model where the batch size was set to 48 (vs 16 in the rest), and * represents a model where the patience was set to 80.
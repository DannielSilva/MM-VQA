# ECIR Submission - MMBERT Reproducibiliy Study

## Extending the MMBERT Model for Medical Visual Question Answering

Abstract: *Models for Visual Question Answering (VQA) on medical images should answer diagnostically relevant natural language questions with basis on visual contents. A recent study in the area proposed MMBERT, a multi-modal encoder model that combines a ResNet backbone to represent images at multiple resolutions, together with a Transformer encoder. By pre-training the model over the Radiology Objects in COntext (ROCO) dataset of images+captions, the authors achieved state-of-the-art performance on the VQA-Med dataset of questions over radiology images, used in ImageCLEF 2019. Taking the source code provided by the authors, we first attempted to reproduce the results for MMBERT, afterwards extending the model in several directions: (a) using a stronger image encoder based on EfficientNetV2, (b) using a multi-modal encoder based on the RealFormer architecture, (c) extending the pre-training task with a contrastive objective, and (d) using a novel loss function for fine-tuning the model to the VQA task, that specifically considers class imbalance. Exactly reproducing the results published for MMBERT was met with some difficulties, and the default hyper-parameters given in the original source code resulted in a lower performance. Our experiments showed that aspects such as the size of the training batches can significantly affect the performance. Moreover, starting from baseline results corresponding to our reproduction of MMBERT, we also show that the proposed extensions can lead to improvements.*

## Train and evaluate on the VQA-Med 2019 dataset

```
python vqamed2019/train.py --run_name='vqa_run_name' --cnn_encoder='tf_efficientnetv2_m' --transformer_model='realformer' --data_dir="ImageClef-2019-VQA-Med_dir" --use_pretrained --model_dir='path_to_pretrained_model' --batch_size=16 --num_vis=5 --hidden_size=768 --num_workers=16 --task='MLM' --save_dir="../ImageClef-2019-VQA-Med/mmbert" --loss='ASLSingleLabel' --epochs=100
```

```
python vqamed2019/eval.py --run_name='eval-model-name' --num_vis=5 --model_dir='model_dir' --transformer='realformer' --heads=8 --cnn_encoder='tf_efficientnetv2_m'
```



Command line arguments: 

| Parameter                 | Default       | Training/Testing       | Description   |	
| :------------------------ |:-------------:|:----------------------:| :-------------|
| --run_name        	      |	              | both                   | 
| --lr              	      |	 	            | training               | learning rate
| --batch_size     		      |    	          | training                 | batch size 
| --num_vis        		      |   	          | both                     | number of visual tokens 
| --hidden_size        		  |               | both                     | dimensionality for the transformer/realformer hidden states 
| --use_pretrained        	|   	          | both                     | flag to load model in fine-tuning and testing
| --con_task                |   	          | pre-train                | contrastive learn task (```simclr``` or ```supcon```)
| --similarity              |   	          | pre-train                | similarity measure between captions for supcon
| --transformer_model       |   	          | both                     | transformer or realformer architecture
| --cnn_encoder             |   	          | both                     | ResNet152 or EfficientNetV2

<!--| --category      		      |    	          | both                   | category of questions to consider -->
<!--| --mixed_precision         |               | both                   | use mixed-precision operations -->
## Pre-train on the ROCO dataset

Example showing how to do model pre-training on ROCO, with the supervised contrastive loss leveraging sentence-bert similarity scores.
```
python pretrain/roco_supcon_train.py -r='contrastive_roco_run_name' --con_task='supcon' --similarity='sentence_transformers' --num_vis=5 --save_dir='save_dir' --cnn_encoder='tf_efficientnetv2_m' --transformer_model='realformer' --data_dir='roco_dir'  --num_workers=16 --batch_size=16 --mlm_prob=0.15 --task='MLM'
```

Example showing how to do model pre-training on ROCO, only with the MLM objective

```
python -u pretrain/roco_train.py -r='mlm-only_roco_run_name' --num_vis=5 --save_dir='save_dir' --cnn_encoder='tf_efficientnetv2_m' --transformer_model='realformer' --data_dir='roco_dir'  --num_workers=16 --batch_size=16 --mlm_prob=0.15 --task='MLM'
```

## Datasets and trained models

1) The Radiology Objects in COntext (ROCO) dataset: https://www.kaggle.com/virajbagal/roco-dataset
    
    a) Download the already processed vocabulary file with medical keywords for the MLM objective [med_vocab.pkl](https://drive.google.com/file/d/1Crd6cYfurb82FOFBcTcehFpmidOfHGfl/view?usp=sharing) - code used in preprocess/roco_data.py

    b) Replace the traindata.csv in roco/train/radiology with the following with backtranslation also for SupCon: [traindata.csv](https://drive.google.com/file/d/1hXcIzB56Re7xCKjAOQ_bB8pgeu_BLiuh/view?usp=sharing) - code used in preprocess/translate_transformers.py
 
2) The VQA-Med 2019 dataset: https://github.com/abachaa/VQA-Med-2019

3) Pretrained models are available here: 

    a) Model pre-trained with supervised contrastive loss with sentence-bert similarity + batch 48 + patience 80 [here](https://drive.google.com/file/d/1lqWkLqTv9AdLg1hlDzT77I3wj7rfA0W1/view?usp=sharing) - achieving 62.80% acc. 64.32% BLEU.

    b) ...

# ECIR Submission - MMBERT Reproducibiliy Study

## Extending the MMBERT Model for Medical Visual Question Answering

Abstract: *Models for Visual Question Answering (VQA) on medical images should answer diagnostically relevant natural language questions with basis on visual contents. A recent study in the area proposed MMBERT, a multi-modal encoder model that combines a ResNet backbone to represent images at multiple resolutions, together with a Transformer encoder. By pre-training the model over the Radiology Objects in COntext (ROCO) dataset of images+captions, the authors achieved state-of-the-art performance on the VQA-Med dataset of questions over radiology images, used in ImageCLEF 2019. Taking the source code provided by the authors, we first attempted to reproduce the results for MMBERT, afterwards extending the model in several directions: (a) using a stronger image encoder based on EfficientNetV2, (b) using a multi-modal encoder based on the RealFormer architecture, (c) extending the pre-training task with a contrastive objective, and (d) using a novel loss function for fine-tuning the model to the VQA task, that specifically considers class imbalance. Exactly reproducing the results published for MMBERT was met with some difficulties, and the default hyper-parameters given in the original source code resulted in a lower performance. Our experiments showed that aspects such as the size of the training batches can significantly affect the performance. Moreover, starting from baseline results corresponding to our reproduction of MMBERT, we also show that the proposed extensions can lead to improvements.*

## Train and evaluate on the VQA-Med 2019 dataset

```
python train.py --run_name give_name --mixed_precision --lr set_lr --category cat_name --batch_size 16 --num_vis set_visual_feats --hidden_size hidden_dim_size
```

```
python eval.py --run_name give_name --mixed_precision --category cat_name --hidden_size hidden_dim_size --use_pretrained
```

Command line arguments: 

| Parameter                 | Default       | Training/Testing       | Description   |	
| :------------------------ |:-------------:|:----------------------:| :-------------|
| --run_name        	      |	              | both                   | 
| --mixed_precision         |               | both                   | use mixed-precision operations
| --lr              	      |	 	            | training               | learning rate
| --category      		      |    	          | both                   | category of questions to consider
| --batch_size     		      |    	          | training               | batch size 
| --num_vis        		      |   	          | training               | number of visual tokens 
| --hidden_size        		  |   	          | both                   | dimensionality for the transformer/realformer hidden states 
| --use_pretrained        	|   	          | testing                |  

## Pre-train on the ROCO dataset

```
python ...
```

## Datasets and trained models

1) The Radiology Objects in COntext (ROCO) dataset: https://github.com/razorx89/roco-dataset 

2) The VQA-Med 2019 dataset: https://github.com/abachaa/VQA-Med-2019

3) Pretrained models are available here: ...  

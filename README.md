# Embedding Disentamglement for Multimodal Learning

## What is it? 

This repository explores embedding disentanglement for multimodal learning in the fashion domain. We use CLIP (Contrastive Language-Image Pretraining) models to perform fashion retrieval, matching images of dresses with their textual descriptions. The project includes methods for finetuning CLIP, enforcing disentanglement between content and subjective information in embeddings, and evaluating retrieval performance on the FEIDEGGER dataset.


## The Dataset

The FEIDEGGER dataset consists of 8732 high-resolution images of dresses, each with 5 textual annotations in German. For more information on the dataset, please check out the [dataset's repository](https://github.com/laumonfe/feidegger/tree/master) and the [LREC 2018 paper](http://aclweb.org/anthology/L18-1070). 


## Requirements

Install dependencies:
```
python install_dependencies.py
```
Or with a Conda environment:

```
python install_dependencies.py --conda --env-name feidegger-mamba
```

## Download the dataset 
```
python feidegger_mamba_prep.py --data_path data/FEIDEGGER_release_1.2.json --output_dir mamba_dataset
```



## Create the embeddings 
You can either use the precomputed embeddings, or you can compute them yourself using... 

## Finetuning the models

You can finetune CLIP as is, or you can Finetune it to enforce disentanglement 


## Visualizing the outputs


1. retrival visualization 
2. fiftyone 
3. clustering 

## Metric calculation 
Evaluation using the retrieval 



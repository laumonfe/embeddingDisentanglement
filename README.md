# Embedding Disentamglement for Multimodal Learning

## :mag: What is it? 

This repository explores embedding disentanglement for multimodal learning in the fashion domain. We use CLIP (Contrastive Language-Image Pretraining) models to perform fashion retrieval, matching images of dresses with their textual descriptions. The project includes methods for finetuning CLIP, enforcing disentanglement between content and subjective information in embeddings, and evaluating retrieval performance on the FEIDEGGER dataset.


## :dress: The Dataset

The FEIDEGGER dataset consists of 8732 high-resolution images of dresses, each with 5 textual annotations in German. For more information on the dataset, please check out the [dataset's repository](https://github.com/laumonfe/feidegger/tree/master) and the [LREC 2018 paper](http://aclweb.org/anthology/L18-1070). 


## :package: Requirements

Install dependencies in your environment:

```
pip install -r requirements.txt
```

## :arrow_down: Download the dataset 
You can download the FEIDEGGER dataset using: 
```
python dataset/feidegger_crawler.py --data_path dataset/FEIDEGGER_release_1.2.json --output_dir [path/to/dataset]
```

## :rocket: Finetuning the models

You can finetune CLIP as is, or you can finetune it to enforce disentanglement 

## :gear: Create the embeddings 
You can either use the precomputed embeddings, or you can compute them yourself using ´compute_embeddings.py´


**What does it do?**

- Loads a CSV file containing image paths and text descriptions.
- Loads the specified model (pretrained, finetuned, or disentangled).
- Computes embeddings for each image and text pair.
- Saves the embeddings as `.npy` files for downstream tasks (retrieval, visualization, etc.).

**Usage**

Run from the command line:

```
python compute_embeddings.py --model_kind [pretrained|finetuned|disentangled] --pretrained_dir [path/to/models] --csv_path [path/to/csv]
```

- `--model_kind`: Choose which model to use:
  - `pretrained`: Baseline CLIP models.
  - `finetuned`: CLIP models finetuned on FEIDEGGER.
  - `disentangled`: CLIP models finetuned with disentanglement loss.
- `--pretrained_dir`: (Optional) Path to the directory containing your pretrained or finetuned models.
- `--csv_path`: Path to the CSV file containing image paths and text descriptions.

**Output**

- Image embeddings:  
  `data/embeddings/[model_kind]_clip-ViT-B-32-multilingual-v1/image_embeddings_clip-ViT-B-32_[model_kind].npy`
- Text embeddings:  
  `data/embeddings/[model_kind]_clip-ViT-B-32-multilingual-v1/text_embeddings_clip-ViT-B-32-multilingual-v1_[model_kind].npy`



## :art: Visualizing the outputs

After calculating the embeddings, you can either qualitative asses retrival or explore how the latent space looks using the scripts inside visualizations. 

## :bar_chart: Metric calculation 
For a quantitative evaluation of the models, youi can run the ´retrieval_evaluation.py´

This script evaluates retrieval performance for the FEIDEGGER dataset using pre-computed image and text embeddings.

**What does it do?**

- Loads image and text embeddings for a specified model (pretrained, finetuned, or disentangled).
- Loads metadata from a CSV file.
- Computes retrieval metrics: Recall@K, Precision@K, and mean ground truth rank for text-to-image retrieval.
- Saves per-query results (including recall, precision, and rank) to a CSV file.

**Usage**

Run from the command line:

```
python retrieval.py --model_kind [pretrained|finetuned|disentangled] --csv_path [path/to/csv]
```

- `--model_kind`: Choose which model's embeddings to evaluate.
- `--csv_path`: Path to the CSV file containing image paths and text descriptions.

**Example:**

```
python retrieval.py --model_kind finetuned --csv_path data/embeddings/feidegger_visualization_data.csv
```

**Output**

- Prints mean Recall@K, Precision@K, and mean ground truth rank for each K.
- Saves a CSV file with per-query results:
  - `retrieval_results_[model_kind].csv`

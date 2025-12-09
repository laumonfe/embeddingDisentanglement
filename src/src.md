# Source Code (`src/`) Overview

This folder contains the core source code for the FEIDEGGER multimodal learning and retrieval project.

## Structure

- **models/**  
  Model classes and wrappers for text and vision encoders (e.g., CLIP, DistilBERT).

- **losses/**  
  Custom loss functions, including disentanglement and contrastive losses.

- **data_loader.py**  
  Dataset and DataLoader utilities for loading and batching data.

- **utils.py**  
  General utility functions for configuration, weights, and model loading.

- **compute_embeddings.py**  
  Scripts for generating and saving image/text embeddings.

- **clip_finetuned.py**  
  Finetuning CLIP models on the FEIDEGGER dataset.

- **clip_disentangled_loss2.py**  
  Training with disentanglement loss to separate content and subjective information.

- **retrieval.py**  
  Evaluation scripts for retrieval metrics (Recall@K, Precision@K, etc.).

## Usage

- Import modules from `src/` in your training, evaluation, or visualization scripts.
- Run scripts directly for tasks like finetuning, embedding computation, or metric evaluation.

## How to Train

To train a model on the FEIDEGGER dataset:

1. **Prepare the dataset and embeddings**  
   Make sure your data and embeddings are ready (see main README).

2. **Finetune CLIP**  
   Run:
   ```
   python src/clip_finetuned.py
   ```
   This will finetune CLIP on the FEIDEGGER dataset.

3. **Train with Disentanglement Loss**  
   Run:
   ```
   python src/clip_disentangled_loss2.py
   ```
   This will train the model with disentanglement loss to separate content and subjective information.

4. **Monitor Training**  
   Use TensorBoard or logs to monitor loss and metrics.

## Notes

- Ensure all dependencies in `requirements.txt` are installed.
- See the main project README for setup and workflow instructions.

---
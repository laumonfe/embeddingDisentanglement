# FEIDEGGER Dataset Preparation for Mamba Networks

This script downloads the complete FEIDEGGER dataset and prepares it for training with Mamba network architectures. Mamba networks are efficient state space models for sequence modeling, requiring properly formatted sequential data.

## Features

- Downloads all FEIDEGGER dataset images
- Processes images to consistent size and format
- Tokenizes text descriptions using transformer models
- Creates sequence pairs of images and descriptions
- Splits data into training, validation, and test sets
- Saves everything in HDF5 format for efficient loading
- Provides PyTorch Dataset class for easy integration

## Requirements

Install the necessary dependencies:

```bash
pip install numpy pandas tqdm pillow requests torch torchvision transformers h5py huggingface_hub[hf_xet]
```

The `huggingface_hub[hf_xet]` package is optional but recommended for faster downloads from Hugging Face Hub using Xet Storage technology.

## Usage

### Basic Usage

```bash
python feidegger_mamba_prep.py --data_path data/FEIDEGGER_release_1.2.json --output_dir mamba_dataset
```

### Advanced Options

```bash
python feidegger_mamba_prep.py \
    --data_path data/FEIDEGGER_release_1.2.json \
    --output_dir mamba_dataset \
    --image_size 224 224 \
    --max_length 128 \
    --batch_size 32 \
    --num_workers 4 \
    --language_model google/flan-t5-base \
    --seed 42
```

### Arguments

- `--data_path`: Path to the FEIDEGGER JSON file
- `--output_dir`: Directory to save processed data
- `--image_size`: Target image size (height width)
- `--max_length`: Maximum sequence length for text
- `--batch_size`: Batch size for processing
- `--num_workers`: Number of workers for parallel processing
- `--language_model`: Pretrained model for tokenization
- `--seed`: Random seed for reproducibility

## Output Structure

The script generates:

1. `mamba_dataset/images/`: Directory containing all downloaded images
2. `mamba_dataset/feidegger_mamba_dataset.h5`: HDF5 file containing:
   - Processed images as tensors
   - Tokenized descriptions
   - Train/val/test splits
3. `mamba_dataset/feidegger_mamba_metadata.csv`: CSV file with metadata about all samples

## Using the Dataset in PyTorch

```python
from torch.utils.data import DataLoader
from feidegger_mamba_prep import FeideggerMambaDataset

# Create dataset
train_dataset = FeideggerMambaDataset('mamba_dataset/feidegger_mamba_dataset.h5', split='train')
val_dataset = FeideggerMambaDataset('mamba_dataset/feidegger_mamba_dataset.h5', split='val')
test_dataset = FeideggerMambaDataset('mamba_dataset/feidegger_mamba_dataset.h5', split='test')

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Example usage in training loop
for batch in train_loader:
    images = batch['image']          # Shape: [batch_size, 3, height, width]
    input_ids = batch['input_ids']   # Shape: [batch_size, max_length]
    attention_mask = batch['attention_mask']  # Shape: [batch_size, max_length]
    
    # Your Mamba model training code here...
```

## Recommended Mamba Network Setup

For using this dataset with Mamba networks, consider the following architecture:

1. Use a vision encoder (e.g., ResNet, ViT) to process images into feature vectors
2. Feed image features and text embeddings into a Mamba sequence model
3. Train with appropriate sequence modeling objectives

## Notes

- The script requires internet access to download images
- Processing the complete dataset may take significant time depending on your hardware
- Adjust `max_length` based on the average description length in your dataset
- Consider using a GPU for faster image processing
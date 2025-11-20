#!/usr/bin/env python
"""
FEIDEGGER Dataset Preparation for Mamba Networks

This script downloads the complete FEIDEGGER dataset and prepares it
for training with Mamba network architectures by:
1. Downloading all images
2. Processing images to a consistent format
3. Tokenizing text descriptions
4. Creating sequence pairs
5. Splitting data appropriately
6. Saving in a format suitable for Mamba training

Mamba networks are efficient state space models for sequence modeling,
so this preparation focuses on creating good sequential data.
"""

import json
import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import requests
from io import BytesIO
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from transformers import AutoTokenizer
import h5py
from concurrent.futures import ThreadPoolExecutor, as_completed

# Try to import Hugging Face Hub with Xet Storage support
try:
    import huggingface_hub
    try:
        import hf_xet
        print("Xet Storage support is enabled for faster downloads")
    except ImportError:
        # Check if huggingface_hub has Xet support enabled
        try:
            from huggingface_hub.constants import HF_HUB_XET_ENABLED
            if HF_HUB_XET_ENABLED:
                print("Xet Storage support is enabled via huggingface_hub[hf_xet]")
            else:
                print("Xet Storage is enabled for this repo, but the 'hf_xet' package is not installed.")
                print("Falling back to regular HTTP download. For better performance, install the package with:")
                print("`pip install huggingface_hub[hf_xet]` or `pip install hf_xet`")
        except ImportError:
            print("Xet Storage support not available. Using standard downloads.")
except ImportError:
    print("huggingface_hub not found, using standard downloads only.")

# Add parent directory to path to import FeideggerCrawler
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from likely_delete.feidegger_crawler import FeideggerCrawler

class FeideggerPreprocessor:
    """
    Preprocesses the FEIDEGGER dataset for use with Mamba networks.
    """
    
    def __init__(self, data_path, output_dir, image_size=(224, 224), 
                 max_length=128, batch_size=32, num_workers=4,
                 language_model="google/flan-t5-base", use_xet=True):
        """
        Initialize the preprocessor.
        
        Args:
            data_path (str): Path to the FEIDEGGER JSON file
            output_dir (str): Directory to save processed data
            image_size (tuple): Target image size (height, width)
            max_length (int): Maximum sequence length for text
            batch_size (int): Batch size for processing
            num_workers (int): Number of workers for parallel processing
            language_model (str): Pretrained model for tokenization
            use_xet (bool): Whether to use Xet Storage for downloads from Hugging Face Hub
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.image_size = image_size
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.language_model = language_model
        self.use_xet = use_xet
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        self.images_dir = os.path.join(output_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Load data
        self.crawler = FeideggerCrawler(data_path)
        
        # Initialize tokenizer
        try:
            # Configure Hugging Face Hub to use Xet Storage if available and requested
            kwargs = {}
            if self.use_xet:
                try:
                    # Check if huggingface_hub is installed with xet support
                    import importlib.util
                    if importlib.util.find_spec('hf_xet') is not None or importlib.util.find_spec('huggingface_hub.constants'):
                        kwargs['use_xet'] = True
                        print(f"Using Xet Storage for downloading {language_model}")
                except ImportError:
                    pass
            
            self.tokenizer = AutoTokenizer.from_pretrained(language_model, **kwargs)
            print(f"Loaded tokenizer from {language_model}")
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            print("Using basic tokenization instead.")
            self.tokenizer = None
            
        # Image transformation
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
    
    def download_images(self):
        """
        Download all images from the dataset.
        
        Returns:
            dict: Mapping from item indices to local image paths
        """
        print(f"Downloading {len(self.crawler.data)} images...")
        image_paths = {}
        
        def download_image(idx, item):
            url = item.get('url')
            if not url:
                return idx, None
            
            try:
                # Extract filename from URL and create a unique name
                filename = f"{idx:05d}.jpg"
                filepath = os.path.join(self.images_dir, filename)
                
                # Skip if already downloaded
                if os.path.exists(filepath):
                    return idx, filepath
                
                # Download and save the image
                response = requests.get(url, timeout=10)
                if response.status_code != 200:
                    return idx, None
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                # Verify the image can be opened
                try:
                    img = Image.open(filepath)
                    img.verify()  # Verify it's a valid image
                    return idx, filepath
                except:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    return idx, None
                
            except Exception as e:
                print(f"Error downloading image {idx}: {e}")
                return idx, None
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_idx = {
                executor.submit(download_image, idx, item): idx 
                for idx, item in enumerate(self.crawler.data)
            }
            
            for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx)):
                idx, path = future.result()
                if path:
                    image_paths[idx] = path
        
        print(f"Downloaded {len(image_paths)} images successfully")
        return image_paths
    
    def process_images(self, image_paths):
        """
        Process images to a consistent format.
        
        Args:
            image_paths (dict): Mapping from item indices to local image paths
            
        Returns:
            dict: Mapping from item indices to processed image tensors
        """
        print("Processing images...")
        processed_images = {}
        
        for idx, path in tqdm(image_paths.items()):
            try:
                img = Image.open(path).convert('RGB')
                tensor = self.transform(img)
                processed_images[idx] = tensor
            except Exception as e:
                print(f"Error processing image {idx} at {path}: {e}")
        
        return processed_images
    
    def tokenize_descriptions(self):
        """
        Tokenize all text descriptions in the dataset.
        
        Returns:
            dict: Mapping from (item_idx, desc_idx) to tokenized description
        """
        print("Tokenizing descriptions...")
        tokenized_descriptions = {}
        
        for item_idx, item in enumerate(tqdm(self.crawler.data)):
            descriptions = item.get('descriptions', [])
            for desc_idx, desc in enumerate(descriptions):
                if self.tokenizer:
                    # Use the pretrained tokenizer
                    encoded = self.tokenizer(
                        desc, 
                        max_length=self.max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
                    )
                    tokenized_descriptions[(item_idx, desc_idx)] = {
                        'input_ids': encoded['input_ids'].squeeze(),
                        'attention_mask': encoded['attention_mask'].squeeze()
                    }
                else:
                    # Simple tokenization (just for demonstration)
                    words = desc.split()
                    word_ids = [hash(word) % 10000 for word in words]  # Simple hash-based tokenization
                    if len(word_ids) > self.max_length:
                        word_ids = word_ids[:self.max_length]
                    else:
                        word_ids = word_ids + [0] * (self.max_length - len(word_ids))
                    
                    tokenized_descriptions[(item_idx, desc_idx)] = {
                        'input_ids': torch.tensor(word_ids),
                        'attention_mask': torch.tensor([1] * len(words) + [0] * (self.max_length - len(words)))
                    }
        
        return tokenized_descriptions
    
    def create_dataset_splits(self, image_paths, tokenized_descriptions):
        """
        Create train, validation and test splits of the dataset.
        
        Args:
            image_paths (dict): Mapping from item indices to image paths
            tokenized_descriptions (dict): Mapping from (item_idx, desc_idx) to tokenized text
            
        Returns:
            tuple: (train_data, val_data, test_data) for Mamba network training
        """
        print("Creating dataset splits...")
        
        # Create image-text pairs
        pairs = []
        for item_idx, item in enumerate(self.crawler.data):
            if item_idx not in image_paths:
                continue
                
            image_path = image_paths[item_idx]
            split = item.get('split')
            
            for desc_idx, _ in enumerate(item.get('descriptions', [])):
                if (item_idx, desc_idx) in tokenized_descriptions:
                    pairs.append({
                        'item_idx': item_idx,
                        'desc_idx': desc_idx,
                        'image_path': image_path,
                        'split': split
                    })
        
        # Split the data based on the provided split values if available
        # If not, do a random 80/10/10 split
        train_data = []
        val_data = []
        test_data = []
        
        # Check if we have split information
        if all(pair.get('split') for pair in pairs):
            # Group splits
            split_groups = {}
            for pair in pairs:
                split = pair.get('split')
                if split not in split_groups:
                    split_groups[split] = []
                split_groups[split].append(pair)
            
            # Determine train/val/test based on splits
            # This is a simple approach; you might want to adapt this based on specific requirements
            splits = list(split_groups.keys())
            random.shuffle(splits)
            
            num_splits = len(splits)
            train_splits = splits[:int(0.8 * num_splits)]
            val_splits = splits[int(0.8 * num_splits):int(0.9 * num_splits)]
            test_splits = splits[int(0.9 * num_splits):]
            
            for split in train_splits:
                train_data.extend(split_groups[split])
            for split in val_splits:
                val_data.extend(split_groups[split])
            for split in test_splits:
                test_data.extend(split_groups[split])
        else:
            # Random split
            random.shuffle(pairs)
            n = len(pairs)
            train_data = pairs[:int(0.8 * n)]
            val_data = pairs[int(0.8 * n):int(0.9 * n)]
            test_data = pairs[int(0.9 * n):]
        
        print(f"Split dataset into {len(train_data)} train, {len(val_data)} validation, "
              f"and {len(test_data)} test samples")
        
        return train_data, val_data, test_data
    
    def save_hdf5_dataset(self, processed_images, tokenized_descriptions, splits):
        """
        Save the processed dataset in HDF5 format.
        
        Args:
            processed_images (dict): Mapping from item indices to processed image tensors
            tokenized_descriptions (dict): Mapping from (item_idx, desc_idx) to tokenized text
            splits (tuple): (train_data, val_data, test_data) for Mamba network training
        """
        print("Saving dataset in HDF5 format...")
        
        train_data, val_data, test_data = splits
        
        # Create a single HDF5 file
        h5_path = os.path.join(self.output_dir, "feidegger_mamba_dataset.h5")
        with h5py.File(h5_path, 'w') as f:
            # Create groups for train, val, test
            train_group = f.create_group("train")
            val_group = f.create_group("val")
            test_group = f.create_group("test")
            
            # Save metadata
            f.attrs['num_train'] = len(train_data)
            f.attrs['num_val'] = len(val_data)
            f.attrs['num_test'] = len(test_data)
            f.attrs['image_size'] = self.image_size
            f.attrs['max_length'] = self.max_length
            
            # Helper function to save a data split
            def save_split(group, data_split):
                for i, pair in enumerate(tqdm(data_split)):
                    item_idx = pair['item_idx']
                    desc_idx = pair['desc_idx']
                    
                    # Create a group for this sample
                    sample_group = group.create_group(f"sample_{i}")
                    
                    # Save metadata
                    sample_group.attrs['item_idx'] = item_idx
                    sample_group.attrs['desc_idx'] = desc_idx
                    sample_group.attrs['image_path'] = pair['image_path']
                    if 'split' in pair:
                        sample_group.attrs['original_split'] = pair['split']
                    
                    # Save image tensor
                    if item_idx in processed_images:
                        img_tensor = processed_images[item_idx].numpy()
                        sample_group.create_dataset("image", data=img_tensor)
                    
                    # Save tokenized description
                    if (item_idx, desc_idx) in tokenized_descriptions:
                        tokens = tokenized_descriptions[(item_idx, desc_idx)]
                        sample_group.create_dataset("input_ids", data=tokens['input_ids'].numpy())
                        sample_group.create_dataset("attention_mask", data=tokens['attention_mask'].numpy())
            
            # Save each split
            print("Saving training data...")
            save_split(train_group, train_data)
            
            print("Saving validation data...")
            save_split(val_group, val_data)
            
            print("Saving test data...")
            save_split(test_group, test_data)
        
        print(f"Dataset saved to {h5_path}")
    
    def save_metadata_csv(self, splits):
        """
        Save dataset metadata and splits to CSV for easier inspection.
        
        Args:
            splits (tuple): (train_data, val_data, test_data) for Mamba network training
        """
        train_data, val_data, test_data = splits
        
        # Function to convert a split to a DataFrame
        def split_to_df(data_split, split_name):
            rows = []
            for pair in data_split:
                item_idx = pair['item_idx']
                desc_idx = pair['desc_idx']
                
                # Get original text
                if item_idx < len(self.crawler.data):
                    item = self.crawler.data[item_idx]
                    descriptions = item.get('descriptions', [])
                    if desc_idx < len(descriptions):
                        text = descriptions[desc_idx]
                    else:
                        text = "N/A"
                else:
                    text = "N/A"
                
                rows.append({
                    'item_idx': item_idx,
                    'desc_idx': desc_idx,
                    'image_path': pair['image_path'],
                    'original_split': pair.get('split', 'unknown'),
                    'mamba_split': split_name,
                    'text': text
                })
            
            return pd.DataFrame(rows)
        
        # Create DataFrames for each split
        train_df = split_to_df(train_data, 'train')
        val_df = split_to_df(val_data, 'val')
        test_df = split_to_df(test_data, 'test')
        
        # Combine into a single DataFrame
        all_df = pd.concat([train_df, val_df, test_df])
        
        # Save to CSV
        csv_path = os.path.join(self.output_dir, "feidegger_mamba_metadata.csv")
        all_df.to_csv(csv_path, index=False)
        print(f"Metadata saved to {csv_path}")
    
    def process(self):
        """
        Process the entire dataset.
        """
        # Step 1: Download images
        image_paths = self.download_images()
        
        # Step 2: Process images
        processed_images = self.process_images(image_paths)
        
        # Step 3: Tokenize descriptions
        tokenized_descriptions = self.tokenize_descriptions()
        
        # Step 4: Create dataset splits
        splits = self.create_dataset_splits(image_paths, tokenized_descriptions)
        
        # Step 5: Save dataset in HDF5 format
        self.save_hdf5_dataset(processed_images, tokenized_descriptions, splits)
        
        # Step 6: Save metadata for easier inspection
        self.save_metadata_csv(splits)
        
        print("Processing complete!")


class FeideggerMambaDataset(Dataset):
    """
    PyTorch dataset for FEIDEGGER data prepared for Mamba networks.
    """
    
    def __init__(self, h5_path, split='train'):
        """
        Initialize the dataset.
        
        Args:
            h5_path (str): Path to the HDF5 file
            split (str): Dataset split ('train', 'val', or 'test')
        """
        self.h5_path = h5_path
        self.split = split
        
        # Get dataset size
        with h5py.File(h5_path, 'r') as f:
            self.size = f.attrs[f'num_{split}']
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            sample_group = f[self.split][f'sample_{idx}']
            
            # Get image
            img = torch.tensor(sample_group['image'][()])
            
            # Get text tokens
            input_ids = torch.tensor(sample_group['input_ids'][()])
            attention_mask = torch.tensor(sample_group['attention_mask'][()])
            
            return {
                'image': img,
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='FEIDEGGER Dataset Preparation for Mamba Networks'
    )
    parser.add_argument('--data_path', type=str, 
                        default='data/FEIDEGGER_release_1.2.json',
                        help='Path to the FEIDEGGER JSON file')
    parser.add_argument('--output_dir', type=str, 
                        default='mamba_dataset',
                        help='Directory to save processed data')
    parser.add_argument('--image_size', type=int, nargs=2,
                        default=[224, 224],
                        help='Target image size (height width)')
    parser.add_argument('--max_length', type=int, 
                        default=128,
                        help='Maximum sequence length for text')
    parser.add_argument('--batch_size', type=int, 
                        default=32,
                        help='Batch size for processing')
    parser.add_argument('--num_workers', type=int, 
                        default=4,
                        help='Number of workers for parallel processing')
    parser.add_argument('--language_model', type=str,
                        default="google/flan-t5-base",
                        help='Pretrained model for tokenization')
    parser.add_argument('--seed', type=int, 
                        default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--use_xet', action='store_true',
                        help='Use Xet Storage for faster downloads from Hugging Face Hub')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create and run the preprocessor
    preprocessor = FeideggerPreprocessor(
        data_path=args.data_path,
        output_dir=args.output_dir,
        image_size=tuple(args.image_size),
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        language_model=args.language_model,
        use_xet=args.use_xet
    )
    
    preprocessor.process()


if __name__ == "__main__":
    main()
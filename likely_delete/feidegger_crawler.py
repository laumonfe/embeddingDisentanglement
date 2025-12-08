#!/usr/bin/env python
"""
FEIDEGGER Dataset Crawler

This script allows you to load, explore, and analyze the FEIDEGGER dataset,
which contains fashion images and descriptions in German.

The dataset consists of 8732 high-resolution images of dresses, each with
5 textual annotations in German.
"""

import json
import os
import random
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
import re
import pandas as pd
from tqdm import tqdm
import argparse

class FeideggerCrawler:
    """
    A class to crawl and analyze the FEIDEGGER dataset.
    """
    
    def __init__(self, data_path):
        """
        Initialize the crawler with the path to the dataset.
        
        Args:
            data_path (str): Path to the FEIDEGGER JSON file
        """
        self.data_path = data_path
        self.data = None
        self.load_data()
        
    def load_data(self):
        """Load the FEIDEGGER dataset from JSON file."""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as file:
                self.data = json.load(file)
            print(f"Successfully loaded {len(self.data)} items from the dataset.")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            self.data = []


    
    def download_images(self, output_dir, max_images=None, split=None):
        """
        Download images from the dataset.
        
        Args:
            output_dir (str): Directory to save the images
            max_images (int, optional): Maximum number of images to download
            split (str, optional): Download only images from this split
        """
        if not self.data:
            print("No data loaded")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Filter by split if specified
        items = self.data
        if split is not None:
            items = [item for item in items if item.get('split') == str(split)]
        
        # Limit the number of images if specified
        if max_images is not None:
            items = items[:min(max_images, len(items))]
        
        print(f"Downloading {len(items)} images to {output_dir}...")
        
        for i, item in enumerate(tqdm(items)):
            url = item.get('url')
            if not url:
                continue
            
            try:
                # Extract filename from URL
                filename = os.path.basename(url).split('?')[0]
                filepath = os.path.join(output_dir, f"{i+1}_{filename}")
                
                # Download and save the image
                response = requests.get(url)
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
            except Exception as e:
                print(f"Error downloading image {url}: {e}")
    
    def export_to_csv(self, output_path):
        """
        Export the dataset to a CSV file.
        
        Args:
            output_path (str): Path to save the CSV file
        """
        if not self.data:
            print("No data loaded")
            return
        
        # Convert to DataFrame
        rows = []
        for item in self.data:
            url = item.get('url', '')
            split = item.get('split', '')
            descriptions = item.get('descriptions', [])
            
            # Create a row for each description
            for i, desc in enumerate(descriptions):
                rows.append({
                    'url': url,
                    'split': split,
                    'description_id': i + 1,
                    'description': desc
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"Dataset exported to {output_path}")


def main():
    """Main function to run the crawler from command line."""
    parser = argparse.ArgumentParser(description='FEIDEGGER Dataset Crawler')
    parser.add_argument('--data_path', type=str, 
                        default='data/FEIDEGGER_release_1.2.json',
                        help='Path to the FEIDEGGER JSON file')
    parser.add_argument('--download_dir', type=str, default='images',
                        help='Directory to save downloaded images')
    parser.add_argument('--output_csv', type=str, default='feidegger_dataset.csv',
                        help='Path to save the exported CSV file')
    
    args = parser.parse_args()
    
    # Create crawler
    crawler = FeideggerCrawler(args.data_path)
    crawler.download_images(args.download_dir, args.max_images, args.split)
    crawler.export_to_csv(args.output_csv)

if __name__ == "__main__":
    main()
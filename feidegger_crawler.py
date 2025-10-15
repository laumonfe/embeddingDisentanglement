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
    
    def get_dataset_statistics(self):
        """
        Get basic statistics about the dataset.
        
        Returns:
            dict: Dictionary containing dataset statistics
        """
        if not self.data:
            return {"error": "No data loaded"}
        
        # Count items by split
        split_counts = Counter([item.get('split', 'unknown') for item in self.data])
        
        # Count descriptions
        total_descriptions = sum(len(item.get('descriptions', [])) for item in self.data)
        avg_descriptions_per_item = total_descriptions / len(self.data) if self.data else 0
        
        # Analyze description length
        desc_lengths = [len(desc) for item in self.data 
                        for desc in item.get('descriptions', [])]
        
        stats = {
            "total_items": len(self.data),
            "split_distribution": dict(split_counts),
            "total_descriptions": total_descriptions,
            "avg_descriptions_per_item": avg_descriptions_per_item,
            "avg_description_length": sum(desc_lengths) / len(desc_lengths) if desc_lengths else 0,
            "min_description_length": min(desc_lengths) if desc_lengths else 0,
            "max_description_length": max(desc_lengths) if desc_lengths else 0
        }
        
        return stats
    
    def show_random_samples(self, num_samples=5, download_images=False):
        """
        Display random samples from the dataset.
        
        Args:
            num_samples (int): Number of samples to display
            download_images (bool): Whether to download and display images
        """
        if not self.data:
            print("No data loaded")
            return
        
        samples = random.sample(self.data, min(num_samples, len(self.data)))
        
        for i, sample in enumerate(samples):
            print(f"\nSample {i+1}:")
            print(f"URL: {sample.get('url', 'N/A')}")
            print(f"Split: {sample.get('split', 'N/A')}")
            print("Descriptions:")
            for j, desc in enumerate(sample.get('descriptions', [])):
                print(f"  {j+1}. {desc}")
            
            if download_images:
                try:
                    response = requests.get(sample.get('url'))
                    img = Image.open(BytesIO(response.content))
                    plt.figure(figsize=(5, 5))
                    plt.imshow(img)
                    plt.title(f"Sample {i+1}")
                    plt.axis('off')
                    plt.show()
                except Exception as e:
                    print(f"Error displaying image: {e}")
    
    def search_descriptions(self, query, case_sensitive=False):
        """
        Search for items with descriptions matching the query.
        
        Args:
            query (str): Query string to search for
            case_sensitive (bool): Whether to perform case-sensitive search
            
        Returns:
            list: Items with matching descriptions
        """
        if not self.data:
            return []
        
        results = []
        
        for item in self.data:
            for desc in item.get('descriptions', []):
                if case_sensitive:
                    if query in desc:
                        results.append(item)
                        break
                else:
                    if query.lower() in desc.lower():
                        results.append(item)
                        break
        
        return results
    
    def get_items_by_split(self, split):
        """
        Get items belonging to a specific split.
        
        Args:
            split (str): Split identifier
            
        Returns:
            list: Items belonging to the specified split
        """
        if not self.data:
            return []
        
        return [item for item in self.data if item.get('split') == str(split)]
    
    def extract_common_words(self, num_words=50, min_length=3):
        """
        Extract the most common words in the descriptions.
        
        Args:
            num_words (int): Number of top words to return
            min_length (int): Minimum length of words to consider
            
        Returns:
            list: List of (word, frequency) tuples
        """
        if not self.data:
            return []
        
        # Combine all descriptions
        all_text = ' '.join([desc for item in self.data 
                           for desc in item.get('descriptions', [])])
        
        # Simple tokenization (split by spaces and remove punctuation)
        words = re.findall(r'\b[a-zA-ZäöüÄÖÜß]+\b', all_text.lower())
        
        # Filter by minimum length
        words = [word for word in words if len(word) >= min_length]
        
        # Count frequency
        word_counts = Counter(words)
        
        # Return the most common words
        return word_counts.most_common(num_words)
    
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
    parser.add_argument('--action', type=str, 
                        choices=['stats', 'sample', 'download', 'search', 'export', 'words'],
                        default='stats',
                        help='Action to perform')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to display')
    parser.add_argument('--download_dir', type=str, default='images',
                        help='Directory to save downloaded images')
    parser.add_argument('--max_images', type=int, default=100,
                        help='Maximum number of images to download')
    parser.add_argument('--split', type=str, default=None,
                        help='Split to filter by')
    parser.add_argument('--search_query', type=str, default='',
                        help='Query to search for in descriptions')
    parser.add_argument('--output_csv', type=str, default='feidegger_dataset.csv',
                        help='Path to save the exported CSV file')
    
    args = parser.parse_args()
    
    # Create crawler
    crawler = FeideggerCrawler(args.data_path)
    
    # Perform requested action
    if args.action == 'stats':
        stats = crawler.get_dataset_statistics()
        print("\nDataset Statistics:")
        for key, value in stats.items():
            print(f"{key}: {value}")
    
    elif args.action == 'sample':
        crawler.show_random_samples(args.num_samples, download_images=True)
    
    elif args.action == 'download':
        crawler.download_images(args.download_dir, args.max_images, args.split)
    
    elif args.action == 'search':
        results = crawler.search_descriptions(args.search_query)
        print(f"\nFound {len(results)} items matching '{args.search_query}':")
        for i, item in enumerate(results[:10]):  # Show first 10 results
            print(f"\nResult {i+1}:")
            print(f"URL: {item.get('url', 'N/A')}")
            print(f"Split: {item.get('split', 'N/A')}")
            print("Matching Descriptions:")
            for j, desc in enumerate(item.get('descriptions', [])):
                if args.search_query.lower() in desc.lower():
                    print(f"  {j+1}. {desc}")
    
    elif args.action == 'export':
        crawler.export_to_csv(args.output_csv)
    
    elif args.action == 'words':
        common_words = crawler.extract_common_words(50)
        print("\nMost Common Words:")
        for word, count in common_words:
            print(f"{word}: {count}")


if __name__ == "__main__":
    main()
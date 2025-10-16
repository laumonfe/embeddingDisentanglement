#!/usr/bin/env python
"""
FEIDEGGER Dataset Crawler Example

This script demonstrates how to use the FeideggerCrawler class
to analyze and work with the FEIDEGGER dataset.
"""

from feidegger_crawler import FeideggerCrawler
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    """Example usage of the FeideggerCrawler."""
    # Path to the dataset
    data_path = 'data/FEIDEGGER_release_1.2.json'
    
    print(f"Initializing crawler with dataset at {data_path}...")
    crawler = FeideggerCrawler(data_path)
    
    # Get and print dataset statistics
    print("\n1. Generating dataset statistics...")
    stats = crawler.get_dataset_statistics()
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Plot split distribution if matplotlib is available
    print("\n2. Plotting split distribution...")
    splits = stats.get('split_distribution', {})
    plt.figure(figsize=(10, 6))
    plt.bar(splits.keys(), splits.values())
    plt.title('Distribution of Items by Split')
    plt.xlabel('Split')
    plt.ylabel('Count')
    plt.savefig('split_distribution.png')
    print(f"  Plot saved as split_distribution.png")
    
    # Show some sample descriptions
    print("\n3. Showing sample descriptions...")
    crawler.show_random_samples(num_samples=2, download_images=False)
    
    # Search for items containing a specific word
    print("\n4. Searching for items with 'blau' in descriptions...")
    blue_items = crawler.search_descriptions("blau")
    print(f"  Found {len(blue_items)} items with 'blau' in descriptions")
    
    # Extract common words
    print("\n5. Extracting most common words...")
    common_words = crawler.extract_common_words(num_words=15)
    print("  Most common words:")
    for word, count in common_words:
        print(f"    {word}: {count}")
    
    # Plot word frequencies
    print("\n6. Plotting word frequencies...")
    plt.figure(figsize=(12, 8))
    words = [word for word, _ in common_words]
    counts = [count for _, count in common_words]
    plt.barh(words, counts)
    plt.title('Most Common Words in Descriptions')
    plt.xlabel('Frequency')
    plt.tight_layout()
    plt.savefig('word_frequencies.png')
    print(f"  Plot saved as word_frequencies.png")
    
    # Create a small folder with a few images
    print("\n7. Downloading a few sample images...")
    os.makedirs('sample_images', exist_ok=True)
    crawler.download_images('sample_images', max_images=5)
    
    # Export a subset of the data to CSV
    print("\n8. Exporting a subset to CSV...")
    # Get items from split 7
    split_7_items = crawler.get_items_by_split('7')
    # Create a new crawler with just these items
    subset_crawler = FeideggerCrawler(data_path)
    subset_crawler.data = split_7_items[:10]  # Just take first 10 items
    # Export to CSV
    subset_crawler.export_to_csv('split7_sample.csv')
    print(f"  Exported sample to split7_sample.csv")
    
    print("\nAll example operations completed successfully!")

if __name__ == "__main__":
    main()
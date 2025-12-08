# FEIDEGGER Dataset Crawler

This script allows you to explore, analyze and work with the FEIDEGGER dataset, which contains fashion images and descriptions in German.

## Overview

The FEIDEGGER dataset consists of 8732 high-resolution images of dresses, each with 5 textual annotations in German. This crawler provides the following functionality:

- Load and explore the dataset
- Generate dataset statistics
- View random samples with descriptions and images
- Search for specific terms in descriptions
- Extract common words used in the descriptions
- Download images from the dataset
- Export the dataset to CSV format

## Requirements

1. Install dependencies:
```
python install_dependencies.py
```
Or with a Conda environment:

```
python install_dependencies.py --conda --env-name feidegger-mamba
```



## Usage

### Basic Usage

```bash
python feidegger_crawler.py --data_path data/FEIDEGGER_release_1.2.json --action stats
```

### Available Actions

1. **Get dataset statistics**

```bash
python feidegger_crawler.py --action stats
```

2. **Show random samples**

```bash
python feidegger_crawler.py --action sample --num_samples 5
```

3. **Download images**

```bash
python feidegger_crawler.py --action download --download_dir images --max_images 100
```

4. **Search for descriptions**

```bash
python feidegger_crawler.py --action search --search_query "blau"
```

5. **Export to CSV**

```bash
python feidegger_crawler.py --action export --output_csv feidegger_dataset.csv
```

6. **Extract common words**

```bash
python feidegger_crawler.py --action words
```

### Filter by Split

You can filter by dataset split for certain actions:

```bash
python feidegger_crawler.py --action download --split 7 --download_dir split7_images
```

## Using as a Python Module

You can also import the crawler in your own Python scripts:

```python
from feidegger_crawler import FeideggerCrawler

# Initialize the crawler
crawler = FeideggerCrawler('data/FEIDEGGER_release_1.2.json')

# Get dataset statistics
stats = crawler.get_dataset_statistics()
print(stats)

# Show random samples
crawler.show_random_samples(num_samples=3, download_images=True)

# Search descriptions
results = crawler.search_descriptions('rot')
print(f"Found {len(results)} items with 'rot' in descriptions")
```

## Notes

- The image download functionality requires a stable internet connection as it retrieves images from the provided URLs
- For large operations (like downloading all images), consider using the `max_images` parameter to limit the number of items processed
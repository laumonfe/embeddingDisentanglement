# FEIDEGGER Crawler

This script downloads images and prepares metadata for the FEIDEGGER dataset.

## Features

- Loads dataset from a JSON file.
- Downloads images from URLs in the dataset.
- Creates train/val/test splits.
- Saves metadata (image paths and descriptions) to a CSV file.

## Usage

```bash
python feidegger_crawler.py --data_path <path_to_json> --download_dir <output_folder> --output_csv <csv_file> --max_images <N>
```

### Arguments

- `--data_path`: Path to the FEIDEGGER JSON dataset (default: `data/FEIDEGGER_release_1.2.json`)
- `--download_dir`: Directory to save downloaded images (default: `data/images`)
- `--output_csv`: Output CSV file for metadata (default: `data/feidegger_metadata.csv`)
- `--max_images`: Maximum number of images to download (default: `None`)



## Output

- Images are saved in the specified download directory.
- Metadata CSV is saved under the same download directory. It contains image paths, descriptions, and split information.


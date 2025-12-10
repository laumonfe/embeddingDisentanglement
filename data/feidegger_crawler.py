import os
import json
import pandas as pd
import requests
from tqdm import tqdm

class FeideggerCrawler:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = self.load_data()

    def load_data(self):
        try:
            with open(self.data_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            print(f"Loaded {len(data)} items from the dataset.")
            return data
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return []

    def download_images(self, output_dir, max_images=None):
        if not self.data:
            print("No data loaded")
            return {}
        os.makedirs(output_dir, exist_ok=True)
        items = self.data[:max_images] if max_images else self.data
        image_paths = {}
        print(f"Downloading {len(items)} images to {output_dir}...")
        for i, item in enumerate(tqdm(items)):
            url = item.get('url')
            if not url:
                continue
            filename = f"{i+1}_{os.path.basename(url).split('?')[0]}"
            filepath = os.path.join(output_dir, filename)
            try:
                response = requests.get(url)
                response.raise_for_status()
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                image_paths[i] = filepath
            except Exception as e:
                print(f"Error downloading image {url}: {e}")
        return image_paths

    def create_splits_and_save_metadata(self, image_paths, output_csv):
        print("Creating dataset splits...")
        # Group all pairs by item_idx
        item_idx_to_pairs = {}
        for item_idx, item in enumerate(self.data):
            if item_idx not in image_paths:
                continue
            image_path = image_paths[item_idx]
            if not os.path.isfile(image_path):
                continue
            for desc_idx, _ in enumerate(item.get('descriptions', [])):
                text = item['descriptions'][desc_idx] if desc_idx < len(item['descriptions']) else "N/A"
                item_idx_to_pairs.setdefault(item_idx, []).append({
                    'item_idx': item_idx,
                    'desc_idx': desc_idx,
                    'image_path': image_path,
                    'text': text
                })
        item_indices = list(item_idx_to_pairs.keys())
        n_items = len(item_indices)
        train_cut = int(0.8 * n_items)
        val_cut = int(0.9 * n_items)
        train_items = item_indices[:train_cut]
        val_items = item_indices[train_cut:val_cut]
        test_items = item_indices[val_cut:]

        all_data = []
        for split_name, split_items in zip(['train', 'val', 'test'], [train_items, val_items, test_items]):
            for item_idx in split_items:
                for pair in item_idx_to_pairs[item_idx]:
                    pair['split'] = split_name
                    all_data.append(pair)

        df = pd.DataFrame(all_data)
        df.to_csv(output_csv, index=False)
        print(f"Metadata saved to {output_csv}")
        # Return splits for compatibility
        train = [row for row in all_data if row['split'] == 'train']
        val = [row for row in all_data if row['split'] == 'val']
        test = [row for row in all_data if row['split'] == 'test']
        return train, val, test

def main():
    import argparse
    parser = argparse.ArgumentParser(description='FEIDEGGER Dataset Crawler')
    parser.add_argument('--data_path', type=str, default='data/FEIDEGGER_release_1.2.json')
    parser.add_argument('--download_dir', type=str, default='data')
    parser.add_argument('--output_csv', type=str, default='feidegger_metadata.csv')
    parser.add_argument('--max_images', type=int, default=None)
    args = parser.parse_args()

    crawler = FeideggerCrawler(args.data_path)
    image_paths = crawler.download_images(os.path.join(args.download_dir,"images"), args.max_images)
    crawler.create_splits_and_save_metadata(image_paths, os.path.join(args.download_dir,args.output_csv))

if __name__ == "__main__":
    main()
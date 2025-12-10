import torch
from PIL import Image
from collections import defaultdict
from torch.utils.data import Dataset


class CLIPDataset(Dataset):
    def __init__(self, dataframe, image_preprocessor, text_tokenizer):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame with columns:
                ['item_idx', 'desc_idx', 'image_path', 'original_split', 'split', 'text']
            image_transform (callable, optional): Transform to apply to images.
            text_transform (callable, optional): Transform to apply to text (e.g., tokenizer).
        """
        self.df = dataframe.reset_index(drop=True)
        self.image_preprocessor = image_preprocessor
        self.text_tokenizer = text_tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['image_path']
        text = row['text']

        # Load image
        image = Image.open(image_path).convert("RGB")
        processed_image = self.image_preprocessor(images=image, return_tensors="pt")
        pixel_values = processed_image["pixel_values"].squeeze(0)

        # Tokenize text
        tokens = self.text_tokenizer([text], return_tensors="pt", padding=True, truncation=True)
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'item_idx': row['item_idx'],
            'desc_idx': row['desc_idx'],
            'image_path': image_path
        }
    


class GroupedCLIPDataset(torch.utils.data.Dataset):
    def __init__(self, grouped_data, image_preprocessor, text_tokenizer):
        self.grouped_data = grouped_data
        self.image_preprocessor = image_preprocessor
        self.text_tokenizer = text_tokenizer

    def __len__(self):
        return len(self.grouped_data)

    def __getitem__(self, idx):
        group = self.grouped_data[idx]
        image_path = group[0]['image_path']
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.image_preprocessor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        texts = [row['text'] for row in group]
        tokenized = self.text_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        return {
            "pixel_values": pixel_values,
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "item_idx": [row['item_idx'] for row in group],
            "desc_idx": [row['desc_idx'] for row in group],
            "image_path": image_path
        }

def group_by_image(df):
    grouped = defaultdict(list)
    for idx, row in df.iterrows():
        grouped[row['image_path']].append(row)
    return list(grouped.values())
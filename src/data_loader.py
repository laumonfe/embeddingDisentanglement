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




# Add this block at the end of the file
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from PIL import Image
    import os 
    from src.models import PretrainedCLIPVision, PretrainedDistilBert
    CSV_PATH = r"data/feidegger_metadata.csv"
    pretrained_dir = "pretrained_models"
    from src.train_clip import collate_fn


    import pandas as pd
    from torch.utils.data import DataLoader

    df = pd.read_csv(CSV_PATH)

    test_df = df[df["split"] == "test"]

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Load pretrained models
    pretrained_img_model_path = os.path.join(pretrained_dir, "sentence-transformers--clip-ViT-B-32")
    pretrained_text_model_path = os.path.join(pretrained_dir, "sentence-transformers--clip-ViT-B-32-multilingual-v1")

    image_encoder = PretrainedCLIPVision(pretrained_img_model_path, device)
    text_encoder = PretrainedDistilBert(pretrained_text_model_path, device)
    
    test_dataset = CLIPDataset(test_df, image_encoder.processor,  text_encoder.tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True, collate_fn=collate_fn)

    batch = next(iter(test_loader))
    #print(batch)
    # Processed image tensor
    img_tensor = batch["pixel_values"][5].cpu()
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    img_disp = img_tensor.clone()
    for c in range(3):
        img_disp[c] = img_disp[c] * std[c] + mean[c]
    img_disp = img_disp.clamp(0, 1)
    img_np = img_disp.permute(1, 2, 0).numpy()  # Convert to HWC for matplotlib

    import matplotlib.pyplot as plt
    plt.figure()
    plt.title("Processed Image (matplotlib)")
    plt.imshow(img_np)
    plt.axis('off')
    plt.show()
# limitations :( 
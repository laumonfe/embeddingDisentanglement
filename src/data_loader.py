import torch 
from PIL import Image


# For finetuning, you need to create a custom dataset and Trainer
class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, texts, processor):
        self.image_paths = image_paths
        self.texts = texts
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        text = self.texts[idx]
        inputs = self.processor(text=[text], images=image, return_tensors="pt", padding=True)
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        return item

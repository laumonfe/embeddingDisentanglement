# TextEncoder.py
import os 
from pandas.core import base
import torch
import json 
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
from transformers import CLIPVisionModel, CLIPProcessor, CLIPModel


def load_json_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    

def load_projection_weights(weights_path, device):
    """
    Loads projection weights from a file and remaps keys if necessary.
    Returns a state_dict ready for nn.Linear.
    """
    proj_weights = torch.load(weights_path, map_location=device)
    if "linear.weight" in proj_weights:
        proj_weights["weight"] = proj_weights.pop("linear.weight")
    if "linear.bias" in proj_weights:
        proj_weights["bias"] = proj_weights.pop("linear.bias")
    return proj_weights

class ProjectedDistilBert(nn.Module):
    def __init__(self, model_path, device):
        super().__init__()
        self.proj_weights_path = os.path.join(model_path,  "text_projection.bin")
        self.config_path = os.path.join(model_path, "config.json")
        self.proj_weights = load_projection_weights(self.proj_weights_path, device)
        self.proj_cfg = load_json_config(self.config_path).get("projection", None)
        self.model = DistilBertModel.from_pretrained(model_path).to(device)
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.projection = nn.Linear(self.proj_cfg["in_features"], self.proj_cfg["out_features"], bias=self.proj_cfg["bias"]).to(device)
        self.projection.load_state_dict(self.proj_weights)
        self.device = device

    def encode(self, text):
        tokens = self.tokenizer([text], return_tensors="pt", padding=True, truncation=True)
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        with torch.no_grad():
            outputs = self.model(**tokens)
            pooled = outputs.last_hidden_state.mean(dim=1)
            emb = self.projection(pooled)
        return emb.cpu().numpy()[0]
    


class ProjectedCLIPVision(nn.Module):
    def __init__(self, model_path, device):
        super().__init__()
        self.proj_weights_path = os.path.join(model_path, "visual_projection.bin")
        self.config_path = os.path.join(model_path, "config.json")
        self.proj_weights = load_projection_weights(self.proj_weights_path, device)
        self.proj_cfg = load_json_config(self.config_path).get("visual_projection", None)
        self.model = CLIPVisionModel.from_pretrained(model_path).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.projection = nn.Linear(self.proj_cfg["in_features"], self.proj_cfg["out_features"], bias=self.proj_cfg["bias"]).to(device)
        self.projection.load_state_dict(self.proj_weights)
        self.device = device

    def encode(self, image):
        processed = self.processor(images=image, return_tensors="pt")
        pixel_values = processed["pixel_values"].to(self.device)
        with torch.no_grad():
            outputs = self.model(pixel_values)
            pooled = outputs.pooler_output
            emb = self.projection(pooled)
        return emb.cpu().numpy()[0]
    

class PretrainedCLIPVision(nn.Module):
    def __init__(self, model_dir, device):
        super().__init__()
        self.model = CLIPVisionModel.from_pretrained(model_dir).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_dir)
        clip_model = CLIPModel.from_pretrained(model_dir)
        self.visual_projection = clip_model.visual_projection.to(device)
        self.device = device

    def encode(self, image):
        processed = self.processor(images=image, return_tensors="pt")
        pixel_values = processed["pixel_values"].to(self.device)
        with torch.no_grad():
            outputs = self.model(pixel_values)
            pooled = outputs.pooler_output
            emb = self.visual_projection(pooled)
        return emb.cpu().numpy()[0]
    
    
class PretrainedDistilBert(nn.Module):
    def __init__(self, model_path, device):
        super().__init__()
        self.proj_weights_path = os.path.join(model_path, "2_Dense/pytorch_model.bin")
        self.config_path = os.path.join(model_path, "2_Dense/config.json")
        self.proj_weights = load_projection_weights(self.proj_weights_path, device)
        self.proj_cfg = load_json_config(self.config_path)
        self.model = DistilBertModel.from_pretrained(model_path).to(device)
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.projection = nn.Linear(self.proj_cfg["in_features"], self.proj_cfg["out_features"], bias=self.proj_cfg["bias"]).to(device)
        self.projection.load_state_dict(self.proj_weights)
        self.device = device

    def encode(self, text):
        tokens = self.tokenizer([text], return_tensors="pt", padding=True, truncation=True)
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        with torch.no_grad():
            outputs = self.model(**tokens)
            pooled = outputs.last_hidden_state.mean(dim=1)
            emb = self.projection(pooled)
        return emb.cpu().numpy()[0]

if __name__ == "__main__":

    from sentence_transformers import util

    # Paths to pretrained models
    pretrained_img_model_path = r"pretrained_models/sentence-transformers--clip-ViT-B-32"
    pretrained_text_model_path = r"pretrained_models/sentence-transformers--clip-ViT-B-32-multilingual-v1"


    # Paths to finetuned models
    finetuned_text_model_path = r"output/finetuned_baseline\best_model/text_encoder"
    finetuned_img_model_path = r"output/finetuned_baseline\best_model/vision_encoder"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    baseline_image_encoder = PretrainedCLIPVision(pretrained_img_model_path, device)
    baseline_text_encoder = PretrainedDistilBert(pretrained_text_model_path, device)
    
    finetuned_image_encoder = ProjectedCLIPVision(finetuned_img_model_path, device)
    finetuned_text_encoder = ProjectedDistilBert(finetuned_text_model_path, device)

    query = "A dog playing with a ball."
    baseline_emb = baseline_text_encoder.encode(query)
    finetuned_emb = finetuned_text_encoder.encode(query)    

    sims = util.cos_sim(torch.tensor(baseline_emb), torch.tensor(finetuned_emb))[0]
    print("Cosine similarity between baseline and finetuned text embeddings:", sims.item())

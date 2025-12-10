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
    proj_weights = torch.load(weights_path, map_location=device)
    if "linear.weight" in proj_weights:
        proj_weights["weight"] = proj_weights.pop("linear.weight")
    if "linear.bias" in proj_weights:
        proj_weights["bias"] = proj_weights.pop("linear.bias")
    return proj_weights

# Finetuned Text Encoder 
class ProjectedDistilBert(nn.Module):
    def __init__(self, model_path, device):
        super().__init__()
        self.proj_weights_path = os.path.join(model_path,  "text_projection.bin")
        self.config_path = os.path.join(model_path, "proj_config.json")
        self.proj_weights = load_projection_weights(self.proj_weights_path, device)
        self.proj_cfg = load_json_config(self.config_path).get("projection", None)
        self.model = DistilBertModel.from_pretrained(model_path).to(device)
        self.config = self.model.config 
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

    

# Finetuned Image Encoder
class ProjectedCLIPVision(nn.Module):
    def __init__(self, model_path, device):
        super().__init__()
        self.proj_weights_path = os.path.join(model_path, "visual_projection.bin")
        self.config_path = os.path.join(model_path, "config.json")
        self.proj_weights = load_projection_weights(self.proj_weights_path, device)
        self.proj_cfg = load_json_config(self.config_path).get("visual_projection", None)
        self.model = CLIPVisionModel.from_pretrained(model_path).to(device)
        self.config = self.model.config
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
    
# Pretrained Image Encoder
class PretrainedCLIPVision(nn.Module):
    def __init__(self, model_dir, device):
        super().__init__()
        self.model = CLIPVisionModel.from_pretrained(model_dir).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_dir)
        clip_model = CLIPModel.from_pretrained(model_dir)
        self.config = self.model.config
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
    
 # Pretrained Text Encoder   
class PretrainedDistilBert(nn.Module):
    def __init__(self, model_path, device):
        super().__init__()
        self.proj_weights_path = os.path.join(model_path, "2_Dense/pytorch_model.bin")
        self.config_path = os.path.join(model_path, "2_Dense/config.json")
        self.proj_weights = load_projection_weights(self.proj_weights_path, device)
        self.proj_cfg = load_json_config(self.config_path)
        self.model = DistilBertModel.from_pretrained(model_path).to(device)
        self.config = self.model.config
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
    
# Finetune both encoders together
class FinetuneCLIP(nn.Module):
    def __init__(self, vision_encoder, text_encoder):
        super().__init__()
        self.vision_encoder = vision_encoder
        #self.visual_projection = visual_projection
        self.text_encoder = text_encoder

    def forward(self, pixel_values, input_ids, attention_mask):
        # Vision encoding
        vision_outputs = self.vision_encoder.model(pixel_values)
        vision_pooled = vision_outputs.pooler_output
        vision_embeds = self.vision_encoder.visual_projection(vision_pooled)

        # Text encoding
        text_outputs = self.text_encoder.model(input_ids=input_ids, attention_mask=attention_mask)
        text_pooled = text_outputs.last_hidden_state.mean(dim=1)  
        text_embeds = self.text_encoder.projection(text_pooled)
        return {
            "vision_embeds": vision_embeds,
            "text_embeds": text_embeds
        }
    
    def save_from_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        os.makedirs(os.path.join(save_directory, "vision_encoder"), exist_ok=True)
        os.makedirs(os.path.join(save_directory, "text_encoder"), exist_ok=True)

        # Save combined model weights
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

        # Save vision encoder weights/config using HuggingFace method if available
        if hasattr(self.vision_encoder.model, "save_pretrained"):
            self.vision_encoder.model.save_pretrained(os.path.join(save_directory, "vision_encoder"))

        # Save visual projection weights and config if present
        if hasattr(self.vision_encoder, "visual_projection"):
            torch.save(self.vision_encoder.visual_projection.state_dict(), os.path.join(save_directory, "vision_encoder/visual_projection.bin"))
            vision_config = self.vision_encoder.config.to_dict() if hasattr(self.vision_encoder, "config") else {}
            vision_config["visual_projection"] = {
                "in_features": self.vision_encoder.visual_projection.in_features,
                "out_features": self.vision_encoder.visual_projection.out_features,
                "bias": self.vision_encoder.visual_projection.bias is not None
            }
            with open(os.path.join(save_directory, "vision_encoder/proj_config.json"), "w", encoding="utf-8") as f:
                json.dump(vision_config, f, indent=2)

        # Save text encoder weights/config using HuggingFace method if available
        if hasattr(self.text_encoder.model, "save_pretrained"):
            self.text_encoder.model.save_pretrained(os.path.join(save_directory, "text_encoder"))


        # Save text projection weights and config if present
        if hasattr(self.text_encoder, "projection"):
            torch.save(self.text_encoder.projection.state_dict(), os.path.join(save_directory, "text_encoder/text_projection.bin"))
            text_config = self.text_encoder.config.to_dict() if hasattr(self.text_encoder, "config") else {}
            text_config["projection"] = {
                "in_features": self.text_encoder.projection.in_features,
                "out_features": self.text_encoder.projection.out_features,
                "bias": self.text_encoder.projection.bias is not None
            }
            with open(os.path.join(save_directory, "text_encoder/proj_config.json"), "w", encoding="utf-8") as f:
                json.dump(text_config, f, indent=2)

        # # # Save tokenizer and processor if provided
        if self.text_encoder.tokenizer is not None:
            self.text_encoder.tokenizer.save_pretrained(os.path.join(save_directory, "text_encoder"))
        if self.vision_encoder.processor is not None:
            self.vision_encoder.processor.save_pretrained(os.path.join(save_directory, "vision_encoder"))

        # Save combined config
        combined_config = {
            "vision_encoder": self.vision_encoder.config.to_dict() if hasattr(self.vision_encoder, "config") else {},
            "text_encoder": self.text_encoder.config.to_dict() if hasattr(self.text_encoder, "config") else {}
        }
        with open(os.path.join(save_directory, "config.json"), "w", encoding="utf-8") as f:
            json.dump(combined_config, f, indent=2)


if __name__ == "__main__":

    from sentence_transformers import util

    # Paths to pretrained models
    pretrained_img_model_path = r"pretrained_models/sentence-transformers--clip-ViT-B-32"
    pretrained_text_model_path = r"pretrained_models/sentence-transformers--clip-ViT-B-32-multilingual-v1"


    # Paths to finetuned models
    # finetuned_text_model_path = r"output/finetuned_baseline/best_model/text_encoder"
    # finetuned_img_model_path = r"output/finetuned_baseline/best_model/vision_encoder"
    finetuned_text_model_path = r"output/disentangled_clip_loss_test/best_model/text_encoder"
    finetuned_img_model_path = r"output/disentangled_clip_loss_test/best_model/vision_encoder"

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

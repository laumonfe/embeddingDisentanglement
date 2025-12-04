import torch
import torch.nn as nn
from transformers import  DistilBertTokenizer, CLIPProcessor
import json


# Load projection config
with open("pretrained_models/sentence-transformers--clip-ViT-B-32-multilingual-v1/2_Dense/config.json") as f:
    proj_cfg = json.load(f)

# Create projection layer
text_model_path = "pretrained_models/sentence-transformers--clip-ViT-B-32-multilingual-v1"

# Load projection weights and remap keys if necessary
PROJ_WEIGHTS_PATH = "pretrained_models/sentence-transformers--clip-ViT-B-32-multilingual-v1/2_Dense/pytorch_model.bin"
proj_weights = torch.load(PROJ_WEIGHTS_PATH)
if "linear.weight" in proj_weights:
    proj_weights["weight"] = proj_weights.pop("linear.weight")
if "linear.bias" in proj_weights:
    proj_weights["bias"] = proj_weights.pop("linear.bias")
projection = nn.Linear(proj_cfg["in_features"], proj_cfg["out_features"], bias=proj_cfg["bias"])
projection.load_state_dict(proj_weights)

def project_DistilBert_to_CLIP(text, text_model):
    device = next(text_model.parameters()).device
    text_tokenizer = DistilBertTokenizer.from_pretrained(text_model_path)
    tokens = text_tokenizer([text], return_tensors="pt", padding=True, truncation=True)
    tokens = {k: v.to(device) for k, v in tokens.items()}  # Move all tensors to device
    with torch.no_grad():
        outputs = text_model(**tokens)
        pooled = outputs.last_hidden_state.mean(dim=1)
        emb = projection.to(device)(pooled)
    return emb.cpu().numpy()[0]  # Shape: (512,)



# def clip_encode(model,  inputs,  modality="text"):
#     device = next(model.parameters()).device
#     if modality == "text":
#         return project_DistilBert_to_CLIP(inputs, model)
#     elif modality == "image":
#         processor = CLIPProcessor.from_pretrained(model.name_or_path)
#         processed = processor(images=inputs, return_tensors="pt")
#         with torch.no_grad():
#             vision_outputs = model.vision_model(processed["pixel_values"].to(device))
#             pooled = vision_outputs.pooler_output.to(device)
#             emb = model.visual_projection(pooled)
#         return emb.cpu().numpy()[0]
#     else:
#         raise ValueError("modality must be 'text' or 'image'")
def clip_encode(model, inputs, modality="text"):
    device = next(model.parameters()).device
    if modality == "text":
        # Add debug print for text input
        return project_DistilBert_to_CLIP(inputs, model)
    elif modality == "image":
        processor = CLIPProcessor.from_pretrained(model.name_or_path)
        processed = processor(images=inputs, return_tensors="pt")
        pixel_values = processed["pixel_values"]
        pixel_values = pixel_values.to(device)
        with torch.no_grad():
            vision_outputs = model.vision_model(pixel_values)
            pooled = vision_outputs.pooler_output
            pooled = pooled.to(device)
            emb = model.visual_projection(pooled)
        return emb.cpu().numpy()[0]
    else:
        raise ValueError("modality must be 'text' or 'image'")
import json
import torch
import torch.nn as nn
from transformers import DistilBertModel


def load_distilbert_with_projection(model_path, proj_config_path, proj_weights_path):
    # Load projection config
    with open(proj_config_path) as f:
        proj_cfg = json.load(f)

    # Load DistilBERT model
    text_model = DistilBertModel.from_pretrained(model_path)

    # Load projection weights and remap keys if necessary
    proj_weights = torch.load(proj_weights_path)
    if "linear.weight" in proj_weights:
        proj_weights["weight"] = proj_weights.pop("linear.weight")
    if "linear.bias" in proj_weights:
        proj_weights["bias"] = proj_weights.pop("linear.bias")
    projection = nn.Linear(proj_cfg["in_features"], proj_cfg["out_features"], bias=proj_cfg["bias"])
    projection.load_state_dict(proj_weights)

    # Attach projection layer to text_model
    text_model.projection = projection
    return text_model
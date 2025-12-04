import os
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

def load_distilbert_with_projection_finetuned(model_path, device="cpu"):
    """
    Loads a finetuned DistilBertModel and attaches the projection layer using config and weights from model_path.
    Ensures all model components are on the specified device.
    """
    import os
    import json
    import torch
    import torch.nn as nn
    from transformers import DistilBertModel

    # Paths for config and weights
    proj_config_path = os.path.join(model_path, "config.json")
    proj_weights_path = os.path.join(model_path, "text_projection.bin")

    # Load projection config
    with open(proj_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    proj_cfg = config.get("projection", None)
    if proj_cfg is None:
        raise ValueError("projection config not found in text_encoder/config.json")

    # Load DistilBERT model
    text_model = DistilBertModel.from_pretrained(model_path).to(device)

    # Load projection weights and remap keys if necessary
    proj_weights = torch.load(proj_weights_path, map_location=device)
    if "linear.weight" in proj_weights:
        proj_weights["weight"] = proj_weights.pop("linear.weight")
    if "linear.bias" in proj_weights:
        proj_weights["bias"] = proj_weights.pop("linear.bias")
    projection = nn.Linear(proj_cfg["in_features"], proj_cfg["out_features"], bias=proj_cfg["bias"]).to(device)
    projection.load_state_dict(proj_weights)

    # Attach projection layer to text_model
    text_model.projection = projection
    return text_model

def load_vision_with_projection(vision_model_dir, device="cpu"):
    """
    Loads a CLIPVisionModel and attaches the visual_projection layer from saved files.
    Ensures all model components are on the specified device.
    """
    import torch
    import json
    import torch.nn as nn
    from transformers import CLIPVisionModel

    # Load vision encoder
    vision_model = CLIPVisionModel.from_pretrained(vision_model_dir).to(device)

    # Load projection weights and config
    proj_weights_path = os.path.join(vision_model_dir, "visual_projection.bin")
    config_path = os.path.join(vision_model_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    proj_cfg = config.get("visual_projection", None)
    if proj_cfg is None:
        raise ValueError("visual_projection config not found in vision_encoder/config.json")

    proj_weights = torch.load(proj_weights_path, map_location=device)
    visual_projection = nn.Linear(
        proj_cfg["in_features"], proj_cfg["out_features"], bias=proj_cfg["bias"]
    ).to(device)
    visual_projection.load_state_dict(proj_weights)
    vision_model.visual_projection = visual_projection
    return vision_model

def load_pretrained_CLIP_vision(model_dir):
    from transformers import CLIPModel, CLIPVisionModel
    clip_model = CLIPModel.from_pretrained(model_dir)
    vision_model =  CLIPVisionModel.from_pretrained(model_dir)
    vision_model.visual_projection = clip_model.visual_projection
    return vision_model
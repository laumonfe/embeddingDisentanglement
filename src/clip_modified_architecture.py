import torch
from torch import nn
from transformers import CLIPModel, CLIPProcessor

# Load CLIP model and processor
model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# Access the vision and text encoder modules
vision_encoder = clip_model.vision_model
text_encoder = clip_model.text_model

# Example: print all layers of the vision encoder
print("Vision Encoder Layers:")
for name, module in vision_encoder.named_modules():
    print(name, module)

# Example: print all layers of the text encoder
print("\nText Encoder Layers:")
for name, module in text_encoder.named_modules():
    print(name, module)

# Example: add a custom layer after the pooled output of the vision encoder
class CustomCLIP(nn.Module):
    def __init__(self, vision_encoder, text_encoder, embed_dim=512):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        # Example: add a new linear layer after vision encoder
        self.custom_vision_head = nn.Linear(vision_encoder.config.hidden_size, embed_dim)
        # Example: add a new linear layer after text encoder
        self.custom_text_head = nn.Linear(text_encoder.config.hidden_size, embed_dim)

    def forward(self, pixel_values, input_ids, attention_mask):
        # Vision encoding
        vision_outputs = self.vision_encoder(pixel_values)
        vision_pooled = vision_outputs.pooler_output  # [batch, hidden_size]
        vision_embeds = self.custom_vision_head(vision_pooled)  # [batch, embed_dim]

        # Text encoding
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_pooled = text_outputs.pooler_output  # [batch, hidden_size]
        text_embeds = self.custom_text_head(text_pooled)  # [batch, embed_dim]

        # Example: concatenate embeddings
        concat_embeds = torch.cat([vision_embeds, text_embeds], dim=1)  # [batch, embed_dim*2]
        return {
            "vision_embeds": vision_embeds,
            "text_embeds": text_embeds,
            "concat_embeds": concat_embeds
        }

# Instantiate your custom model with pretrained weights
custom_clip = CustomCLIP(vision_encoder, text_encoder)

# Example usage:
# inputs = processor(text=["a red dress"], images=Image.open("path/to/image.jpg"), return_tensors="pt", padding=True)
# outputs = custom_clip(
#     pixel_values=inputs["pixel_values"],
#     input_ids=inputs["input_ids"],
#     attention_mask=inputs["attention_mask"]
# )
# print(outputs["concat_embeds"].shape)

# Now you have full access to all layers and can modify, add, or replace them as needed.
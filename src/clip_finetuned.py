import torch
from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel, CLIPTextModel
from transformers import TrainingArguments, Trainer
from PIL import Image
import pandas as pd
from data_loader import CLIPDataset
from training_loop import CustomCLIPTrainer

# Load CSV
csv_path = "visualization_explorer/feidegger_visualization_data_valid.csv"
df = pd.read_csv(csv_path)

# Filter train split
train_df = df[df["mamba_split"] == "train"]
image_paths = train_df["image_path"].tolist()
texts = train_df["text"].tolist()

# Load CLIP model and processor
model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# Access vision and text transformer separately
vision_model = clip_model.vision_model
text_model = clip_model.text_model

# Example: encode one image and one text
sample_image = Image.open(image_paths[0]).convert("RGB")
sample_text = texts[0]

inputs = processor(text=[sample_text], images=sample_image, return_tensors="pt", padding=True)
with torch.no_grad():
    vision_outputs = vision_model(inputs["pixel_values"])
    text_outputs = text_model(inputs["input_ids"])

# You can now access the hidden states:
image_embeds = vision_outputs.last_hidden_state
text_embeds = text_outputs.last_hidden_state



train_dataset = CLIPDataset(image_paths, texts, processor)

def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    return {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "attention_mask": attention_mask
    }

training_args = TrainingArguments(
    output_dir="output/finetuned-clip-custom",
    per_device_train_batch_size=4,
    num_train_epochs=5,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    remove_unused_columns=False,
    fp16=True,
    report_to="none"
)



trainer = CustomCLIPTrainer(
    model=clip_model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn,
)

trainer.train()
clip_model.save_pretrained("output/finetuned-clip-custom")
import torch
from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel, CLIPTextModel
from transformers import TrainingArguments, Trainer
from PIL import Image
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses

# # Load CSV
# csv_path = r"data/embeddings/feidegger_visualization_data.csv"
# df = pd.read_csv(csv_path)

# # Filter train split
# train_df = df[df["split"] == "train"]
# image_paths = train_df["image_path"].tolist()
# texts = train_df["text"].tolist()

# Load CLIP model and processor

img_model =r'pretrained_models\models--sentence-transformers--clip-ViT-B-32'
text_model =r'pretrained_models\models--sentence-transformers--clip-ViT-B-32-multilingual-v1'

# Load models
image_model = CLIPModel.from_pretrained(img_model)
image_processor = CLIPProcessor.from_pretrained(img_model)

text_model = CLIPModel.from_pretrained(text_model)
text_processor = CLIPProcessor.from_pretrained(text_model)
# Access vision and text transformer separately
vision_model = image_model.vision_model
text_model = text_model.text_model

print("VISION")
print(vision_model)
print("TEXT")
print(text_model)

# # Example: encode one image and one text
# sample_image = Image.open(image_paths[0]).convert("RGB")
# sample_text = texts[0]

# inputs = processor(text=[sample_text], images=sample_image, return_tensors="pt", padding=True)
# with torch.no_grad():
#     vision_outputs = vision_model(inputs["pixel_values"])
#     text_outputs = text_model(inputs["input_ids"])

# # You can now access the hidden states:
# image_embeds = vision_outputs.last_hidden_state
# text_embeds = text_outputs.last_hidden_state

# # For finetuning, you need to create a custom dataset and Trainer
# class CLIPDataset(torch.utils.data.Dataset):
#     def __init__(self, image_paths, texts, processor):
#         self.image_paths = image_paths
#         self.texts = texts
#         self.processor = processor

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         image = Image.open(self.image_paths[idx]).convert("RGB")
#         text = self.texts[idx]
#         inputs = self.processor(text=[text], images=image, return_tensors="pt", padding=True)
#         item = {key: val.squeeze(0) for key, val in inputs.items()}
#         return item

# train_dataset = CLIPDataset(image_paths, texts, processor)

# def collate_fn(batch):
#     input_ids = torch.stack([item["input_ids"] for item in batch])
#     pixel_values = torch.stack([item["pixel_values"] for item in batch])
#     attention_mask = torch.stack([item["attention_mask"] for item in batch])
#     return {
#         "input_ids": input_ids,
#         "pixel_values": pixel_values,
#         "attention_mask": attention_mask
#     }

# training_args = TrainingArguments(
#     output_dir="output/finetuned-clip-custom",
#     per_device_train_batch_size=4,
#     num_train_epochs=5,
#     logging_steps=10,
#     save_steps=100,
#     save_total_limit=2,
#     remove_unused_columns=False,
#     fp16=True,
#     report_to="none"
# )

# # Define a custom Trainer to access both vision and text models
# class CustomCLIPTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
#         outputs = model(**inputs)
#         # CLIPModel returns logits_per_image and logits_per_text
#         loss = outputs.loss
#         return (loss, outputs) if return_outputs else loss

# trainer = CustomCLIPTrainer(
#     model=clip_model,
#     args=training_args,
#     train_dataset=train_dataset,
#     data_collator=collate_fn,
# )

# trainer.train()
# clip_model.save_pretrained("output/finetuned-clip-custom")
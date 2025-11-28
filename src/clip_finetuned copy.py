import torch
import torch.nn as nn
import pandas as pd
from data_loader import CLIPDataset
from torch.utils.data import DataLoader
from utils import load_distilbert_with_projection
from transformers import  CLIPModel, DistilBertTokenizer, CLIPProcessor
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


class FinetuneCLIP(nn.Module):
    def __init__(self, vision_encoder, visual_projection, text_encoder):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.visual_projection = visual_projection
        self.text_encoder = text_encoder

    def forward(self, pixel_values, input_ids, attention_mask):
        # Vision encoding
        vision_outputs = self.vision_encoder(pixel_values)
        vision_pooled = vision_outputs.pooler_output
        vision_embeds = self.visual_projection(vision_pooled)

        # Text encoding
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_pooled = text_outputs.last_hidden_state.mean(dim=1)  # <-- FIXED LINE
        text_embeds = self.text_encoder.projection(text_pooled)
        return {
            "vision_embeds": vision_embeds,
            "text_embeds": text_embeds
        }



def contrastive_loss(image_embeds, text_embeds, temperature=0.07):
    # Normalize
    image_embeds = nn.functional.normalize(image_embeds, dim=-1)
    text_embeds = nn.functional.normalize(text_embeds, dim=-1)
    logits = image_embeds @ text_embeds.t() / temperature
    labels = torch.arange(logits.size(0)).to(logits.device)
    loss_img = nn.CrossEntropyLoss()(logits, labels)
    loss_txt = nn.CrossEntropyLoss()(logits.t(), labels)
    return (loss_img + loss_txt) / 2

def collate_fn(batch):
    input_ids = pad_sequence([item["input_ids"] for item in batch], batch_first=True, padding_value=0)
    attention_mask = pad_sequence([item["attention_mask"] for item in batch], batch_first=True, padding_value=0)
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    return {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "attention_mask": attention_mask
    }

if __name__ == "__main__":


    # Load CSV
    csv_path = r"data/embeddings/feidegger_visualization_data.csv"
    df = pd.read_csv(csv_path)

    # Filter train split
    train_df = df[df["split"] == "train"]
    train_df[:100]
    
    # image_paths = train_df["image_path"].tolist()
    # texts = train_df["text"].tolist()

    # Load CLIP model and processor

    img_model_path = r"pretrained_models/sentence-transformers--clip-ViT-B-32"
    text_model_path = r"pretrained_models/sentence-transformers--clip-ViT-B-32-multilingual-v1"
    text_model_config_path = r"pretrained_models/sentence-transformers--clip-ViT-B-32-multilingual-v1/2_Dense/config.json"
    text_model_projection_weights_path = r"pretrained_models/sentence-transformers--clip-ViT-B-32-multilingual-v1/2_Dense/pytorch_model.bin"


    # Load vision model submodule from CLIP
    CLIP_model = CLIPModel.from_pretrained(img_model_path)
    vision_model = CLIP_model.vision_model
    visual_projection = CLIP_model.visual_projection

    clip_preprocesor = CLIPProcessor.from_pretrained(img_model_path)

    # Load multilingual text model 
    text_model = load_distilbert_with_projection(text_model_path, text_model_config_path, text_model_projection_weights_path)
    text_tokenizer = DistilBertTokenizer.from_pretrained(text_model_path)

    # Prepare DataLoader
    train_dataset = CLIPDataset(train_df, clip_preprocesor,  text_tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    model = FinetuneCLIP(vision_model, visual_projection, text_model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    writer = SummaryWriter("output/tensorboard_logs")

    # model.train()
    # model.train()
    # global_step = 0
    # for epoch in range(2):
    #     for batch in train_loader:
    #         optimizer.zero_grad()
    #         outputs = model(
    #             pixel_values=batch["pixel_values"],
    #             input_ids=batch["input_ids"],
    #             attention_mask=batch["attention_mask"]
    #         )
    #         loss = contrastive_loss(outputs["vision_embeds"], outputs["text_embeds"])
    #         loss.backward()
    #         optimizer.step()
    #         print(f"Loss: {loss.item()}")
    #         writer.add_scalar("Loss/train", loss.item(), global_step)
    #         global_step += 1

    # writer.close()

    num_epochs = 2
    model.train()
    global_step = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            outputs = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            loss = contrastive_loss(outputs["vision_embeds"], outputs["text_embeds"])
            loss.backward()
            optimizer.step()
            writer.add_scalar("Loss/train", loss.item(), global_step)
            epoch_loss += loss.item()
            global_step += 1

            progress_bar.set_postfix({"batch_loss": loss.item()})

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished. Average loss: {avg_epoch_loss:.4f}")

    writer.close()


    torch.save(model.state_dict(), "output/finetuned_clip_combined.pt")
    torch.save(model.text_encoder.state_dict(), "output/finetuned_clip_text_encoder.pt")
    torch.save(model.vision_encoder.state_dict(), "output/finetuned_clip_vision_encoder.pt")

    model.save_pretrained("output/finetuned_clip")


    from compute_embeddings import compute_embeddings, load_embeddings
    from german_retrieval import get_split_embeddings
    from german_retrieval_custom import retrieve_images_by_image, retrieve_images_by_text, plot_images, compute_image_embeddings, compute_text_embeddings
    import os 
    test_df = df[df["split"] == "test"]

    finetuned_img_emb = r"data/embeddings/data/embeddings/finetuned_clip-ViT-B-32-multilingual-v1/finetuned_clip_image_embeddings.npy"
    finetuned_text_emb = r"data/embeddings/data/embeddings/finetuned_clip-ViT-B-32-multilingual-v1/finetuned_clip_text_embeddings.npy"

    compute_embeddings(model.text_encoder, model.vision_encoder, test_df, finetuned_img_emb, finetuned_text_emb)
    image_embeddings = load_embeddings(finetuned_img_emb)
    text_embeddings = load_embeddings(finetuned_text_emb)
    test_df, test_img_emb, test_txt_emb = get_split_embeddings(df, image_embeddings, text_embeddings, "test")

    query = "red dress"
    ########### Same QUery Only in the test split ###########
    print("Text-to-Image Retrieval Example Test:")
    results = retrieve_images_by_text(query, text_model, test_img_emb, test_df,  top_k=5)
    plot_images(results, "Text-to-Image Retrieval (M-CLIP)", query=query, query_type="text")

    print("\nImage-to-Image Retrieval Example Test:")
    example_image = results[0][0]
    print(f"Using example image: {example_image}")
    results = retrieve_images_by_image(example_image, model.vision_encoder, test_img_emb, test_df, top_k=5)
    plot_images(results, "Image-to-Image Retrieval (M-CLIP)", query=example_image, query_type="image")



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
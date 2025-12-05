from numpy._core.multiarray import CLIP
import torch
import torch.nn as nn
import pandas as pd
from src.data_loader import CLIPDataset
from torch.utils.data import DataLoader
from src.utils import load_distilbert_with_projection, load_distilbert_with_projection_finetuned, load_vision_with_projection, load_pretrained_CLIP_vision
from transformers import  CLIPModel, DistilBertTokenizer, CLIPProcessor, CLIPVisionModel
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import os
import json






class FinetuneCLIP(nn.Module):
    def __init__(self, vision_encoder, text_encoder):
        super().__init__()
        self.vision_encoder = vision_encoder
        #self.visual_projection = visual_projection
        self.text_encoder = text_encoder

    def forward(self, pixel_values, input_ids, attention_mask):
        # Vision encoding
        vision_outputs = self.vision_encoder(pixel_values)
        vision_pooled = vision_outputs.pooler_output
        vision_embeds = self.vision_encoder.visual_projection(vision_pooled)

        # Text encoding
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_pooled = text_outputs.last_hidden_state.mean(dim=1)  # <-- FIXED LINE
        text_embeds = self.text_encoder.projection(text_pooled)
        return {
            "vision_embeds": vision_embeds,
            "text_embeds": text_embeds
        }
    
    def save_from_pretrained(self, save_directory, text_tokenizer=None, image_processor=None):
        os.makedirs(save_directory, exist_ok=True)
        os.makedirs(os.path.join(save_directory, "vision_encoder"), exist_ok=True)
        os.makedirs(os.path.join(save_directory, "text_encoder"), exist_ok=True)

        # Save combined model weights
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

        # Save vision encoder weights/config using HuggingFace method if available
        if hasattr(self.vision_encoder, "save_pretrained"):
            self.vision_encoder.save_pretrained(os.path.join(save_directory, "vision_encoder"))
        else:
            print("Saving vision encoder without HuggingFace method.")
            torch.save(self.vision_encoder.state_dict(), os.path.join(save_directory, "vision_encoder/pytorch_model.bin"))
            vision_config = self.vision_encoder.config.to_dict() if hasattr(self.vision_encoder, "config") else {}
            with open(os.path.join(save_directory, "vision_encoder/config.json"), "w", encoding="utf-8") as f:
                json.dump(vision_config, f, indent=2)

        # Save visual projection weights and config if present
        if hasattr(self.vision_encoder, "visual_projection"):
            torch.save(self.vision_encoder.visual_projection.state_dict(), os.path.join(save_directory, "vision_encoder/visual_projection.bin"))
            vision_config = self.vision_encoder.config.to_dict() if hasattr(self.vision_encoder, "config") else {}
            vision_config["visual_projection"] = {
                "in_features": self.vision_encoder.visual_projection.in_features,
                "out_features": self.vision_encoder.visual_projection.out_features,
                "bias": self.vision_encoder.visual_projection.bias is not None
            }
            with open(os.path.join(save_directory, "vision_encoder/config.json"), "w", encoding="utf-8") as f:
                json.dump(vision_config, f, indent=2)

        # Save text encoder weights/config using HuggingFace method if available
        if hasattr(self.text_encoder, "save_pretrained"):
            self.text_encoder.save_pretrained(os.path.join(save_directory, "text_encoder"))
        else:
            torch.save(self.text_encoder.state_dict(), os.path.join(save_directory, "text_encoder/pytorch_model.bin"))
            text_config = self.text_encoder.config.to_dict() if hasattr(self.text_encoder, "config") else {}
            with open(os.path.join(save_directory, "text_encoder/config.json"), "w", encoding="utf-8") as f:
                json.dump(text_config, f, indent=2)

        # Save text projection weights and config if present
        if hasattr(self.text_encoder, "projection"):
            torch.save(self.text_encoder.projection.state_dict(), os.path.join(save_directory, "text_encoder/text_projection.bin"))
            text_config = self.text_encoder.config.to_dict() if hasattr(self.text_encoder, "config") else {}
            text_config["projection"] = {
                "in_features": self.text_encoder.projection.in_features,
                "out_features": self.text_encoder.projection.out_features,
                "bias": self.text_encoder.projection.bias is not None
            }
            with open(os.path.join(save_directory, "text_encoder/config.json"), "w", encoding="utf-8") as f:
                json.dump(text_config, f, indent=2)

        # Save tokenizer and processor if provided
        if text_tokenizer is not None:
            text_tokenizer.save_pretrained(os.path.join(save_directory, "text_encoder"))
        if image_processor is not None:
            image_processor.save_pretrained(os.path.join(save_directory, "vision_encoder"))

        # Save combined config
        combined_config = {
            "vision_encoder": self.vision_encoder.config.to_dict() if hasattr(self.vision_encoder, "config") else {},
            "text_encoder": self.text_encoder.config.to_dict() if hasattr(self.text_encoder, "config") else {}
        }
        with open(os.path.join(save_directory, "config.json"), "w", encoding="utf-8") as f:
            json.dump(combined_config, f, indent=2)

def disentanglement_loss(embeddings):
    """
    Penalizes off-diagonal covariance (total correlation) to encourage disentanglement.
    embeddings: (batch_size, dim)
    """
    # Center embeddings
    mean = embeddings.mean(dim=0, keepdim=True)
    centered = embeddings - mean
    # Covariance matrix
    cov = (centered.T @ centered) / (embeddings.size(0) - 1)
    # Zero diagonal, sum absolute off-diagonal
    off_diag = cov - torch.diag(torch.diag(cov))
    return off_diag.abs().sum()

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


def save_training_config(save_directory, num_epochs, optimizer, batch_size, learning_rate, device, additional_params=None):
    """
    Save training configuration to a JSON file.
    """
    config = {
        "num_epochs": num_epochs,
        "optimizer": optimizer.__class__.__name__,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "device": str(device),
        "num_training_samples": len(train_loader.dataset)  # Add number of training samples
    }
    if additional_params:
        config.update(additional_params)
    with open(os.path.join(save_directory, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":

    import os 
    from compute_embeddings import compute_embeddings, load_embeddings
    from german_retrieval import retrieve_images_by_image, retrieve_images_by_text, plot_images
    
    # Load CSV
    csv_path = r"data/embeddings/feidegger_visualization_data.csv"
    df = pd.read_csv(csv_path)

    # Filter train split
    train_df = df[df["split"] == "train"]
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Paths to pretrained models
    img_model_path = r"pretrained_models/sentence-transformers--clip-ViT-B-32"
    img_model_config_path = r"pretrained_models/sentence-transformers--clip-ViT-B-32/config.json"
    text_model_path = r"pretrained_models/sentence-transformers--clip-ViT-B-32-multilingual-v1"
    text_model_config_path = r"pretrained_models/sentence-transformers--clip-ViT-B-32-multilingual-v1/2_Dense/config.json"
    text_model_projection_weights_path = r"pretrained_models/sentence-transformers--clip-ViT-B-32-multilingual-v1/2_Dense/pytorch_model.bin"

    output_directory = r"output/finetuned_baseline"

    learning_rate = 1e-4
    batch_size = 16

    vision_model = load_pretrained_CLIP_vision(img_model_path)
    vision_model = vision_model.to(device)
    clip_preprocesor = CLIPProcessor.from_pretrained(img_model_path)

    # Load multilingual text model 
    text_model = load_distilbert_with_projection(text_model_path, text_model_config_path, text_model_projection_weights_path)
    text_model = text_model.to(device)
    text_tokenizer = DistilBertTokenizer.from_pretrained(text_model_path)

    # Prepare DataLoader
    train_dataset = CLIPDataset(train_df, clip_preprocesor,  text_tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = FinetuneCLIP(vision_model, text_model)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    writer = SummaryWriter(os.path.join(output_directory,"tensorboard_logs"), comment= "Finetune_CLIP")

    num_epochs = 100
    model.train()
    global_step = 0

    best_loss = float("inf")
    best_model_dir = None

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            # Move batch tensors to device
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            loss = contrastive_loss(outputs["vision_embeds"], outputs["text_embeds"])
            # Add disentanglement loss for both modalities
            lambda_dis = 0.01  # Weight for disentanglement penalty (tune as needed)
            dis_loss_img = disentanglement_loss(outputs["vision_embeds"])
            dis_loss_txt = disentanglement_loss(outputs["text_embeds"])
            loss = loss + lambda_dis * (dis_loss_img + dis_loss_txt)

            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            writer.add_scalar("Loss/train", loss.item(), global_step)
            epoch_loss += loss.item()
            global_step += 1

            progress_bar.set_postfix({"batch_loss": loss.item()})

        # Save model at each epoch
        step_save_dir = os.path.join(output_directory, f"epoch_{epoch}")
        model.save_from_pretrained(step_save_dir, text_tokenizer=text_tokenizer, image_processor=clip_preprocesor)

        # Save best model so far
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model_dir = os.path.join(output_directory, "best_model")
            model.save_from_pretrained(best_model_dir, text_tokenizer=text_tokenizer, image_processor=clip_preprocesor)

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished. Average loss: {avg_epoch_loss:.4f}")

    print(f"Best model saved at {best_model_dir} with loss {best_loss:.4f}")

    writer.close()

    model.save_from_pretrained(output_directory, text_tokenizer=text_tokenizer, image_processor=clip_preprocesor)
     
    # Example usage after training:
    save_training_config(
        output_directory,
        num_epochs=num_epochs,
        optimizer=optimizer,
        batch_size=train_loader.batch_size,
        learning_rate=learning_rate,
        device=device,
        additional_params={"training ": global_step}
    )


    test_df = df[df["split"] == "test"]

    finetuned_img_emb = r"data/embeddings/data/embeddings/finetuned_clip-ViT-B-32-multilingual-v1/finetuned_clip_image_embeddings.npy"
    finetuned_text_emb = r"data/embeddings/data/embeddings/finetuned_clip-ViT-B-32-multilingual-v1/finetuned_clip_text_embeddings.npy"

    img_model = load_vision_with_projection(os.path.join(output_directory, "vision_encoder"))
    text_model = load_distilbert_with_projection_finetuned(os.path.join(output_directory, "text_encoder"))

    compute_embeddings(text_model, img_model, test_df, finetuned_img_emb, finetuned_text_emb)
    image_embeddings = load_embeddings(finetuned_img_emb)
    text_embeddings = load_embeddings(finetuned_text_emb)
    # # #test_df, test_img_emb, test_txt_emb = get_split_embeddings(df, image_embeddings, text_embeddings, "test")

    query = "red dress"
    ########### Same QUery Only in the test split ###########
    print("Text-to-Image Retrieval Example Test:")
    results = retrieve_images_by_text(query, text_model, image_embeddings, test_df,  top_k=5)
    plot_images(results, "Text-to-Image Retrieval (M-CLIP)", query=query, query_type="text")

    print("\nImage-to-Image Retrieval Example Test:")
    example_image = results[0][0]
    print(f"Using example image: {example_image}")
    results = retrieve_images_by_image(example_image,img_model, image_embeddings, test_df, top_k=5)
    plot_images(results, "Image-to-Image Retrieval (M-CLIP)", query=example_image, query_type="image")
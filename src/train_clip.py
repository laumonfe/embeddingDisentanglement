
import os
import json
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter

# Custom imports 
from src.models import FinetuneCLIP
from src.data_loader import CLIPDataset
from src.losses import disentangled_clip_loss, contrastive_loss
from src.models import PretrainedCLIPVision, PretrainedDistilBert, FinetuneCLIP


def train(model, train_loader, optimizer, writer, output_directory, num_epochs,
    device, loss_fn, save_every=10, patience=3):
    
    best_loss = float("inf")
    best_model_dir = None
    global_step = 0
    best_val_loss = float("inf")
    epochs_no_improve = 0
    stopped_epoch = num_epochs

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            loss = loss_fn(outputs["vision_embeds"], outputs["text_embeds"])
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            writer.add_scalar("Loss/train", loss.item(), global_step)
            epoch_loss += loss.item()
            global_step += 1
            progress_bar.set_postfix({"batch_loss": loss.item()})

        if (epoch + 1) % save_every == 0:
            step_save_dir = os.path.join(output_directory, f"epoch_{epoch+1}")
            model.save_from_pretrained(step_save_dir)

        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_val_loss = validate(model, val_loader, device, loss_fn)

        print(f"Epoch {epoch+1} finished. Train loss: {avg_epoch_loss:.4f} | Val loss: {avg_val_loss:.4f}")
        writer.add_scalar("Loss/val", avg_val_loss, epoch)

        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_dir = os.path.join(output_directory, "best_model")
            model.save_from_pretrained(best_model_dir)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                stopped_epoch = epoch + 1
                print(f"Early stopping triggered after {stopped_epoch} epochs. Best model saved at {best_model_dir} with val loss {best_val_loss:.4f}")
                writer.close()
                return stopped_epoch

    print(f"Best model saved at {best_model_dir} with val loss {best_val_loss:.4f}")
    writer.close()
    return stopped_epoch

def validate(model, val_loader, device, loss_fn):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            loss = loss_fn(outputs["vision_embeds"], outputs["text_embeds"])
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    model.train()
    return avg_val_loss

def collate_fn(batch):
    input_ids = pad_sequence([item["input_ids"] for item in batch], batch_first=True, padding_value=0)
    attention_mask = pad_sequence([item["attention_mask"] for item in batch], batch_first=True, padding_value=0)
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    return {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "attention_mask": attention_mask
    }


def save_training_config(save_directory, num_epochs, optimizer, batch_size, learning_rate, device, train_loader, additional_params=None):
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

    parser = argparse.ArgumentParser(description="Train Disentangled CLIP model on FEIDEGGER dataset.")
    parser.add_argument("--csv_path",type=str,default="dataset/feidegger_visualization_data.csv",help="Path to the CSV file containing image paths and text descriptions.")
    parser.add_argument("--model_kind", choices=["finetuned", "disentangled"], default="disentangled", help="Which model to use: finetuned or disentangled.")
    parser.add_argument("--pretrained_dir", type=str, default="pretrained_models", help="Directory containing pretrained CLIP and DistilBERT models.")
    parser.add_argument("--output_directory", type=str, default=None, help="Directory to save the trained model and logs.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")  
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--save_every", type=int, default=10, help="Save model every N epochs.")
    parser.add_argument("--patience", type=int, default=3, help="Patience for early stopping.")

    args = parser.parse_args()


    df = pd.read_csv(args.csv_path)
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained models
    pretrained_img_model_path = os.path.join(args.pretrained_dir, "sentence-transformers--clip-ViT-B-32")
    pretrained_text_model_path = os.path.join(args.pretrained_dir, "sentence-transformers--clip-ViT-B-32-multilingual-v1")

    image_encoder = PretrainedCLIPVision(pretrained_img_model_path, device)
    text_encoder = PretrainedDistilBert(pretrained_text_model_path, device)


    # Prepare DataLoader
    train_dataset = CLIPDataset(train_df, image_encoder.processor,  text_encoder.tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    val_dataset = CLIPDataset(val_df, image_encoder.processor, text_encoder.tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = FinetuneCLIP(image_encoder, text_encoder)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    if args.model_kind == "disentangled":
        loss = disentangled_clip_loss
    elif args.model_kind == "finetuned":
        loss = contrastive_loss


    if args.output_directory is None:
         output_dir = f"output/{args.model_kind}_clip"
    else:
         output_dir = args.output_directory

    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(output_dir,"tensorboard_logs"))

    stopped_epoch = train(model, train_loader, optimizer, writer, output_dir, args.num_epochs, device, loss, args.save_every, args.patience)
    
    save_training_config(output_dir, num_epochs=stopped_epoch, optimizer=optimizer, batch_size=train_loader.batch_size,
    learning_rate=args.learning_rate, device=device, train_loader=train_loader, additional_params={"loss": loss.__name__, "early_stopp_epoch": stopped_epoch})

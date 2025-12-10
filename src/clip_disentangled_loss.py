
import os
import torch
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Custom imports 
from src.models import FinetuneCLIP
from src.data_loader import CLIPDataset
from src.losses import disentangled_clip_loss
from src.training_loop import train_disentangled_clip
from src.utils import save_training_config, collate_fn
from src.models import PretrainedCLIPVision, PretrainedDistilBert, FinetuneCLIP


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train Disentangled CLIP model on FEIDEGGER dataset.")
    parser.add_argument("--csv_path",type=str,default="dataset/feidegger_visualization_data.csv",help="Path to the CSV file containing image paths and text descriptions.")
    parser.add_argument("--pretrained_dir", type=str, default="pretrained_models", help="Directory containing pretrained CLIP and DistilBERT models.")
    parser.add_argument("--output_directory", type=str, default="output/disentangled_clip_loss_test", help="Directory to save the trained model and logs.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")  
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--save_every", type=int, default=10, help="Save model every N epochs.")

    args = parser.parse_args()


    df = pd.read_csv(args.csv_path)
    train_df = df[df["split"] == "train"]
    train_df = train_df [:100]

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

    model = FinetuneCLIP(image_encoder, text_encoder)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    writer = SummaryWriter(os.path.join(args.output_directory,"tensorboard_logs"), comment= "Disentangled_CLIP_loss2")

    train_disentangled_clip(
        model, train_loader, optimizer, writer, args.output_directory, args.num_epochs, device, disentangled_clip_loss, args.save_every) 
            
    save_training_config(args.output_directory,num_epochs=args.num_epochs,optimizer=optimizer,batch_size=train_loader.batch_size,
        learning_rate=args.learning_rate,device=device, train_loader=train_loader)
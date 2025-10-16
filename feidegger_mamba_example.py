#!/usr/bin/env python
"""
Example: Training a Mamba Model on the FEIDEGGER dataset

This script demonstrates how to use the prepared FEIDEGGER dataset
to train a Mamba model for image-text understanding.

Note: This is an example script that assumes the dataset has been
prepared using feidegger_mamba_prep.py.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.models import resnet50
import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Import the dataset class from our preparation script
from feidegger_mamba_prep import FeideggerMambaDataset

# Note: For a real implementation, you would use a proper Mamba implementation
# like mamba-ssm, state-spaces, or similar libraries. This is a simplified example.

class SimplifiedMambaBlock(nn.Module):
    """
    A simplified version of a Mamba block for demonstration purposes.
    In a real implementation, you would use the actual Mamba SSM implementation.
    """
    def __init__(self, d_model, d_state=16, expand_factor=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand_factor = expand_factor
        d_inner = int(expand_factor * d_model)
        
        # Projection layers
        self.in_proj = nn.Linear(d_model, d_inner)
        self.out_proj = nn.Linear(d_inner, d_model)
        
        # Simplified SSM parameters (in real Mamba, these would be input-dependent)
        self.A = nn.Parameter(torch.randn(d_inner, d_state))
        self.B = nn.Parameter(torch.randn(d_inner, d_state))
        self.C = nn.Parameter(torch.randn(d_inner, d_state))
        
        # Layer norm
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # Input shape: (batch_size, seq_len, d_model)
        residual = x
        batch_size, seq_len, _ = x.shape
        
        # Layer norm
        x = self.norm(x)
        
        # Project to inner dimension
        x = self.in_proj(x)  # (batch_size, seq_len, d_inner)
        
        # Simplified SSM calculation
        # In a real Mamba implementation, this would use the proper SSM algorithm
        # with input-dependent parameters and parallel scan
        hidden = torch.zeros(batch_size, self.d_state, device=x.device)
        outputs = []
        
        for t in range(seq_len):
            # Simplified recurrence (not the actual Mamba algorithm)
            x_t = x[:, t, :]  # (batch_size, d_inner)
            
            # Update hidden state (simplified)
            hidden = hidden @ torch.diag_embed(torch.sigmoid(x_t @ self.A)) + x_t.unsqueeze(-1) @ self.B.unsqueeze(0)
            
            # Get output
            y_t = torch.bmm(hidden, self.C.unsqueeze(0).repeat(batch_size, 1, 1).transpose(1, 2))
            outputs.append(y_t)
            
        # Stack outputs
        output = torch.cat([o.unsqueeze(1) for o in outputs], dim=1)
        
        # Project back to model dimension
        output = self.out_proj(output)  # (batch_size, seq_len, d_model)
        
        # Residual connection
        return output + residual


class ImageTextMambaModel(nn.Module):
    """
    A model that combines image features and text tokens,
    and processes them using a simplified Mamba network.
    """
    def __init__(
        self, 
        image_encoder_dim=2048, 
        text_vocab_size=32000,
        text_embed_dim=512,
        d_model=512, 
        num_layers=4,
        max_seq_len=128
    ):
        super().__init__()
        
        # Image encoder (pretrained ResNet)
        self.image_encoder = resnet50(pretrained=True)
        self.image_encoder.fc = nn.Linear(self.image_encoder.fc.in_features, image_encoder_dim)
        
        # Image projection to match text dimension
        self.image_proj = nn.Linear(image_encoder_dim, d_model)
        
        # Text embedding
        self.text_embedding = nn.Embedding(text_vocab_size, text_embed_dim)
        self.text_proj = nn.Linear(text_embed_dim, d_model)
        
        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len + 1, d_model))
        
        # Mamba layers
        self.mamba_layers = nn.ModuleList([
            SimplifiedMambaBlock(d_model) for _ in range(num_layers)
        ])
        
        # Final classification/regression head
        self.head = nn.Linear(d_model, 1)  # For similarity score
        
    def encode_image(self, image):
        # Image shape: (batch_size, 3, H, W)
        with torch.no_grad():  # Freeze the image encoder
            image_features = self.image_encoder(image)  # (batch_size, image_encoder_dim)
        
        image_embed = self.image_proj(image_features).unsqueeze(1)  # (batch_size, 1, d_model)
        return image_embed
    
    def encode_text(self, input_ids, attention_mask):
        # Text shape: (batch_size, seq_len)
        text_embed = self.text_embedding(input_ids)  # (batch_size, seq_len, text_embed_dim)
        text_embed = self.text_proj(text_embed)  # (batch_size, seq_len, d_model)
        
        # Apply attention mask
        text_embed = text_embed * attention_mask.unsqueeze(-1)
        
        return text_embed
    
    def forward(self, image, input_ids, attention_mask):
        # Get embeddings
        image_embed = self.encode_image(image)  # (batch_size, 1, d_model)
        text_embed = self.encode_text(input_ids, attention_mask)  # (batch_size, seq_len, d_model)
        
        # Concatenate image embedding at the beginning of text sequence
        # This creates a sequence where the first token represents the image
        x = torch.cat([image_embed, text_embed], dim=1)  # (batch_size, 1+seq_len, d_model)
        
        # Add positional embeddings
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        
        # Apply Mamba layers
        for layer in self.mamba_layers:
            x = layer(x)
        
        # Use the final hidden state for prediction
        score = self.head(x[:, 0, :]).squeeze(-1)  # (batch_size,)
        return score


def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    
    for i, batch in enumerate(tqdm(train_loader, desc="Training")):
        # Get batch data
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Forward pass
        scores = model(images, input_ids, attention_mask)
        
        # Simplified contrastive loss
        # In a real scenario, you would use proper positive and negative pairs
        # This is just a placeholder for demonstration
        batch_size = images.size(0)
        labels = torch.arange(batch_size).to(device)
        loss = F.cross_entropy(scores.reshape(batch_size, -1), labels)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if i % 100 == 99:  # Print every 100 batches
            print(f'Batch {i+1}, Loss: {running_loss / 100:.4f}')
            running_loss = 0.0
    

def evaluate(model, val_loader, device):
    """Evaluate the model."""
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            # Get batch data
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            scores = model(images, input_ids, attention_mask)
            
            # Simplified loss
            batch_size = images.size(0)
            labels = torch.arange(batch_size).to(device)
            loss = F.cross_entropy(scores.reshape(batch_size, -1), labels)
            
            val_loss += loss.item()
    
    return val_loss / len(val_loader)


def main():
    parser = argparse.ArgumentParser(description='Train a Mamba model on FEIDEGGER')
    parser.add_argument('--data_path', type=str, default='mamba_dataset/feidegger_mamba_dataset.h5',
                        help='Path to the HDF5 dataset file')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--output_dir', type=str, default='mamba_output',
                        help='Directory to save model checkpoints')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load datasets
    train_dataset = FeideggerMambaDataset(args.data_path, split='train')
    val_dataset = FeideggerMambaDataset(args.data_path, split='val')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=4)
    
    # Get vocabulary size from the dataset
    with h5py.File(args.data_path, 'r') as f:
        # Estimate vocab size from input_ids
        sample_input_ids = torch.tensor(f['train']['sample_0']['input_ids'][()])
        vocab_size = max(sample_input_ids).item() + 1
        max_seq_len = sample_input_ids.size(0)
    
    # Create model
    model = ImageTextMambaModel(
        text_vocab_size=vocab_size,
        max_seq_len=max_seq_len
    ).to(device)
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Training loop
    best_val_loss = float('inf')
    val_losses = []
    
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        
        # Train
        train_epoch(model, train_loader, optimizer, device)
        
        # Evaluate
        val_loss = evaluate(model, val_loader, device)
        val_losses.append(val_loss)
        print(f"Validation Loss: {val_loss:.4f}")
        
        # Save model if it's the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
        
        # Save checkpoint for the epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss
        }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Plot validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(val_losses)
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(args.output_dir, 'validation_loss.png'))
    
    print("Training complete!")


if __name__ == "__main__":
    main()
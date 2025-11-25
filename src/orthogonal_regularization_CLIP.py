from PIL import Image
import pandas as pd
from torch import nn
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers import util
from torch.utils.data import DataLoader
from sentence_transformers import SentencesDataset
from sentence_transformers import SentenceTransformer, losses, InputExample, datasets

import torch

# Load CSV
csv_path = "visualization_explorer/feidegger_visualization_data_valid.csv"
df = pd.read_csv(csv_path)

class FactorizedDisentanglementLoss(nn.Module):
    def __init__(self, alpha=0.1, n_factors=4):
        super().__init__()
        self.alpha = alpha
        self.n_factors = n_factors

    def forward(self, emb):
        # emb: [batch_size, emb_dim]
        batch_size, emb_dim = emb.shape
        factor_dim = emb_dim // self.n_factors
        loss = 0
        for i in range(self.n_factors):
            factor = emb[:, i*factor_dim:(i+1)*factor_dim]
            # Encourage independence: minimize correlation between factors
            for j in range(i+1, self.n_factors):
                other_factor = emb[:, j*factor_dim:(j+1)*factor_dim]
                corr = torch.abs(torch.mean(factor * other_factor))
                loss += corr
        return self.alpha * loss
    
# pre-processing 
#https://huggingface.co/docs/transformers/main/en/preprocessing

# Build InputExample list
train_examples = []
test_examples = []
val_examples = []
for _, row in df.iloc[:100].iterrows():
    img_path = row["image_path"]
    text = row["text"] 
    if row["mamba_split"] == "train":   
        train_examples.append(InputExample(texts=[img_path, text]))  
    elif row["mamba_split"] == "test":
        test_examples.append(InputExample(texts=[img_path, text]))
    elif row["mamba_split"] == "val":
        val_examples.append(InputExample(texts=[img_path, text]))

# train_dataloader = DataLoader(train_examples, batch_size=8, shuffle=True)
print(f"Number of training examples: {len(train_examples)}")
model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')
train_loss = losses.MultipleNegativesRankingLoss(model)

num_epochs = 10  # Set as needed
# warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

# In your training loop:
disent_loss_fn = FactorizedDisentanglementLoss(alpha=0.1, n_factors=4)
batch_size = 8
train_dataset = SentencesDataset(train_examples, model)
train_dataloader = datasets.NoDuplicatesDataLoader(train_examples, batch_size=8)

# ...existing code...
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    optimizer_class=torch.optim.Adam,
    optimizer_params={'lr': 2e-5},
    show_progress_bar=True
)

# Save model weights after training
model.save("output/orthogonal_clip_model")
print("Model weights saved to output/orthogonal_clip_model")

# For unsupervised disentanglement using the CLIP architecture and pretrained weights, you can:

# Use a factorized latent space:
# Split the embedding into several parts (e.g., color, shape, etc.) and encourage independence between them.

# Add an independence-promoting regularization:
# Use a loss like Total Correlation (TC) or InfoNCE to encourage different parts of the embedding to be statistically independent.

# Keep CLIP’s contrastive loss:
# Continue using the CLIP loss for image-text alignment.
# #     Notes:

# This is a simple independence regularization. For stronger disentanglement, look into InfoNCE, Total Correlation, or β-VAE losses.
# You can reuse CLIP’s pretrained weights and architecture.
# This approach does not require attribute labels.
# Summary:

# Factorize the embedding and add a regularization loss to encourage independence between factors.
# Combine with CLIP’s contrastive loss for alignment.
# No labels required.
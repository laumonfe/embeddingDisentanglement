from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
from PIL import Image

# Prepare your data as InputExample(texts=[...], images=[...])
train_examples = [
    InputExample(texts=["A red dress"], images=[Image.open("img1.jpg")]),
    # ...
]

train_dataloader = DataLoader(train_examples, batch_size=16, shuffle=True)

# Load or define your models
vision_model = ...  # e.g., a ViT or ResNet backbone
text_model = ...    # e.g., a BERT or XLM-R backbone

# Combine into a dual-encoder (SentenceTransformer or custom)
model = SentenceTransformer(modules=[vision_model, text_model])

# Main contrastive loss
contrastive_loss = losses.MultipleNegativesRankingLoss(model)

# Optionally, define a disentanglement loss
def disentanglement_loss(image_emb, text_emb):
    # Example: encourage orthogonality
    return torch.mean(torch.abs(torch.sum(image_emb * text_emb, dim=1)))

# Training loop
for batch in train_dataloader:
    image_emb = model.encode(batch.images, convert_to_tensor=True)
    text_emb = model.encode(batch.texts, convert_to_tensor=True)
    loss = contrastive_loss(image_emb, text_emb)
    loss += disentanglement_loss(image_emb, text_emb) * disentangle_weight
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
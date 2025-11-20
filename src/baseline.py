from sentence_transformers import SentenceTransformer, losses
from sentence_transformers import InputExample
from PIL import Image
from torch.utils.data import DataLoader


train_examples = [
    InputExample(texts=["A red dress with floral pattern"], images=[Image.open("path/to/image1.jpg")]),
    InputExample(texts=["A green velvet dress"], images=[Image.open("path/to/image2.jpg")]),
    # ... more pairs ...
]
train_dataloader = DataLoader(train_examples, batch_size=8, shuffle=True)

model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')
train_loss = losses.MultipleNegativesRankingLoss(model)

num_epochs = 1  # Set as needed
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    output_path="output/finetuned-clip-multilingual"
)
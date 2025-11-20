from PIL import Image
import pandas as pd
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers import util
from torch.utils.data import DataLoader
# how to finetune a CLIP model with text-image pairs
# https://github.com/huggingface/sentence-transformers/blob/1ec4902ddb73a8f33cb5ac2ae0b8f77a930b14d6/examples/training/clip/train_clip.ipynb
# https://github.com/ShawhinT/YouTube-Blog/blob/main/multimodal-ai/4-ft-mm-embeddings/2-finetune_clip_sbert.ipynb

# Load CSV
csv_path = "visualization_explorer/feidegger_visualization_data_valid.csv"
df = pd.read_csv(csv_path)


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

train_dataloader = DataLoader(train_examples, batch_size=8, shuffle=True)
print(f"Number of training examples: {len(train_examples)}")
model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')
train_loss = losses.MultipleNegativesRankingLoss(model)

num_epochs = 10  # Set as needed
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    output_path="output/finetuned-clip-multilingual"
)
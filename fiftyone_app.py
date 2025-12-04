import csv
import fiftyone as fo
import fiftyone.brain as fob
import pandas as pd
from tqdm import tqdm
import numpy as np
import os 
from compute_embeddings import load_embeddings
from german_retrieval import get_split_embeddings

model_kind = "finetuned"  # "pretrained" or "finetuned"
emb_dir = rf"data\embeddings\{model_kind}_clip-ViT-B-32-multilingual-v1"

CSV_PATH = r"data\embeddings\feidegger_visualization_data.csv"
df = pd.read_csv(CSV_PATH)


img_emb_path_all = os.path.join(emb_dir, f"image_embeddings_clip-ViT-B-32_{model_kind}.npy")
text_emb_path_all = os.path.join(emb_dir, f"text_embeddings_clip-ViT-B-32-multilingual-v1_{model_kind}.npy")

image_embeddings = load_embeddings(img_emb_path_all)
text_embeddings = load_embeddings(text_emb_path_all)
# alternatevly, get a subset of a specific split
test_df, test_img_emb, test_txt_emb = get_split_embeddings(df, image_embeddings, text_embeddings, "test")


num_samples = min(len(test_df), len(test_txt_emb), len(test_img_emb))
print(f"Using {num_samples} samples.")
print(f"DataFrame length: {len(df)}"
      f", Text Embeddings length: {len(test_txt_emb)}"
      f", Image Embeddings length: {len(test_img_emb)}")
# Create FiftyOne samples
samples = []
for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Loading embeddings into FiftyOne samples"):
    img_path = row["image_path"]
    text = row["text"]
    sample = fo.Sample(
        filepath=img_path,
        text=text,
        text_embedding=test_txt_emb[idx]['embedding'].tolist(),
        image_embedding=test_img_emb[idx]['embedding'].tolist()
    )
    samples.append(sample) 

# Delete existing dataset if it exists
dataser_name = "feidegger_text_embeddings_finetuned"
if dataser_name in fo.list_datasets():
    fo.delete_dataset(dataser_name)

dataset = fo.Dataset(dataser_name)
dataset.add_samples(samples)

# Compute similarity view 
fob.compute_visualization(
    dataset,
    embeddings=np.array([s.text_embedding for s in dataset]),
    brain_key="text_embedding_viz", 
    method="umap"
)

# Compute similarity view 
fob.compute_visualization(
    dataset,
    embeddings=np.array([s.image_embedding for s in dataset]),
    brain_key="image_embedding_viz", 
    method="umap"
)

# Launch the app
session = fo.launch_app(dataset)
session.wait()
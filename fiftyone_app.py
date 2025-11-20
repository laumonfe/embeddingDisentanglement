import fiftyone as fo
import fiftyone.brain as fob
import pandas as pd
import numpy as np

# Load your CSV and embeddings
csv_path = "visualization_explorer/feidegger_visualization_data_valid.csv"
text_embeddings_path = "data/feidegger_clip-ViT-B-32-multilingual-v1_text_embeddings_baseline.npy"
image_embeddings_path = "data/feidegger_clip-ViT-B-32_image_embeddings_baseline.npy"


df = pd.read_csv(csv_path)
text_embeddings = np.load(text_embeddings_path)
image_embeddings = np.load(image_embeddings_path)
num_samples = min(len(df), len(text_embeddings), len(image_embeddings))
print(f"Using {num_samples} samples.")
print(f"DataFrame length: {len(df)}"
      f", Text Embeddings length: {len(text_embeddings)}"
      f", Image Embeddings length: {len(image_embeddings)}")
# Create FiftyOne samples
samples = []
for idx, row in df.iterrows():
    img_path = row["image_path"]
    text = row["text"]
    sample = fo.Sample(
        filepath=img_path,
        text=text,
        text_embedding=text_embeddings[idx].tolist(),
        image_embedding=image_embeddings[idx].tolist()
    )
    samples.append(sample)

# Delete existing dataset if it exists
if "feidegger_text_embeddings" in fo.list_datasets():
    fo.delete_dataset("feidegger_text_embeddings")

dataset = fo.Dataset("feidegger_text_embeddings")
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
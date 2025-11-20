import fiftyone as fo
import fiftyone.brain as fob
import pandas as pd
import numpy as np

# Load your CSV and embeddings
csv_path = "visualization_explorer/feidegger_visualization_data.csv"
embeddings_path = "data/feidegger_embeddings_CLIP.npy"

df = pd.read_csv(csv_path)
embeddings = np.load(embeddings_path)

# Create FiftyOne samples
samples = []
for idx, row in df.iterrows():
    img_path = row["image_path"]
    text = row["text"]
    sample = fo.Sample(
        filepath=img_path,
        text=text,
        text_embedding=embeddings[idx].tolist(),
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

# Launch the app
session = fo.launch_app(dataset)
session.wait()
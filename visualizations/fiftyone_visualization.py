import os
import re
import fiftyone as fo
import fiftyone.brain as fob
import pandas as pd
import numpy as np
from tqdm import tqdm
from compute_embeddings import load_embeddings
from visualizations.retrieval_visualization import get_split_embeddings

CSV_PATH = r"data/feidegger_metadata.csv"
df = pd.read_csv(CSV_PATH)

EMB_ROOT = "data/embeddings"

def try_load_embeddings(emb_dir, model_kind, data_type):
    if model_kind == "baseline":
        img_emb_path = os.path.join(emb_dir, f"image_embeddings_clip-ViT-B-32_{model_kind}.npy")
        txt_emb_path = os.path.join(emb_dir, f"text_embeddings_clip-ViT-B-32-multilingual-v1_{model_kind}.npy")
    else: 
        img_emb_path = os.path.join(emb_dir, f"image_embeddings_clip-ViT-B-32_{model_kind}_{data_type}.npy")
        txt_emb_path = os.path.join(emb_dir, f"text_embeddings_clip-ViT-B-32-multilingual-v1_{model_kind}_{data_type}.npy")
    
    if not (os.path.exists(img_emb_path) and os.path.exists(txt_emb_path)):
        print(f"Failed to load embeddings from {emb_dir} for {model_kind}-{data_type}.")
        return None, None
    return load_embeddings(img_emb_path), load_embeddings(txt_emb_path)

def create_fiftyone_dataset_for_embeddings(model_kind, data_type, emb_dir):
    image_embeddings, text_embeddings = try_load_embeddings(emb_dir, model_kind, data_type)
    if image_embeddings is None or text_embeddings is None:
        return None

    test_df, test_img_emb, test_txt_emb = get_split_embeddings(df, image_embeddings, text_embeddings, "test")
    num_samples = min(len(test_df), len(test_txt_emb), len(test_img_emb))
    print(f"[{model_kind}-{data_type}] Using {num_samples} samples.")

    samples = []
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"FiftyOne samples [{model_kind}-{data_type}]"):
        img_path = row["image_path"]
        text = row["text"]
        sample = fo.Sample(
            filepath=img_path,
            text=text,
            text_embedding=test_txt_emb[idx]['embedding'].tolist(),
            image_embedding=test_img_emb[idx]['embedding'].tolist(),
            item_idx=row["item_idx"],
            model_kind=model_kind,
            data_type=data_type
        )
        samples.append(sample)

    dataset_name = f"feidegger_{model_kind}_{data_type}_clip"
    if dataset_name in fo.list_datasets():
        dataset = fo.load_dataset(dataset_name)
    else:
        dataset = fo.Dataset(dataset_name)
        dataset.add_samples(samples)

    # Compute UMAP visualizations only if not already present
    if "text_embedding_viz" not in dataset.list_brain_runs():
        fob.compute_visualization(
            dataset,
            embeddings=np.array([s.text_embedding for s in dataset]),
            brain_key="text_embedding_viz",
            method="umap", 
            metric="cosine"

        )
    if "image_embedding_viz" not in dataset.list_brain_runs():
        fob.compute_visualization(
            dataset,
            embeddings=np.array([s.image_embedding for s in dataset]),
            brain_key="image_embedding_viz",
            method="umap", 
            metric="cosine"
        )
    return dataset

def main():
    datasets = []
    for model_kind in ["finetuned", "disentangled"]:
        for data_type in ["default", "grouped"]:
            emb_dir = os.path.join(EMB_ROOT, f"{model_kind}_{data_type}_clip-ViT-B-32-multilingual-v1")
            if os.path.exists(emb_dir):
                dataset = create_fiftyone_dataset_for_embeddings(model_kind, data_type, emb_dir)
                if dataset is not None:
                    datasets.append(dataset)

    # also baseline 
    model_kind = "baseline"
    data_type = "None"
    emb_dir = os.path.join(EMB_ROOT, f"{model_kind}_clip-ViT-B-32-multilingual-v1")
    if os.path.exists(emb_dir):
        dataset = create_fiftyone_dataset_for_embeddings(model_kind, data_type, emb_dir)
        if dataset is not None:
            datasets.append(dataset)

    if datasets:
        session = fo.launch_app(datasets[0], port=6200)  # Launch with the first dataset
        session.wait()
    else:
        print("No datasets were created. Check your embeddings folders and file names.")

if __name__ == "__main__":
    main()
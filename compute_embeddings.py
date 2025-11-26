import os
import pandas as pd
import torch

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer



def load_or_compute_embeddings(model, inputs, emb_save_path):
    need_recompute = True
    if os.path.exists(emb_save_path):
        print(f"Loading embeddings from {emb_save_path}")
        embeddings = np.load(emb_save_path)
        if len(embeddings) == len(inputs):
            need_recompute = False
        else:
            print("Warning: Loaded embeddings do not match input length. Recomputing embeddings.")

    if need_recompute:
        print("Calculating embeddings...")
        embeddings = []
        for inp in tqdm(inputs, total=len(inputs)):
            try:
                with torch.no_grad():
                    emb = model.encode(inp)
                embeddings.append(emb)
            except Exception as e:
                print(f"Error processing {inp}: {e}")
        embeddings = np.array(embeddings)
        os.makedirs(os.path.dirname(emb_save_path), exist_ok=True)
        np.save(emb_save_path, embeddings)
        print("Contains:", len(embeddings) , "embeddings.")
        print(f"Saved embeddings to {emb_save_path}")
    return embeddings


def get_split_data(df, split_name):
    split_df = df[df["mamba_split"] == split_name]
    image_paths = split_df['image_path'].tolist()
    texts = split_df['text'].tolist()
    return image_paths, texts




if __name__ == "__main__":

    CSV_PATH = "visualization_explorer/feidegger_visualization_data.csv"
    df = pd.read_csv(CSV_PATH)

    # Filter for valid images
    valid_indices = [i for i, img_path in enumerate(df['image_path']) if os.path.exists(img_path)]
    filtered_df = df.iloc[valid_indices].reset_index(drop=True)

    img_model = SentenceTransformer('clip-ViT-B-32')
    text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')

    # Calculate "all" split embeddings once
    all_image_paths = filtered_df['image_path'].tolist()
    all_texts = filtered_df['text'].tolist()
    # save directory
    img_emb_path_all = "data/embeddings/clip_image_embeddings_all.npy"
    txt_emb_path_all = "data/embeddings/clip_text_embeddings_all.npy"
    csv_path = "data/embeddings/all_split.csv"

    image_embeddings_all = load_or_compute_embeddings(img_model, all_image_paths, img_emb_path_all)
    text_embeddings_all = load_or_compute_embeddings(text_model, all_texts, txt_emb_path_all)

    if len(text_embeddings_all) == 0 or len(image_embeddings_all) == 0:
        print("Error: Embeddings for 'all' split are empty. Aborting split saving.")
        exit()
    print(f"Saved all split DataFrame to {csv_path}")
    # Save split embeddings by slicing the "all" arrays
    for split in ["train", "val", "test"]:
        split_df = filtered_df[filtered_df["mamba_split"] == split]
        print(f"{split} split length: {len(split_df)}")  
        split_indices = split_df.index.tolist()
        split_img_emb_path = f"data/embeddings/clip_image_embeddings_{split}.npy"
        split_txt_emb_path = f"data/embeddings/clip_text_embeddings_{split}.npy"
        np.save(split_img_emb_path, image_embeddings_all[split_indices])
        np.save(split_txt_emb_path, text_embeddings_all[split_indices])    
        print(f"Saved {split} image embeddings to {split_img_emb_path}")
        print(f"Saved {split} text embeddings to {split_txt_emb_path}")

        # Save split DataFrame as CSV
        split_csv_path = f"data/embeddings/{split}_split.csv"
        split_df.to_csv(split_csv_path, index=False)
        print(f"Saved {split} split DataFrame to {split_csv_path}")
    
    filtered_df.to_csv(csv_path, index=False)   
    print("Done computing and saving embeddings for all splits.")
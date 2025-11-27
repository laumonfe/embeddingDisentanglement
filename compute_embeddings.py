import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
from sentence_transformers import SentenceTransformer



def load_embeddings(emb_save_path):
    if os.path.exists(emb_save_path):
        print(f"Loading embeddings from {emb_save_path}")
        embeddings = np.load(emb_save_path, allow_pickle=True)
        print("Contains:", len(embeddings) , "embeddings.")
        return embeddings
    else:
        print(f"Embeddings file {emb_save_path} not found.")
        return None


def compute_embeddings(text_model, image_model, df, img_emb_save_path, txt_emb_save_path):
    img_embeddings = []
    text_embeddings = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Computing embeddings..."):
        item_idx = row.item_idx
        desc_idx = row.desc_idx
        image_path = row.image_path
        text = row.text
        if not os.path.exists(image_path):
            print(f"Missing image: {image_path}")
            continue
        if not isinstance(text, str) or not text.strip():
            print(f"Invalid text for item {item_idx} {desc_idx}")
            continue
        try:
            img_emb = image_model.encode(Image.open(image_path))
            text_emb = text_model.encode(text)
            img_embeddings.append({'idx': item_idx, 'desc_idx': desc_idx, 'embedding': np.array(img_emb)})
            text_embeddings.append({'idx': item_idx, 'desc_idx': desc_idx, 'embedding': np.array(text_emb)})
        except Exception as e:
            print(f"Error processing {item_idx}: {e}")

    img_embeddings = np.array(img_embeddings, dtype=object)
    text_embeddings = np.array(text_embeddings, dtype=object)
    if dir_path := os.path.dirname(txt_emb_save_path):
        os.makedirs(dir_path, exist_ok=True)
    if dir_path := os.path.dirname(img_emb_save_path):
        os.makedirs(dir_path, exist_ok=True)
    np.save(img_emb_save_path, img_embeddings)
    np.save(txt_emb_save_path, text_embeddings)
    print("Images contain:", len(img_embeddings), "embeddings.")
    print("Text contains:", len(text_embeddings), "embeddings.")
    print(f"Saved embeddings to {img_emb_save_path} and {txt_emb_save_path}")



if __name__ == "__main__":

    CSV_PATH = "data\embeddings\feidegger_visualization_data.csv"
    df = pd.read_csv(CSV_PATH)

    img_model = SentenceTransformer('clip-ViT-B-32')
    text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')

    emb_dir = r"\data\embeddings\baseline_clip-ViT-B-32-multilingual-v1"
    img_emb_path_all = os.path.join(emb_dir, "image_embeddings_clip-ViT-B-32_baseline.npy")
    txt_emb_path_all = os.path.join(emb_dir,"text_embeddings_clip-ViT-B-32-multilingual-v1_baseline.npy")

    image_embeddings_all, text_embeddings_all = compute_embeddings(
        text_model, img_model, df, img_emb_path_all, txt_emb_path_all
    )


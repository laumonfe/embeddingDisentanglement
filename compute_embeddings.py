import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
from sentence_transformers import SentenceTransformer
from src.clip_utils import clip_encode
import torch


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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_model = image_model.to(device)
    text_model = text_model.to(device)
    
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
            img_emb = clip_encode(image_model, Image.open(image_path), modality="image")
            text_emb = clip_encode(text_model, text, modality="text")
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
    from src.utils import load_vision_with_projection, load_distilbert_with_projection_finetuned
    model_kind = "finetuned"  # Options: "baseline", "finetuned", "disentangled"

    CSV_PATH = r"data\embeddings\feidegger_visualization_data.csv"
    df = pd.read_csv(CSV_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_kind == "baseline":
        print("Using baseline models...")
        img_model = SentenceTransformer('clip-ViT-B-32')
        text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')

        emb_dir = r"data\embeddings\baseline_clip-ViT-B-32-multilingual-v1"
        img_emb_path_all = os.path.join(emb_dir, "image_embeddings_clip-ViT-B-32_baseline.npy")
        txt_emb_path_all = os.path.join(emb_dir,"text_embeddings_clip-ViT-B-32-multilingual-v1_baseline.npy")


    if model_kind == "finetuned":
        print("Using finetuned models...")
        model_directory = r"output/finetuned_baseline"
        emb_dir = r"data\embeddings\finetuned_clip-ViT-B-32-multilingual-v1"
        img_model = load_vision_with_projection(os.path.join(model_directory, "vision_encoder"), device)
        text_model = load_distilbert_with_projection_finetuned(os.path.join(model_directory, "text_encoder"), device)
        img_emb_path_all = os.path.join(emb_dir, "image_embeddings_clip-ViT-B-32_finetuned.npy")
        txt_emb_path_all = os.path.join(emb_dir,"text_embeddings_clip-ViT-B-32-multilingual-v1_finetuned.npy")

    if model_kind == "disentangled":
        print("Using disentangled models...")
        raise NotImplementedError("Disentangled model embedding computation not implemented yet.")


    # img_model = SentenceTransformer('clip-ViT-B-32')
    # text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')

    # emb_dir = r"data\embeddings\baseline_clip-ViT-B-32-multilingual-v1"
    # img_emb_path_all = os.path.join(emb_dir, "image_embeddings_clip-ViT-B-32_baseline.npy")
    # txt_emb_path_all = os.path.join(emb_dir,"text_embeddings_clip-ViT-B-32-multilingual-v1_baseline.npy")
    image_model = img_model.to(device)
    text_model = text_model.to(device)
    compute_embeddings(text_model, img_model, df, img_emb_path_all, txt_emb_path_all)

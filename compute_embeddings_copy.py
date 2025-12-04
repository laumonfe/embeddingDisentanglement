import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch

def load_model(model_kind, model_type, model_directory, device):
    if model_kind == "baseline":
        from sentence_transformers import SentenceTransformer
        if model_type == "image":
            return SentenceTransformer('clip-ViT-B-32')
        else:
            return SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')
    elif model_kind == "finetuned":
        if model_type == "image":
            from src.utils import load_vision_with_projection
            return load_vision_with_projection(os.path.join(model_directory, "vision_encoder"), device)
        else:
            from src.utils import load_distilbert_with_projection_finetuned
            return load_distilbert_with_projection_finetuned(os.path.join(model_directory, "text_encoder"), device)
    else:
        raise NotImplementedError(f"{model_kind} model loading not implemented.")

def compute_embeddings(image_encoder, text_encoder, df, img_emb_save_path, txt_emb_save_path):
    img_embeddings, text_embeddings = [], []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Computing embeddings..."):
        item_idx, desc_idx, image_path, text = row.item_idx, row.desc_idx, row.image_path, row.text
        if not os.path.exists(image_path):
            print(f"Missing image: {image_path}")
            continue
        if not isinstance(text, str) or not text.strip():
            print(f"Invalid text for item {item_idx} {desc_idx}")
            continue
        try:
            img_emb = image_encoder.encode(Image.open(image_path)) if hasattr(image_encoder, "encode") else image_encoder.encode(Image.open(image_path), convert_to_tensor=False)
            text_emb = text_encoder.encode(text) if hasattr(text_encoder, "encode") else text_encoder.encode(text, convert_to_tensor=False)
            img_embeddings.append({'idx': item_idx, 'desc_idx': desc_idx, 'embedding': np.array(img_emb)})
            text_embeddings.append({'idx': item_idx, 'desc_idx': desc_idx, 'embedding': np.array(text_emb)})
        except Exception as e:
            print(f"Error processing {item_idx}: {e}")

    for path, embeddings in [(img_emb_save_path, img_embeddings), (txt_emb_save_path, text_embeddings)]:
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        np.save(path, np.array(embeddings, dtype=object))
    print(f"Saved embeddings to {img_emb_save_path} and {txt_emb_save_path}")

if __name__ == "__main__":
    model_kind = "finetuned"  # Options: "baseline", "finetuned", "disentangled"
    CSV_PATH = r"data\embeddings\feidegger_visualization_data.csv"
    df = pd.read_csv(CSV_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_directory = r"output/finetuned_baseline" if model_kind == "finetuned" else None
    emb_dir = {
        "baseline": r"data\embeddings\baseline_clip-ViT-B-32-multilingual-v1",
        "finetuned": r"data\embeddings\finetuned_clip-ViT-B-32-multilingual-v1"
    }[model_kind]
    img_emb_path_all = os.path.join(emb_dir, f"image_embeddings_clip-ViT-B-32_{model_kind}.npy")
    txt_emb_path_all = os.path.join(emb_dir, f"text_embeddings_clip-ViT-B-32-multilingual-v1_{model_kind}.npy")

    image_encoder = load_model(model_kind, "image", model_directory, device)
    text_encoder = load_model(model_kind, "text", model_directory, device)
    compute_embeddings(image_encoder, text_encoder, df, img_emb_path_all, txt_emb_path_all)

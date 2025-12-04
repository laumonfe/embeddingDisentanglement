import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from src.models import PretrainedCLIPVision, PretrainedDistilBert, ProjectedCLIPVision, ProjectedDistilBert


def load_embeddings(emb_save_path):
    if os.path.exists(emb_save_path):
        print(f"Loading embeddings from {emb_save_path}")
        embeddings = np.load(emb_save_path, allow_pickle=True)
        print("Contains:", len(embeddings) , "embeddings.")
        return embeddings
    else:
        print(f"Embeddings file {emb_save_path} not found.")
        return None

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
    
    model_kind = "finetuned"  # "pretrained" or "finetuned"
    emb_dir = rf"data\embeddings\{model_kind}_clip-ViT-B-32-multilingual-v1"
    
    CSV_PATH = r"data\embeddings\feidegger_visualization_data.csv"
    df = pd.read_csv(CSV_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_emb_path_all = os.path.join(emb_dir, f"image_embeddings_clip-ViT-B-32_{model_kind}.npy")
    txt_emb_path_all = os.path.join(emb_dir, f"text_embeddings_clip-ViT-B-32-multilingual-v1_{model_kind}.npy")

    if model_kind == "pretrained":
        # Paths to pretrained models
        pretrained_img_model_path = r"pretrained_models/sentence-transformers--clip-ViT-B-32"
        pretrained_text_model_path = r"pretrained_models/sentence-transformers--clip-ViT-B-32-multilingual-v1"
        image_encoder = PretrainedCLIPVision(pretrained_img_model_path, device)
        text_encoder = PretrainedDistilBert(pretrained_text_model_path, device)

    if model_kind == "finetuned":
        # Paths to finetuned models
        finetuned_text_model_path = r"output/finetuned_baseline/best_model/text_encoder"
        finetuned_img_model_path = r"output/finetuned_baseline/best_model/vision_encoder"
        image_encoder = ProjectedCLIPVision(finetuned_img_model_path, device)
        text_encoder = ProjectedDistilBert(finetuned_text_model_path, device)

    compute_embeddings(image_encoder, text_encoder, df, img_emb_path_all, txt_emb_path_all)

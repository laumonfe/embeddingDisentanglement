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



def get_default_paths(model_kind, dataset_type, device):
    if model_kind == "baseline":
        img_model_path = "pretrained_models/sentence-transformers--clip-ViT-B-32"
        txt_model_path = "pretrained_models/sentence-transformers--clip-ViT-B-32-multilingual-v1"
        img_encoder = PretrainedCLIPVision(img_model_path, device)
        txt_encoder = PretrainedDistilBert(txt_model_path, device)
        emb_dir = f"data/embeddings/baseline_clip-ViT-B-32-multilingual-v1"
    else:
        base = f"output/{model_kind}_{dataset_type}_clip"
        img_model_path = os.path.join(base, "best_model", "vision_encoder")
        txt_model_path = os.path.join(base, "best_model", "text_encoder")
        img_encoder = ProjectedCLIPVision(img_model_path, device)
        txt_encoder = ProjectedDistilBert(txt_model_path, device)
        emb_dir = f"data/embeddings/{model_kind}_{dataset_type}_clip-ViT-B-32-multilingual-v1"
    return img_encoder, txt_encoder, emb_dir
if __name__ == "__main__":
    
    import argparse

    parser = argparse.ArgumentParser(description="Compute embeddings for FEIDEGGER dataset.")
    parser.add_argument("--model_kind",choices=["baseline", "finetuned", "disentangled", "all"],default="all",help="Which model to use: baseline, finetuned or disentangled ")
    parser.add_argument("--dataset_type", type=str, choices=["default", "grouped"], default="default", help="Type of dataset grouping: default or grouped")
    parser.add_argument("--csv_path",type=str,default="data/feidegger_metadata.csv",help="Path to the CSV file containing image paths and text descriptions.")
    args = parser.parse_args()

    
    CSV_PATH = args.csv_path
    df = pd.read_csv(CSV_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_kind == "all":
        configs = [("disentangled", "default"),
                    ("disentangled", "grouped"),
                    ("finetuned", "default"),
                    ("finetuned", "grouped"),
                    ("baseline", None)]
            
        for model_kind, dataset_type in configs:
            try:
                img_encoder, txt_encoder, emb_dir = get_default_paths(model_kind, dataset_type, device)
                img_emb_path = os.path.join(emb_dir, f"image_embeddings_clip-ViT-B-32_{model_kind}_{dataset_type}.npy")
                txt_emb_path = os.path.join(emb_dir, f"text_embeddings_clip-ViT-B-32-multilingual-v1_{model_kind}_{dataset_type}.npy")
                compute_embeddings(img_encoder, txt_encoder, df, img_emb_path, txt_emb_path)
            except Exception as e:
                print(f"Failed for {model_kind}, {dataset_type}: {e}")
                continue
                        
    else: 

        print(f"Processing: {args.model_kind}, {args.dataset_type}")
        img_encoder, txt_encoder, emb_dir = get_default_paths(args.model_kind, args.dataset_type, device)
        img_emb_path = os.path.join(emb_dir, f"image_embeddings_clip-ViT-B-32_{args.model_kind}_{args.dataset_type}.npy")
        txt_emb_path = os.path.join(emb_dir, f"text_embeddings_clip-ViT-B-32-multilingual-v1_{args.model_kind}_{args.dataset_type}.npy")
        compute_embeddings(img_encoder, txt_encoder, df, img_emb_path, txt_emb_path)


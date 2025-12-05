import numpy as np
import torch
from tqdm import tqdm
import pandas as pd

def compute_recall_precision_table(text_embeddings, image_embeddings, k_list=[1, 5]):
    """
    Computes Recall@K and Precision@K for each sample and stores results in a DataFrame.
    """
    img_emb_matrix = np.stack([e['embedding'] for e in image_embeddings])
    img_emb_tensor = torch.tensor(img_emb_matrix)
    results = []

    for i, txt_emb_dict in tqdm(enumerate(text_embeddings), total=len(text_embeddings), desc="Evaluating Recall/Precision@K"):
        txt_emb = torch.tensor(txt_emb_dict['embedding']).unsqueeze(0)
        sims = torch.nn.functional.cosine_similarity(txt_emb, img_emb_tensor, dim=1)
        gt_idx = txt_emb_dict['idx']
        gt_img_indices = [j for j, e in enumerate(image_embeddings) if e['idx'] == gt_idx]
        row = {'text_idx': gt_idx}
        for k in k_list:
            topk_indices = torch.topk(sims, k).indices.tolist()
            recall = int(any(j in topk_indices for j in gt_img_indices))
            precision = sum(j in gt_img_indices for j in topk_indices) / k
            row[f"recall@{k}"] = recall
            row[f"precision@{k}"] = precision
        results.append(row)
    df_results = pd.DataFrame(results)
    # Compute and print mean metrics
    for k in k_list:
        print(f"Recall@{k}: {df_results[f'recall@{k}'].mean():.4f}")
        print(f"Precision@{k}: {df_results[f'precision@{k}'].mean():.4f}")
    return df_results

if __name__ == "__main__":
    import os
    from compute_embeddings import load_embeddings

    model_kind = "finetuned"  # or "pretrained"
    emb_dir = rf"data\embeddings\{model_kind}_clip-ViT-B-32-multilingual-v1"
    CSV_PATH = r"data\embeddings\feidegger_visualization_data.csv"
    df = pd.read_csv(CSV_PATH)

    img_emb_path_all = os.path.join(emb_dir, f"image_embeddings_clip-ViT-B-32_{model_kind}.npy")
    text_emb_path_all = os.path.join(emb_dir, f"text_embeddings_clip-ViT-B-32-multilingual-v1_{model_kind}.npy")

    image_embeddings = load_embeddings(img_emb_path_all)
    text_embeddings = load_embeddings(text_emb_path_all)

    results_df = compute_recall_precision_table(text_embeddings, image_embeddings, k_list=[1, 5])
    results_df.to_csv(f"retrieval_results_{model_kind}.csv", index=False)
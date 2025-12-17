import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from sentence_transformers import  util

def compute_recall_precision_table(text_embeddings, image_embeddings, k_list=[1, 5]):
    """
    Computes Recall@K and Precision@K for each sample and stores results in a DataFrame.
    Also records the rank/position of the ground truth image for each text query.
    """
    img_emb_matrix = np.stack([e['embedding'] for e in image_embeddings])
    img_emb_tensor = torch.tensor(img_emb_matrix)
    results = []

    for i, txt_emb_dict in tqdm(enumerate(text_embeddings), total=len(text_embeddings), desc="Evaluating Recall/Precision@K"):
        txt_emb = torch.tensor(txt_emb_dict['embedding']).unsqueeze(0)
        sims = util.cos_sim(txt_emb, img_emb_tensor)[0]
        gt_idx = txt_emb_dict['idx']
        gt_img_indices = [j for j, e in enumerate(image_embeddings) if e['idx'] == gt_idx]
        row = {'text_idx': gt_idx}

        # Sort all indices by similarity (descending)
        sorted_indices = torch.argsort(sims, descending=True).tolist()
        # Find the minimum rank among all ground truth images
        gt_ranks = [sorted_indices.index(j) for j in gt_img_indices if j in sorted_indices]
        rank = min(gt_ranks) if gt_ranks else -1  # -1 if not found
        row['gt_rank'] = rank

        for k in k_list:
            topk_indices = sorted_indices[:k]
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
    print(f"Mean GT Rank: {df_results['gt_rank'][df_results['gt_rank']!=-1].mean():.2f}")
    return df_results

def get_emb_dir(model_kind, dataset_type=None):
    if model_kind == "baseline":
        return f"data/embeddings/baseline_clip-ViT-B-32-multilingual-v1"
    else:
        return f"data/embeddings/{model_kind}_{dataset_type}_clip-ViT-B-32-multilingual-v1"

if __name__ == "__main__":
    import os
    import argparse
    from compute_embeddings import load_embeddings
    from visualizations.retrieval_visualization import get_split_embeddings

    # parser = argparse.ArgumentParser(description="Compute embeddings for FEIDEGGER dataset.")
    # parser.add_argument(
    #     "--model_kind",
    #     choices=["pretrained", "finetuned", "disentangled"],
    #     default="disentangled",
    #     help="Which model to use: pretrained (baseline), finetuned (on FEIDEGGER), or disentangled (on FEIDEGGER)."
    # )
    # parser.add_argument(
    #     "--csv_path",
    #     type=str,
    #     default="data/embeddings/feidegger_visualization_data.csv",
    #     help="Path to the CSV file containing image paths and text descriptions."
    # )

    # args = parser.parse_args()
    # model_kind = args.model_kind
    # CSV_PATH = args.csv_path

    # emb_dir = rf"data\embeddings\{model_kind}_clip-ViT-B-32-multilingual-v1"
    # df = pd.read_csv(CSV_PATH)

    # img_emb_path_all = os.path.join(emb_dir, f"image_embeddings_clip-ViT-B-32_{model_kind}.npy")
    # text_emb_path_all = os.path.join(emb_dir, f"text_embeddings_clip-ViT-B-32-multilingual-v1_{model_kind}.npy")

    # image_embeddings = load_embeddings(img_emb_path_all)
    # text_embeddings = load_embeddings(text_emb_path_all)

    # test_df, test_img_emb, test_txt_emb = get_split_embeddings(df, image_embeddings, text_embeddings, "test")


    # results_df = compute_recall_precision_table(test_txt_emb, test_img_emb, k_list=[1, 5, 10, 50])
    # results_df.to_csv(f"retrieval_results_{model_kind}.csv", index=False)

    parser = argparse.ArgumentParser(description="Aggregate retrieval metrics for all models.")
    parser.add_argument(
        "--csv_path",
        type=str,
        default="data/feidegger_metadata.csv",
        help="Path to the CSV file containing image paths and text descriptions."
    )
    args = parser.parse_args()
    CSV_PATH = args.csv_path
    df = pd.read_csv(CSV_PATH)

    configs = [
        ("disentangled", "default"),
        ("disentangled", "grouped"),
        ("finetuned", "default"),
        ("finetuned", "grouped"),
        ("baseline", None)
    ]

    summary_rows = []
    for model_kind, dataset_type in configs:
        print(f"\nEvaluating: {model_kind}, {dataset_type}")
        emb_dir = get_emb_dir(model_kind, dataset_type)
        img_emb_path_all = os.path.join(emb_dir, f"image_embeddings_clip-ViT-B-32_{model_kind}_{dataset_type}.npy" if dataset_type else f"image_embeddings_clip-ViT-B-32_{model_kind}.npy")
        text_emb_path_all = os.path.join(emb_dir, f"text_embeddings_clip-ViT-B-32-multilingual-v1_{model_kind}_{dataset_type}.npy" if dataset_type else f"text_embeddings_clip-ViT-B-32-multilingual-v1_{model_kind}.npy")

        try:
            image_embeddings = load_embeddings(img_emb_path_all)
            text_embeddings = load_embeddings(text_emb_path_all)
            test_df, test_img_emb, test_txt_emb = get_split_embeddings(df, image_embeddings, text_embeddings, "test")
            results_df = compute_recall_precision_table(test_txt_emb, test_img_emb, k_list=[1, 5, 10, 50])
            results_df.to_csv(f"retrieval_results_{model_kind}_{dataset_type}.csv", index=False)

            # Aggregate mean metrics
            row = {
                "model_kind": model_kind,
                "dataset_type": dataset_type if dataset_type else "default",
                "recall@1": results_df["recall@1"].mean(),
                "recall@5": results_df["recall@5"].mean(),
                "recall@10": results_df["recall@10"].mean(),
                "recall@50": results_df["recall@50"].mean(),
                "precision@1": results_df["precision@1"].mean(),
                "precision@5": results_df["precision@5"].mean(),
                "precision@10": results_df["precision@10"].mean(),
                "precision@50": results_df["precision@50"].mean(),
                "mean_gt_rank": results_df["gt_rank"][results_df["gt_rank"] != -1].mean()
            }
            summary_rows.append(row)
        except Exception as e:
            print(f"Failed for {model_kind}, {dataset_type}: {e}")
            continue

    # Create and save summary table
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("retrieval_summary.csv", index=False)
    print("\nSummary table saved as retrieval_summary.csv")
    print(summary_df)
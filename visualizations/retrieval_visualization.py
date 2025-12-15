import os

import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sentence_transformers import  util
from compute_embeddings import load_embeddings
import json
from src.models import PretrainedCLIPVision, PretrainedDistilBert, ProjectedCLIPVision, ProjectedDistilBert


def get_top_k_unique_images(sims, image_embeddings, df, top_k):
    sorted_indices = torch.argsort(sims, descending=True).tolist()
    seen = set()
    results = []
    for i in sorted_indices:
        idx = image_embeddings[i]['idx']
        img_path = df.loc[df['item_idx'] == idx, 'image_path'].values[0]
        if img_path not in seen:
            seen.add(img_path)
            results.append((img_path, sims[i].item()))
        if len(results) == top_k:
            break
    return results


def retrieve_images_by_text(query, text_model, image_embeddings, df, top_k=5):
    text_emb = text_model.encode(query)
    # Stack all embeddings into a matrix for similarity computation
    emb_matrix = np.stack([e['embedding'] for e in image_embeddings])
    sims = util.cos_sim(torch.tensor(text_emb), torch.tensor(emb_matrix))[0]    
    results = get_top_k_unique_images(sims, image_embeddings, df, top_k)
    return results


def retrieve_images_by_image(query_image_path, image_model, image_embeddings, df, top_k=5):
    query_emb = image_model.encode(Image.open(query_image_path))
    emb_matrix = np.stack([e['embedding'] for e in image_embeddings])
    sims = util.cos_sim(torch.tensor(query_emb), torch.tensor(emb_matrix))[0]
    results = get_top_k_unique_images(sims, image_embeddings, df, top_k)
    return results


def plot_images(results, title, query=None, query_type="text"):
    n_results = len(results)
    if query_type == "image":
        fig = plt.figure(figsize=(3 * (n_results + 1), 6))
        # Plot the query image first
        ax = plt.subplot(1, n_results + 1, 1)
        img = Image.open(query)
        plt.imshow(img)
        # Add red border
        rect = patches.Rectangle(
            (0, 0), img.size[0], img.size[1],
            linewidth=4, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        plt.title("Query Image", fontsize=10, pad=10, color='red', weight="bold")
        plt.axis('off')
        # Plot the results
        for i, (img_path, score) in enumerate(results):
            plt.subplot(1, n_results + 1, i + 2)
            img = Image.open(img_path)
            plt.imshow(img)
            plt.title(f"Score: {score:.2f}", fontsize=10, pad=10)
            plt.axis('off')
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        return fig
    else:
        fig = plt.figure(figsize=(3 * n_results, 6))
        for i, (img_path, score) in enumerate(results):
            plt.subplot(1, n_results, i + 1)
            img = Image.open(img_path)
            plt.imshow(img)
            plt.title(f"Score: {score:.2f}", fontsize=10, pad=10)
            plt.axis('off')
        plt.suptitle(title, fontsize=14)
        # Subtitle (query)
        if query is not None:
            fig.text(0.5, 0.91, f"Query: {query}", ha='center', fontsize=10, color='red',weight="bold")
        plt.tight_layout()
        return fig

def get_split_embeddings(df, image_embeddings, text_embeddings, split_name):
    """
    Returns filtered DataFrame and corresponding image/text embeddings for a given split.
    Matches both 'idx' and 'desc_idx'.
    """
    split_df = df[df["split"] == split_name]
    split_keys = set(zip(split_df["item_idx"], split_df["desc_idx"]))
    split_image_embeddings = [e for e in image_embeddings if (e['idx'], e['desc_idx']) in split_keys]
    split_text_embeddings = [e for e in text_embeddings if (e['idx'], e['desc_idx']) in split_keys]
    return split_df.reset_index(drop=True), np.array(split_image_embeddings, dtype=object), np.array(split_text_embeddings, dtype=object)

def save_retrieval_plots(fig, out_dir, model_kind, dataset_type, mode):
    fname = f"{mode}_{model_kind}_{dataset_type}.png"
    fig.savefig(os.path.join(out_dir, fname), bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    #query = "ein wunderschönes und sehr festliches langes Kleid" # "a beautiful and very festive long dress"
    #query = "ein kurzes schwarzes Kleid"  # a short black dress
    #query = "ein glitzerndes und schickes Kleid"  # a glitter and fancy dress
    #query = "ein grünes Samtkleid mit V-Ausschnitt und langen Ärmeln" #"a velvet green dress with a V-neck and long sleeves" 
    #"spring dress perfect for a picnic date"  
    # #"a red dress with floral pattern"

    # query = "A dress that whispers rebellion."


    # model_kind = "finetuned"  # "pretrained" or "finetuned"
    # emb_dir = rf"data\embeddings\{model_kind}_clip-ViT-B-32-multilingual-v1"
    
    CSV_PATH = r"data/feidegger_metadata.csv"
    df = pd.read_csv(CSV_PATH)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # img_emb_path_all = os.path.join(emb_dir, f"image_embeddings_clip-ViT-B-32_{model_kind}.npy")
    # text_emb_path_all = os.path.join(emb_dir, f"text_embeddings_clip-ViT-B-32-multilingual-v1_{model_kind}.npy")

    # if model_kind == "baseline":
    #     # Paths to pretrained models
    #     pretrained_img_model_path = r"pretrained_models/sentence-transformers--clip-ViT-B-32"
    #     pretrained_text_model_path = r"pretrained_models/sentence-transformers--clip-ViT-B-32-multilingual-v1"
    #     image_encoder = PretrainedCLIPVision(pretrained_img_model_path, device)
    #     text_encoder = PretrainedDistilBert(pretrained_text_model_path, device)

    # if model_kind == "finetuned":
    #     # Paths to finetuned models
    #     finetuned_text_model_path = r"output/finetuned_baseline/best_model/text_encoder"
    #     finetuned_img_model_path = r"output/finetuned_baseline/best_model/vision_encoder"
    #     image_encoder = ProjectedCLIPVision(finetuned_img_model_path, device)
    #     text_encoder = ProjectedDistilBert(finetuned_text_model_path, device)


    
    # image_embeddings = load_embeddings(img_emb_path_all)
    # text_embeddings = load_embeddings(text_emb_path_all)
    # # alternatevly, get a subset of a specific split
    # test_df, test_img_emb, test_txt_emb = get_split_embeddings(df, image_embeddings, text_embeddings, "test")

    # #query = "red dress"
    # id = 42
    # query = test_df.iloc[id]['text']
    # gt_img = test_df.iloc[id]['image_path']
    # plt.imshow(Image.open(gt_img))

    # ########### Same Query Only in the test split ###########
    # print("Text-to-Image Retrieval Example Test:")
    # results = retrieve_images_by_text(query, text_encoder, test_img_emb, test_df,  top_k=5)
    # plot_images(results, "Text-to-Image Retrieval (M-CLIP)", query=query, query_type="text")

    # print("\nImage-to-Image Retrieval Example Test:")
    # example_image = results[0][0]
    # print(f"Using example image: {example_image}")
    # results = retrieve_images_by_image(example_image, image_encoder, test_img_emb, test_df, top_k=5)
    # plot_images(results, "Image-to-Image Retrieval (M-CLIP)", query=example_image, query_type="image")

    # List of 10 test IDs (you can customize these)
    import shutil

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ids = [5, 12, 23, 42, 55, 67, 78, 88, 101, 123]

    # All model configs to test
    configs = [
        ("disentangled", "default"),
        ("disentangled", "grouped"),
        ("finetuned", "default"),
        ("finetuned", "grouped"),
        ("baseline", None)
    ]

    results = []

    for model_kind, dataset_type in configs:
        print(f"\nEvaluating model: {model_kind}, {dataset_type}")
        # Set up embedding directory and model loading as in your script
        try: 
            if model_kind == "baseline":

                emb_dir = f"data/embeddings/baseline_clip-ViT-B-32-multilingual-v1"
                img_emb_path_all = os.path.join(emb_dir, f"image_embeddings_clip-ViT-B-32_{model_kind}.npy")
                text_emb_path_all = os.path.join(emb_dir, f"text_embeddings_clip-ViT-B-32-multilingual-v1_{model_kind}.npy")
                image_encoder = PretrainedCLIPVision("pretrained_models/sentence-transformers--clip-ViT-B-32", device)
                text_encoder = PretrainedDistilBert("pretrained_models/sentence-transformers--clip-ViT-B-32-multilingual-v1", device)
            else:
                emb_dir = f"data/embeddings/{model_kind}_{dataset_type}_clip-ViT-B-32-multilingual-v1"
                img_emb_path_all = os.path.join(emb_dir, f"image_embeddings_clip-ViT-B-32_{model_kind}_{dataset_type}.npy")
                text_emb_path_all = os.path.join(emb_dir, f"text_embeddings_clip-ViT-B-32-multilingual-v1_{model_kind}_{dataset_type}.npy")
                img_model_path = f"/mnt/netstorage/projects/clip/{model_kind}_{dataset_type}_clip/epoch_20/vision_encoder"
                txt_model_path = f"/mnt/netstorage/projects/clip/{model_kind}_{dataset_type}_clip/epoch_20/text_encoder"
                image_encoder = ProjectedCLIPVision(img_model_path, device)
                text_encoder = ProjectedDistilBert(txt_model_path, device)
        except Exception as e:
            print(f"Failed to load models for {model_kind}, {dataset_type}: {e}")
            continue

        image_embeddings = load_embeddings(img_emb_path_all)
        text_embeddings = load_embeddings(text_emb_path_all)
        if image_embeddings is None or text_embeddings is None:
            print(f"Skipping {model_kind}, {dataset_type} due to missing embeddings.")
            continue
        try:
            test_df, test_img_emb, test_txt_emb = get_split_embeddings(df, image_embeddings, text_embeddings, "test")
        except Exception as e:
            print(f"Failed to get split embeddings for {model_kind}, {dataset_type}: {e}")
            continue
        
        for idx in test_ids:
            if idx >= len(test_df):
                continue  # skip if index out of range

            query_text = test_df.iloc[idx]['text']
            query_img_path = test_df.iloc[idx]['image_path']

            out_dir = os.path.join("data/retrieval_results", str(idx))
            os.makedirs(out_dir, exist_ok=True)
            shutil.copy(query_img_path, os.path.join(out_dir, "query_img.png"))


            # Text-to-Image Retrieval
            text2image_results = retrieve_images_by_text(query_text, text_encoder, test_img_emb, test_df, top_k=5)
            text2image = plot_images(text2image_results, "Text-to-Image Retrieval (M-CLIP)", query=query_text, query_type="text")
            save_retrieval_plots(text2image, out_dir, model_kind, dataset_type, "text2img")


            # Image-to-Image Retrieval
            best_match_image = text2image_results[0][0]
            image2image_results = retrieve_images_by_image(best_match_image, image_encoder, test_img_emb, test_df, top_k=5)
            image2image= plot_images(image2image_results, "Image-to-Image Retrieval (M-CLIP)", query=best_match_image, query_type="image")
            save_retrieval_plots(image2image, out_dir, model_kind, dataset_type, "img2img")


            # i2i_dir = f"retrieval_results/{model_kind}_{dataset_type}/query_{idx}/image_to_image"
            # save_retrieved_images(i2i_results, i2i_dir, "i2i")

            # # Image-to-Image Retrieval
            # i2i_results = retrieve_images_by_image(query_img_path, image_encoder, test_img_emb, test_df, top_k=5)
            # i2i_dir = f"retrieval_results/{model_kind}_{dataset_type}/query_{idx}/image_to_image"
            # save_retrieved_images(i2i_results, i2i_dir, "i2i")

            # # Optionally, also save the query image for reference
            # try:
            #     shutil.copy(query_img_path, f"retrieval_results/{model_kind}_{dataset_type}/query_{idx}/query_image.jpg")
            # except Exception as e:
            #     print(f"Could not copy query image {query_img_path}: {e}")
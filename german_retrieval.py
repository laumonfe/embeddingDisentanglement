import os
import torch
import collections
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sentence_transformers import SentenceTransformer, util
from compute_embeddings import load_embeddings
from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel, CLIPTextModel, DistilBertModel, DistilBertTokenizer
from src.clip_utils import clip_encode
import json
from transformers import DistilBertModel, DistilBertConfig


def retrieve_images_by_text(query, text_model, image_embeddings, df, top_k=5):
    with torch.no_grad():
        #text_emb = text_model.encode([query])
        text_emb = clip_encode(text_model, query, modality="text")
    # Stack all embeddings into a matrix for similarity computation
    emb_matrix = np.stack([e['embedding'] for e in image_embeddings])
    sims = util.cos_sim(torch.tensor(text_emb), torch.tensor(emb_matrix))[0]
    # Get indices sorted by similarity (descending)
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


def retrieve_images_by_image(query_image_path, image_model, image_embeddings, df, top_k=5):
    with torch.no_grad():
        #query_emb = image_model.encode(Image.open(query_image_path))
        query_emb = clip_encode(image_model, Image.open(query_image_path), modality="image")
    emb_matrix = np.stack([e['embedding'] for e in image_embeddings])
    sims = util.cos_sim(torch.tensor(query_emb), torch.tensor(emb_matrix))[0]
    # Exclude identical images
    identical_mask = np.all(emb_matrix == query_emb, axis=1)
    sims[identical_mask] = -float("inf")
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


def plot_images(results, title, query=None, query_type="text"):
    n_results = len(results)
    if query_type == "image":
        plt.figure(figsize=(3 * (n_results + 1), 6))
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
            plt.title(f"{os.path.basename(img_path)}\nscore: {score:.2f}", fontsize=10, pad=10)
            plt.axis('off')
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.show()
    else:
        fig = plt.figure(figsize=(3 * n_results, 6))
        for i, (img_path, score) in enumerate(results):
            plt.subplot(1, n_results, i + 1)
            img = Image.open(img_path)
            plt.imshow(img)
            plt.title(f"{os.path.basename(img_path)}\nscore: {score:.2f}", fontsize=10, pad=10)
            plt.axis('off')
        plt.suptitle(title, fontsize=14)
        # Subtitle (query)
        if query is not None:
            fig.text(0.5, 0.91, f"Query: {query}", ha='center', fontsize=10, color='red',weight="bold")
        plt.tight_layout()
        plt.show()

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


if __name__ == "__main__":
    #query = "ein wunderschönes und sehr festliches langes Kleid" # "a beautiful and very festive long dress"
    #query = "ein kurzes schwarzes Kleid"  # a short black dress
    #query = "ein glitzerndes und schickes Kleid"  # a glitter and fancy dress
    #query = "ein grünes Samtkleid mit V-Ausschnitt und langen Ärmeln" #"a velvet green dress with a V-neck and long sleeves" 
   
    #"spring dress perfect for a picnic date"  
    # #"a red dress with floral pattern"

    # query = "A dress that whispers rebellion."



    CSV_PATH = r"data\embeddings\feidegger_visualization_data.csv"
    # IMG_EMB_PATH = r"data\embeddings\baseline_clip-ViT-B-32-multilingual-v1\image_embeddings_clip-ViT-B-32_baseline.npy"
    # TXT_EMB_PATH = r"data\embeddings\baseline_clip-ViT-B-32-multilingual-v1\text_embeddings_clip-ViT-B-32-multilingual-v1_baseline.npy"
    IMG_EMB_PATH = r"data\embeddings\finetuned_clip-ViT-B-32-multilingual-v1\image_embeddings_clip-ViT-B-32_finetuned.npy"
    TXT_EMB_PATH = r"data\embeddings\finetuned_clip-ViT-B-32-multilingual-v1\text_embeddings_clip-ViT-B-32-multilingual-v1_finetuned.npy"
    # img_model = SentenceTransformer('clip-ViT-B-32')
    # text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')


    img_model_path = r"output\finetuned_vision"
    #img_model_path = r"pretrained_models/sentence-transformers--clip-ViT-B-32"
    #text_model_path = r"pretrained_models/sentence-transformers--clip-ViT-B-32-multilingual-v1"
    text_model_path = r"output\finetuned_text"

    # Load vision model and processor
    img_model = CLIPModel.from_pretrained(img_model_path)

    # Load multilingual text model and tokenizer
    # Load projection config
    with open("pretrained_models/sentence-transformers--clip-ViT-B-32-multilingual-v1/2_Dense/config.json") as f:
        proj_cfg = json.load(f)
    # Load DistilBERT model and tokenizer
    text_model_path = "pretrained_models/sentence-transformers--clip-ViT-B-32-multilingual-v1"
    text_model = DistilBertModel.from_pretrained(text_model_path)
    # tokenizer = DistilBertTokenizer.from_pretrained(text_model_path)

    # # Load projection weights
    # PROJ_WEIGHTS_PATH = "pretrained_models/sentence-transformers--clip-ViT-B-32-multilingual-v1/2_Dense/pytorch_model.bin"
    # projection = torch.nn.Linear(proj_cfg["in_features"], proj_cfg["out_features"], bias=proj_cfg["bias"])
    # projection.load_state_dict(torch.load(PROJ_WEIGHTS_PATH))
    
    df = pd.read_csv(CSV_PATH)
    
    image_embeddings = load_embeddings(IMG_EMB_PATH)
    text_embeddings = load_embeddings(TXT_EMB_PATH)
    # alternatevly, get a subset of a specific split
    test_df, test_img_emb, test_txt_emb = get_split_embeddings(df, image_embeddings, text_embeddings, "test")

    query = "red dress"
    #query= "ein fusslanges kleid ohne ärmel und einem blaulichem farbmuster das in der mitte eine art pyramidenstimmung erzeugt"
    print("Text-to-Image Retrieval Example:")
    results = retrieve_images_by_text(query, text_model, image_embeddings, df,  top_k=5)
    plot_images(results, "Text-to-Image Retrieval (M-CLIP)", query=query, query_type="text")

    print("\nImage-to-Image Retrieval Example:")
    example_image = results[0][0]
    print(f"Using example image: {example_image}")
    results = retrieve_images_by_image(example_image, img_model, image_embeddings, df, top_k=5)
    plot_images(results, "Image-to-Image Retrieval (M-CLIP)", query=example_image, query_type="image")

    ########### Same QUery Only in the test split ###########
    print("Text-to-Image Retrieval Example Test:")
    results = retrieve_images_by_text(query, text_model, test_img_emb, test_df,  top_k=5)
    plot_images(results, "Text-to-Image Retrieval (M-CLIP)", query=query, query_type="text")

    print("\nImage-to-Image Retrieval Example Test:")
    example_image = results[0][0]
    print(f"Using example image: {example_image}")
    results = retrieve_images_by_image(example_image, img_model, test_img_emb, test_df, top_k=5)
    plot_images(results, "Image-to-Image Retrieval (M-CLIP)", query=example_image, query_type="image")
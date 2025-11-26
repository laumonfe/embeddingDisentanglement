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
from compute_embeddings import load_or_compute_embeddings


def retrieve_images_by_text(query, text_model, image_embeddings, df, top_k=5):
    with torch.no_grad():
        text_emb = text_model.encode([query])
    sims = util.cos_sim(text_emb, image_embeddings)[0]  # shape: (num_images,)
    # Get indices sorted by similarity (descending)
    sorted_indices = torch.argsort(sims, descending=True).tolist()
    seen = set()
    results = []
    for i in sorted_indices:
        img_path = df["image_path"][i]
        if img_path not in seen:
            seen.add(img_path)
            results.append((img_path, sims[i].item()))
        if len(results) == top_k:
            break
    return results

def retrieve_images_by_image(query, image_model, image_embeddings, df,  top_k=5):
    with torch.no_grad():
        image_emb = image_model.encode(query)
    sims = util.cos_sim(image_emb, image_embeddings)[0]  # shape: (num_images,)

    # Exclude all images whose embedding is identical to the query embedding
    identical_mask = np.all(image_embeddings == image_emb, axis=1)
    sims[identical_mask] = -float("inf")

    sorted_indices = torch.argsort(sims, descending=True).tolist()
    seen = set()
    results = []
    for i in sorted_indices:
        img_path = df["image_path"][i]
        if img_path not in seen:
            seen.add(img_path)
            results.append((img_path, sims[i].item()))
        if len(results) == top_k:
            break
    return results

def plot_images(results, title):
    plt.figure(figsize=(12, 6))
    for i, (img_path, score) in enumerate(results):
        plt.subplot(1, len(results), i + 1)
        img = Image.open(img_path)
        plt.imshow(img)
        plt.title(f"{os.path.basename(img_path)}\nscore: {score:.2f}",fontsize=10, pad=10)
        plt.axis('off')
    plt.suptitle(title,fontsize=14)
    plt.tight_layout()
    plt.show()


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

if __name__ == "__main__":
    #query = "ein wunderschönes und sehr festliches langes Kleid" # "a beautiful and very festive long dress"
    #query = "ein kurzes schwarzes Kleid"  # a short black dress
    #query = "ein glitzerndes und schickes Kleid"  # a glitter and fancy dress
    #query = "ein grünes Samtkleid mit V-Ausschnitt und langen Ärmeln" #"a velvet green dress with a V-neck and long sleeves" 
    # query = "A dress that whispers rebellion." #"spring dress perfect for a picnic date"  #"a red dress with floral pattern"
    # print("Text-to-Image Retrieval Example:")
    # query = "A dress that whispers rebellion."
    # results = retrieve_images_by_text(query, top_k=5)
    # plot_images(results, "Text-to-Image Retrieval (M-CLIP)")

    # print("\nImage-to-Image Retrieval Example:")
    # example_image = results[0][0]
    # example_idx = filtered_image_paths.index(example_image)
    # print(f"Using example image: {example_image}")
    # print(f"Example image index: {example_idx}")
    # results = retrieve_images_by_image(example_image, top_k=5)
    # plot_images(results, "Image-to-Image Retrieval (M-CLIP)")


    CSV_PATH = r"data\embeddings\baseline_clip-ViT-B-32-multilingual-v1\all_split.csv"
    IMG_EMB_PATH = r"data\embeddings\baseline_clip-ViT-B-32-multilingual-v1\clip_image_embeddings_all.npy"
    TXT_EMB_PATH = r"data\embeddings\baseline_clip-ViT-B-32-multilingual-v1\clip_text_embeddings_all.npy"

    img_model = SentenceTransformer('clip-ViT-B-32')
    text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')

    df = pd.read_csv(CSV_PATH)
    
    print("Number of images in DataFrame:", len(df))
    image_embeddings = load_or_compute_embeddings(img_model, df['image_path'].tolist(), IMG_EMB_PATH)
    text_embeddings = load_or_compute_embeddings(text_model, df['text'].tolist(), TXT_EMB_PATH)

    query = "red dress "
    print("Text-to-Image Retrieval Example:")
    results = retrieve_images_by_text(query, text_model, image_embeddings, df,  top_k=5)
    plot_images(results, "Text-to-Image Retrieval (M-CLIP)", query=query, query_type="text")

    print("\nImage-to-Image Retrieval Example:")
    example_image = results[0][0]
    print(f"Using example image: {example_image}")
    results = retrieve_images_by_image(example_image, img_model, image_embeddings, df, top_k=5)
    plot_images(results, "Image-to-Image Retrieval (M-CLIP)", query=example_image, query_type="image")
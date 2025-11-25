import collections
import os
import pandas as pd
import torch

from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util

# Paths
CSV_PATH = "visualization_explorer/feidegger_visualization_data.csv"
IMG_EMB_PATH = "data/feidegger_clip-ViT-B-32_image_embeddings_baseline2.npy"
TXT_EMB_PATH = "data/feidegger_clip-ViT-B-32-multilingual-v1_text_embeddings_baseline2.npy"

#Load dataset
df = pd.read_csv(CSV_PATH)
image_paths = df['image_path'].tolist()
texts = df['text'].tolist()

# Get indices of valid images in the original DataFrame
valid_indices = [i for i, img_path in enumerate(image_paths) if os.path.exists(img_path)]

# Create a filtered DataFrame as a copy of the original, only with valid items
filtered_df = df.iloc[valid_indices].reset_index(drop=True)
print(f"Filtered dataset to {len(filtered_df)} items with existing images out of {len(df)} total.")
# Save the filtered DataFrame to a new CSV
filtered_df.to_csv("visualization_explorer/feidegger_visualization_data_valid.csv", index=False)
print("Saved filtered DataFrame to visualization_explorer/feidegger_visualization_data_valid.csv")

# Filter for existing images
filtered_image_paths = []
filtered_texts = []
for img_path, text in zip(image_paths, texts):
    if os.path.exists(img_path):
        filtered_image_paths.append(img_path)
        filtered_texts.append(text)

def compute_image_embeddings(img_model, image_paths, emb_path):
    need_recompute = False
    if os.path.exists(emb_path):
        print(f"Loading image embeddings from {emb_path}")
        image_embeddings = np.load(emb_path)
        if len(image_embeddings) != len(image_paths):
            print("Warning: Loaded image embeddings do not match filtered image paths. Recomputing embeddings.")
            need_recompute = True
    else:
        need_recompute = True

    if need_recompute:
        print("Calculating image embeddings...")
        image_embeddings = []
        for img_path in tqdm(image_paths, total=len(image_paths)):
            if not os.path.exists(img_path):
                continue
            try:
                with torch.no_grad():
                    emb = img_model.encode(img_path)
                image_embeddings.append(emb)
            except Exception as e:
                print(f"Could not process {img_path}: {e}")
        image_embeddings = np.array(image_embeddings)
        os.makedirs(os.path.dirname(emb_path), exist_ok=True)
        np.save(emb_path, image_embeddings)
        print(f"Saved image embeddings to {emb_path}")
    return image_embeddings

def compute_text_embeddings(text_model, texts, emb_path):
    if os.path.exists(emb_path):
        print(f"Loading text embeddings from {emb_path}")
        text_embeddings = np.load(emb_path)
        if len(text_embeddings) != len(texts):
            print("Warning: Loaded text embeddings do not match filtered texts. Recomputing embeddings.")
            with torch.no_grad():
                text_embeddings = text_model.encode(texts)
            os.makedirs(os.path.dirname(emb_path), exist_ok=True)
            np.save(emb_path, text_embeddings)
    else:
        print("Calculating text embeddings...")
        with torch.no_grad():
            text_embeddings = text_model.encode(texts)
        os.makedirs(os.path.dirname(emb_path), exist_ok=True)
        np.save(emb_path, text_embeddings)
        print(f"Saved text embeddings to {emb_path}")
    return text_embeddings

def retrieve_images_by_text(query, text_model, image_embeddings, filtered_image_paths, top_k=5):
    with torch.no_grad():
        text_emb = text_model.encode([query])
        # Ensure both are torch tensors
        if not isinstance(text_emb, torch.Tensor):
            text_emb = torch.tensor(text_emb)
        if not isinstance(image_embeddings, torch.Tensor):
            image_embeddings = torch.tensor(image_embeddings)
    sims = util.cos_sim(text_emb, image_embeddings)[0]
    sorted_indices = torch.argsort(sims, descending=True).tolist()
    seen = set()
    results = []
    for i in sorted_indices:
        img_path = filtered_image_paths[i]
        if img_path not in seen:
            seen.add(img_path)
            results.append((img_path, sims[i].item()))
        if len(results) == top_k:
            break
    return results

def retrieve_images_by_image(image_path, img_model, image_embeddings, filtered_image_paths, top_k=5):
    with torch.no_grad():
        image_emb = img_model.encode(image_path)
        if not isinstance(image_emb, torch.Tensor):
            image_emb = torch.tensor(image_emb)
        if not isinstance(image_embeddings, torch.Tensor):
            image_embeddings = torch.tensor(image_embeddings)
    sims = util.cos_sim(image_emb, image_embeddings)[0]
    identical_mask = np.all(image_embeddings.numpy() == image_emb.numpy(), axis=1)
    sims[identical_mask] = -float("inf")
    sorted_indices = torch.argsort(sims, descending=True).tolist()
    seen = set()
    results = []
    for i in sorted_indices:
        img_path = filtered_image_paths[i]
        if img_path not in seen:
            seen.add(img_path)
            results.append((img_path, sims[i].item()))
        if len(results) == top_k:
            break
    return results

def plot_images(results, title, query=None, query_type="text"):
    import matplotlib.patches as patches
    n_results = len(results)
    if query_type == "image":
        plt.figure(figsize=(3 * (n_results + 1), 6))
        ax = plt.subplot(1, n_results + 1, 1)
        img = Image.open(query)
        plt.imshow(img)
        rect = patches.Rectangle(
            (0, 0), img.size[0], img.size[1],
            linewidth=4, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        plt.title("Query Image", fontsize=10, pad=10, color='red', weight="bold")
        plt.axis('off')
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
        if query is not None:
            fig.text(0.5, 0.91, f"Query: {query}", ha='center', fontsize=10, color='red', weight="bold")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Select and load models here
    img_model = SentenceTransformer('clip-ViT-B-32')
    text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')

    # Compute or load embeddings
    image_embeddings = compute_image_embeddings(img_model, filtered_image_paths, IMG_EMB_PATH)
    text_embeddings = compute_text_embeddings(text_model, filtered_texts, TXT_EMB_PATH)

    query = "A hauntingly beautiful dress."
    print("Text-to-Image Retrieval Example:")
    results = retrieve_images_by_text(query, text_model, image_embeddings, filtered_image_paths, top_k=5)
    plot_images(results, "Text-to-Image Retrieval (M-CLIP)", query=query, query_type="text")

    print("\nImage-to-Image Retrieval Example:")
    example_image = results[0][0]
    print(f"Using example image: {example_image}")
    results = retrieve_images_by_image(example_image, img_model, image_embeddings, filtered_image_paths, top_k=5)
    plot_images(results, "Image-to-Image Retrieval (M-CLIP)", query=example_image, query_type="image")
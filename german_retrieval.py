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
IMG_EMB_PATH = "data/feidegger_clip-ViT-B-32_image_embeddings_baseline.npy"
TXT_EMB_PATH = "data/feidegger_clip-ViT-B-32-multilingual-v1_text_embeddings_baseline.npy"

# We use the original clip-ViT-B-32 for encoding images
img_model = SentenceTransformer('clip-ViT-B-32')
# Our text embedding model is aligned to the img_model and maps 50+
# languages to the same vector space
text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')


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

# Compute or load image embeddings
if os.path.exists(IMG_EMB_PATH):
    print(f"Loading image embeddings from {IMG_EMB_PATH}")
    image_embeddings = np.load(IMG_EMB_PATH)
    # Ensure filtered_image_paths matches loaded embeddings
    if len(image_embeddings) != len(filtered_image_paths):
        print("Warning: Loaded image embeddings do not match filtered image paths. Recomputing embeddings.")
        image_embeddings = []
        filtered_image_paths = []
        filtered_texts = []
elif True:
    print("Calculating image embeddings...")
    image_embeddings = []
    valid_image_paths = []
    valid_texts = []
    for img_path, text in tqdm(zip(image_paths, texts), total=len(image_paths)):
        if not os.path.exists(img_path):
            continue
        try:
            with torch.no_grad():
                emb = img_model.encode(Image.open(img_path))
            image_embeddings.append(emb)
            valid_image_paths.append(img_path)
            valid_texts.append(text)
        except Exception as e:
            print(f"Could not process {img_path}: {e}")
    image_embeddings = np.array(image_embeddings)
    filtered_image_paths = valid_image_paths
    filtered_texts = valid_texts
    os.makedirs(os.path.dirname(IMG_EMB_PATH), exist_ok=True)
    np.save(IMG_EMB_PATH, image_embeddings)
    print(f"Saved image embeddings to {IMG_EMB_PATH}")

# For text embeddings (only for filtered_texts)
if os.path.exists(TXT_EMB_PATH):
    print(f"Loading text embeddings from {TXT_EMB_PATH}")
    text_embeddings = np.load(TXT_EMB_PATH)
    if len(text_embeddings) != len(filtered_texts):
        print("Warning: Loaded text embeddings do not match filtered texts. Recomputing embeddings.")
        with torch.no_grad():
            text_embeddings = text_model.encode(filtered_texts)
        os.makedirs(os.path.dirname(TXT_EMB_PATH), exist_ok=True)
        np.save(TXT_EMB_PATH, text_embeddings)
else:
    print("Calculating text embeddings...")
    with torch.no_grad():
        text_embeddings = text_model.encode(filtered_texts)
    os.makedirs(os.path.dirname(TXT_EMB_PATH), exist_ok=True)
    np.save(TXT_EMB_PATH, text_embeddings)
    print(f"Saved text embeddings to {TXT_EMB_PATH}")

def retrieve_images_by_text(query, top_k=5):
    with torch.no_grad():
        text_emb = text_model.encode([query])
    sims = util.cos_sim(text_emb, image_embeddings)[0]  # shape: (num_images,)
    # Get indices sorted by similarity (descending)
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

def retrieve_images_by_image(image_path, top_k=5):
    with torch.no_grad():
        image_emb = img_model.encode(Image.open(image_path))
    sims = util.cos_sim(image_emb, image_embeddings)[0]  # shape: (num_images,)

    # Exclude all images whose embedding is identical to the query embedding
    identical_mask = np.all(image_embeddings == image_emb, axis=1)
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

import matplotlib.patches as patches
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

    query = "A hauntingly beautiful dress."
    print("Text-to-Image Retrieval Example:")
    results = retrieve_images_by_text(query, top_k=5)
    plot_images(results, "Text-to-Image Retrieval (M-CLIP)", query=query, query_type="text")

    print("\nImage-to-Image Retrieval Example:")
    example_image = results[0][0]
    print(f"Using example image: {example_image}")
    results = retrieve_images_by_image(example_image, top_k=5)
    plot_images(results, "Image-to-Image Retrieval (M-CLIP)", query=example_image, query_type="image")
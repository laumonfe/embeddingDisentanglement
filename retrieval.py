import os
import pandas as pd
import torch
import clip
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
# pip install git+https://github.com/openai/CLIP.git
# Paths
CSV_PATH = "visualization_explorer/feidegger_visualization_data.csv"
IMG_EMB_PATH = "data/feidegger_clip_image_embeddings.npy"
TXT_EMB_PATH = "data/feidegger_clip_text_embeddings.npy"

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load dataset
df = pd.read_csv(CSV_PATH)

df = df.drop_duplicates(subset='image_path').iloc[:100]
image_paths = df['image_path'].tolist()
texts = df['text'].tolist()

# def truncate_text(text, max_tokens=76):
#     # Use CLIP's SimpleTokenizer to count tokens and truncate safely
#     print("Truncating text to fit within token limit...")
#     tokenizer = clip.simple_tokenizer.SimpleTokenizer()
#     tokens = tokenizer.encode(text)
#     if len(tokens) > max_tokens:
#         truncated = tokenizer.decode(tokens[:max_tokens])
#         return truncated
#     return text

# # Truncate all texts before embedding
# texts_trunc = [truncate_text(t) for t in texts]

# Compute or load image embeddings
if os.path.exists(IMG_EMB_PATH):
    print(f"Loading image embeddings from {IMG_EMB_PATH}")
    image_embeddings = np.load(IMG_EMB_PATH)
else:
    print("Calculating image embeddings...")
    image_embeddings = []
    for path in tqdm(image_paths):
        try:
            image = preprocess(Image.open(path)).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model.encode_image(image)
            image_embeddings.append(emb.cpu().numpy())
        except Exception as e:
            print(f"Could not process {path}: {e}")
            image_embeddings.append(np.zeros((512,)))
    image_embeddings = np.vstack(image_embeddings)
    image_embeddings /= np.linalg.norm(image_embeddings, axis=1, keepdims=True)
    os.makedirs(os.path.dirname(IMG_EMB_PATH), exist_ok=True)
    np.save(IMG_EMB_PATH, image_embeddings)
    print(f"Saved image embeddings to {IMG_EMB_PATH}")

# Compute or load text embeddings (using truncated texts)
if os.path.exists(TXT_EMB_PATH):
    print(f"Loading text embeddings from {TXT_EMB_PATH}")
    text_embeddings = np.load(TXT_EMB_PATH)
else:
    print("Calculating text embeddings...")
    text_embeddings = []
    skipped_texts = 0
    for t in tqdm(texts):
        try:
            text_token = clip.tokenize([t]).to(device)
            with torch.no_grad():
                emb = model.encode_text(text_token)
            text_embeddings.append(emb.cpu().numpy()[0])
        except Exception as e:
            #print(f"Could not process text: {e}")
            text_embeddings.append(np.zeros((512,)))
            skipped_texts += 1
    print(f"Skipped {skipped_texts} texts due to errors.")
    text_embeddings = np.vstack(text_embeddings)
    # with torch.no_grad():
    #     text_embeddings = model.encode_text(text_tokens).cpu().numpy()
    text_embeddings /= np.linalg.norm(text_embeddings, axis=1, keepdims=True)
    os.makedirs(os.path.dirname(TXT_EMB_PATH), exist_ok=True)
    np.save(TXT_EMB_PATH, text_embeddings)
    print(f"Saved text embeddings to {TXT_EMB_PATH}")

def retrieve_images_by_text(query, top_k=5):
    # query_trunc = truncate_text(query)
    text = clip.tokenize([query]).to(device)
    with torch.no_grad():
        text_emb = model.encode_text(text).cpu().numpy()
    text_emb /= np.linalg.norm(text_emb)
    sims = image_embeddings @ text_emb.T  # cosine similarity
    top_indices = np.argsort(sims.ravel())[::-1][:top_k]
    return [(image_paths[i], sims[i][0]) for i in top_indices]

def retrieve_images_by_image(image_path, top_k=5):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_emb = model.encode_image(image).cpu().numpy()
    image_emb /= np.linalg.norm(image_emb)
    sims = image_embeddings @ image_emb.T  # cosine similarity
    top_indices = np.argsort(sims.ravel())[::-1][:top_k]
    return [(image_paths[i], sims[i][0]) for i in top_indices]

def plot_images(results, title):
    plt.figure(figsize=(12, 4))
    for i, (img_path, score) in enumerate(results):
        plt.subplot(1, len(results), i + 1)
        img = Image.open(img_path)
        plt.imshow(img)
        plt.title(f"{os.path.basename(img_path)}\nscore: {score:.2f}")
        plt.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def show_all_images(image_paths, title="All Dresses"):
    n = len(image_paths)
    cols = 5
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(3 * cols, 3 * rows))
    for i, img_path in enumerate(image_paths):
        plt.subplot(rows, cols, i + 1)
        img = Image.open(img_path)
        plt.imshow(img)
        plt.title(os.path.basename(img_path), fontsize=8)
        plt.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":


    print(f"Image embeddings shape: {image_embeddings.shape}")
    print(f"Text embeddings shape: {text_embeddings.shape}")

    print("Any NaNs in image embeddings?", np.isnan(image_embeddings).any())
    print("Any NaNs in text embeddings?", np.isnan(text_embeddings).any())

    print("Number of NaNs in image embeddings:", np.isnan(image_embeddings).sum())
    print("Number of NaNs in text embeddings:", np.isnan(text_embeddings).sum())

    print("Any all-zero image embeddings?", np.all(image_embeddings == 0, axis=1).sum())
    print("Any all-zero text embeddings?", np.all(text_embeddings == 0, axis=1).sum())

    print("First image embedding:", image_embeddings[0][:5])
    print("First text embedding:", text_embeddings[0][:5])


    print("Text-to-Image Retrieval Example:")
    query = "a short black dress"#"ein wunderschönes und sehr festliches langes Kleid" # "a beautiful and very festive long dress"
    results = retrieve_images_by_text(query, top_k=3)
    plot_images(results, "Text-to-Image Retrieval")

    print("\nImage-to-Image Retrieval Example:")
    example_image = results[0][0]
    results = retrieve_images_by_image(example_image, top_k=3)
    plot_images(results, "Image-to-Image Retrieval")

    print("Showing all dresses in the dataset:")
    #show_all_images(image_paths, "All Dresses")

    print("Text-to-Image: Scores of query with all texts")
    query = "a short black dress"
    text = clip.tokenize([query]).to(device)
    with torch.no_grad():
        text_emb = model.encode_text(text).cpu().numpy()
    text_emb /= np.linalg.norm(text_emb)
    sims = image_embeddings @ text_emb.T  # shape: (num_images, 1)
    for i, (img_path, score) in enumerate(zip(image_paths, sims.ravel())):
        print(f"{i:03d}: {img_path} (score: {score:.3f})")

    print("\nImage-to-Image: Scores of image 42 with all other images")
    example_image = image_paths[42]
    image = preprocess(Image.open(example_image)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_emb = model.encode_image(image).cpu().numpy()
    image_emb /= np.linalg.norm(image_emb)
    sims_img = image_embeddings @ image_emb.T  # shape: (num_images, 1)
    for i, (img_path, score) in enumerate(zip(image_paths, sims_img.ravel())):
        print(f"{i:03d}: {img_path} (score: {score:.3f})")
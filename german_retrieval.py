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
IMG_EMB_PATH = "data/feidegger_mclip_image_embeddings.npy"
TXT_EMB_PATH = "data/feidegger_mclip_text_embeddings.npy"

# We use the original clip-ViT-B-32 for encoding images
img_model = SentenceTransformer('clip-ViT-B-32')
# Our text embedding model is aligned to the img_model and maps 50+
# languages to the same vector space
text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')



# Load dataset
df = pd.read_csv(CSV_PATH)
df = df.drop_duplicates(subset='image_path').iloc[:100]
image_paths = df['image_path'].tolist()
texts = df['text'].tolist()

# Compute or load image embeddings
if os.path.exists(IMG_EMB_PATH):
    print(f"Loading image embeddings from {IMG_EMB_PATH}")
    image_embeddings = np.load(IMG_EMB_PATH)
else:
    print("Calculating image embeddings...")
    image_embeddings = []
    for path in tqdm(image_paths):
        try:
            with torch.no_grad():
                emb = img_model.encode(Image.open(path))
            image_embeddings.append(emb)
        except Exception as e:
            print(f"Could not process {path}: {e}")


    os.makedirs(os.path.dirname(IMG_EMB_PATH), exist_ok=True)
    np.save(IMG_EMB_PATH, image_embeddings)
    print(f"Saved image embeddings to {IMG_EMB_PATH}")



# For text embeddings
if os.path.exists(TXT_EMB_PATH):
    print(f"Loading text embeddings from {TXT_EMB_PATH}")
    text_embeddings = np.load(TXT_EMB_PATH)
else:
    print("Calculating text embeddings...")
    with torch.no_grad():
        text_embeddings = text_model.encode(texts)
    text_embeddings = text_embeddings
    os.makedirs(os.path.dirname(TXT_EMB_PATH), exist_ok=True)
    np.save(TXT_EMB_PATH, text_embeddings)
    print(f"Saved text embeddings to {TXT_EMB_PATH}")

# In retrieve_images_by_text
def retrieve_images_by_text(query, top_k=5):
    with torch.no_grad():
        text_emb = text_model.encode([query])
    sims = util.cos_sim(text_emb, image_embeddings)[0]  # shape: (num_images,)
    top_indices = torch.topk(sims, top_k).indices.tolist()
    return [(image_paths[i], sims[i].item()) for i in top_indices]

def retrieve_images_by_image(image_path, top_k=5):
    with torch.no_grad():
        image_emb = img_model.encode(Image.open(image_path))
    sims = util.cos_sim(image_emb, image_embeddings)[0]  # shape: (num_images,)
    top_indices = torch.topk(sims, top_k).indices.tolist()
    return [(image_paths[i], sims[i].item()) for i in top_indices]

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

if __name__ == "__main__":
    print("Text-to-Image Retrieval Example:")
    #query = "ein wunderschönes und sehr festliches langes Kleid" # "a beautiful and very festive long dress"
    #query = "ein kurzes schwarzes Kleid"  # a short black dress
    #query = "ein glitzerndes und schickes Kleid"  # a glitter and fancy dress
    #query = "ein grünes Samtkleid mit V-Ausschnitt und langen Ärmeln" #"a velvet green dress with a V-neck and long sleeves" 
    query = "A dress that whispers rebellion." #"spring dress perfect for a picnic date"  #"a red dress with floral pattern"
    results = retrieve_images_by_text(query, top_k=5)
    plot_images(results, "Text-to-Image Retrieval (M-CLIP)")

    print("\nImage-to-Image Retrieval Example:")
    example_image = results[0][0]
    results = retrieve_images_by_image(example_image, top_k=3)
    plot_images(results, "Image-to-Image Retrieval (M-CLIP)")
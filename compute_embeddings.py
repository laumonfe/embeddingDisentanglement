import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image

from sentence_transformers import SentenceTransformer



def load_embeddings(emb_save_path):
    if os.path.exists(emb_save_path):
        print(f"Loading embeddings from {emb_save_path}")
        embeddings = np.load(emb_save_path, allow_pickle=True)
        print("Contains:", len(embeddings) , "embeddings.")
        return embeddings
    else:
        print(f"Embeddings file {emb_save_path} not found.")
        return None

# def load_or_compute_embeddings(model, inputs, emb_save_path):
#     need_recompute = True
#     if os.path.exists(emb_save_path):
#         print(f"Loading embeddings from {emb_save_path}")
#         embeddings = np.load(emb_save_path)
#         if len(embeddings) == len(inputs):
#             need_recompute = False
#         else:
#             print("Warning: Loaded embeddings do not match input length. Recomputing embeddings.")

#     if need_recompute:
#         print("Calculating embeddings...")
#         embeddings = []
#         for inp in tqdm(inputs, total=len(inputs)):
#             try:
#                 with torch.no_grad():
#                     emb = model.encode(inp)
#                 embeddings.append(emb)
#             except Exception as e:
#                 print(f"Error processing {inp}: {e}")
#         embeddings = np.array(embeddings)
#         os.makedirs(os.path.dirname(emb_save_path), exist_ok=True)
#         np.save(emb_save_path, embeddings)
#         print("Contains:", len(embeddings) , "embeddings.")
#         print(f"Saved embeddings to {emb_save_path}")
#     return embeddings

#item_idx,desc_idx,image_path,original_split,mamba_split,text

def compute_embeddings(text_model, image_model, df, img_emb_save_path, txt_emb_save_path):
    img_embeddings = []
    text_embeddings = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Computing embeddings..."):
        item_idx = row.item_idx
        desc_idx = row.desc_idx
        image_path = row.image_path
        text = row.text
        if not os.path.exists(image_path):
            print(f"Missing image: {image_path}")
            continue
        if not isinstance(text, str) or not text.strip():
            print(f"Invalid text for item {item_idx} {desc_idx}")
            continue
        try:
            img_emb = image_model.encode(Image.open(image_path))
            text_emb = text_model.encode(text)
            img_embeddings.append({'idx': item_idx, 'desc_idx': desc_idx, 'embedding': np.array(img_emb)})
            text_embeddings.append({'idx': item_idx, 'desc_idx': desc_idx, 'embedding': np.array(text_emb)})
        except Exception as e:
            print(f"Error processing {item_idx}: {e}")

    img_embeddings = np.array(img_embeddings, dtype=object)
    text_embeddings = np.array(text_embeddings, dtype=object)
    if dir_path := os.path.dirname(txt_emb_save_path):
        os.makedirs(dir_path, exist_ok=True)
    if dir_path := os.path.dirname(img_emb_save_path):
        os.makedirs(dir_path, exist_ok=True)
    np.save(img_emb_save_path, img_embeddings)
    np.save(txt_emb_save_path, text_embeddings)
    print("Images contain:", len(img_embeddings), "embeddings.")
    print("Text contains:", len(text_embeddings), "embeddings.")
    print(f"Saved embeddings to {img_emb_save_path} and {txt_emb_save_path}")
    return img_embeddings, text_embeddings

# def load_or_compute_embeddings(model, df, input_column, emb_column, emb_save_path):
#     """
#     Computes embeddings for df[input_column] using model, saves to emb_save_path,
#     and stores the embeddings in df[emb_column].
#     """
#     inputs = df[input_column].tolist()
#     need_recompute = True
#     if os.path.exists(emb_save_path):
#         print(f"Loading embeddings from {emb_save_path}")
#         embeddings = np.load(emb_save_path)
#         if len(embeddings) == len(inputs):
#             need_recompute = False
#         else:
#             print("Warning: Loaded embeddings do not match input length. Recomputing embeddings.")

#     if need_recompute:
#         print("Calculating embeddings...")
#         embeddings = []
#         for inp in tqdm(inputs, total=len(inputs)):
#             try:
#                 with torch.no_grad():
#                     emb = model.encode(inp)
#                 embeddings.append(emb)
#             except Exception as e:
#                 print(f"Error processing {inp}: {e}")
#         embeddings = np.array(embeddings)
#         os.makedirs(os.path.dirname(emb_save_path), exist_ok=True)
#         np.save(emb_save_path, embeddings)
#         print("Contains:", len(embeddings), "embeddings.")
#         print(f"Saved embeddings to {emb_save_path}")

#     # Save embeddings in the DataFrame
#     df[emb_column] = list(embeddings)
#     return df
def get_split_data(df, split_name):
    """
    Extracts image paths and texts for a given split from the DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing at least the columns 'mamba_split', 'image_path', and 'text'.
        split_name (str): The name of the split to filter by (e.g., 'train', 'val', 'test').

    Returns:
        tuple: (image_paths, texts)
            image_paths (list): List of image paths for the specified split.
            texts (list): List of texts corresponding to the image paths for the specified split.
    """

def get_split_data(df, split_name):
    split_df = df[df["mamba_split"] == split_name]
    image_paths = split_df['image_path'].tolist()
    texts = split_df['text'].tolist()
    return image_paths, texts




if __name__ == "__main__":

    CSV_PATH = "visualization_explorer/feidegger_visualization_data_valid.csv"
    df = pd.read_csv(CSV_PATH)

    # Filter for valid images
    # valid_indices = [i for i, img_path in enumerate(df['image_path']) if os.path.exists(img_path)]
    # filtered_df = df.iloc[valid_indices].reset_index(drop=True)

    img_model = SentenceTransformer('clip-ViT-B-32')
    #text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')
    text_model = SentenceTransformer('clip-ViT-B-32')

    # Calculate "all" split embeddings once
    # all_image_paths = df['image_path'].tolist()
    # all_texts = df['text'].tolist()
    # save directory
    img_emb_path_all = "data/testing/clip_image_embeddings_2.npy"
    txt_emb_path_all = "data/testing/clip_text_embeddings_2.npy"
    # csv_path = "data/embeddings/with_embeddings.csv"

    # image_embeddings_all = load_or_compute_embeddings(img_model, all_image_paths, img_emb_path_all)
    # text_embeddings_all = load_or_compute_embeddings(text_model, all_texts, txt_emb_path_all)
    image_embeddings_all, text_embeddings_all = compute_embeddings(
        text_model, img_model, df, img_emb_path_all, txt_emb_path_all
    )

    # if len(text_embeddings_all) == 0 or len(image_embeddings_all) == 0:
    #     print("Error: Embeddings for 'all' split are empty. Aborting split saving.")
    #     exit()
    # print(f"Saved all split DataFrame to {csv_path}")
    # # Save split embeddings by slicing the "all" arrays
    # for split in ["train", "val", "test"]:
    #     split_df = filtered_df[filtered_df["mamba_split"] == split]
    #     print(f"{split} split length: {len(split_df)}")  
    #     split_indices = split_df.index.tolist()
    #     split_img_emb_path = f"data/embeddings/clip_image_embeddings_{split}.npy"
    #     split_txt_emb_path = f"data/embeddings/clip_text_embeddings_{split}.npy"
    #     np.save(split_img_emb_path, image_embeddings_all[split_indices])
    #     np.save(split_txt_emb_path, text_embeddings_all[split_indices])    
    #     print(f"Saved {split} image embeddings to {split_img_emb_path}")
    #     print(f"Saved {split} text embeddings to {split_txt_emb_path}")

    #     # Save split DataFrame as CSV
    #     split_csv_path = f"data/embeddings/{split}_split.csv"
    #     split_df.to_csv(split_csv_path, index=False)
    #     print(f"Saved {split} split DataFrame to {split_csv_path}")
    
    #filtered_df.to_csv(csv_path, index=False)   
    #print("Done computing and saving embeddings for all splits.")
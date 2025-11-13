def load_data(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    return df

def preprocess_data(df, text_column='text', image_column='image_path'):
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame.")
    
    if image_column and image_column not in df.columns:
        df[image_column] = None  # Add a column for images if it doesn't exist

    return df

def save_embeddings(embeddings, path):
    import numpy as np
    np.save(path, embeddings)

def load_embeddings(path):
    import numpy as np
    return np.load(path) if os.path.exists(path) else None

def check_image_path(image_path):
    import os
    return os.path.exists(image_path) and os.path.isfile(image_path)
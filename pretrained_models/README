# Pretrained Models Setup

This project uses pretrained CLIP models from HuggingFace via the `sentence-transformers` library.  
The required model files are stored in the `pretrained_models/` directory.

## How to Download Pretrained Models

1. **Run the following Python script** to automatically download the models and copy them to `pretrained_models/`:

    ```
    python setup_pretrained_models.py
    ```

2. **What the script does:**
    - Downloads the models `'clip-ViT-B-32'` and `'sentence-transformers/clip-ViT-B-32-multilingual-v1'` using `SentenceTransformer`.
    - Copies the downloaded files from your HuggingFace cache to `pretrained_models/`.

3. **If you want to do it manually:**
    - Run:
        ```python
        from sentence_transformers import SentenceTransformer
        SentenceTransformer('clip-ViT-B-32')
        SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')
        ```
    - Find the downloaded files in your HuggingFace cache (usually at `~/.cache/huggingface/hub/`).
    - Copy the relevant folders to `pretrained_models/`.

## Note

- Large model files are **not included in the repository**. You must run the setup script to download them.
- The `pretrained_models/` folder is listed in `.gitignore` and should not be committed to git.

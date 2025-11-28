import os
import shutil
from huggingface_hub import snapshot_download

def download_and_copy(model_name, target_dir, subfolder=None):
    print(f"Downloading {model_name} ...")
    snapshot_path = snapshot_download(repo_id=model_name, cache_dir=None, allow_patterns=["*"])
    print(f"Model downloaded to {snapshot_path}")
    # If a subfolder is specified, copy only that subfolder
    if subfolder:
        src = os.path.join(snapshot_path, subfolder)
        dst = os.path.join(target_dir, model_name.replace("/", "--"))
        if not os.path.exists(dst):
            shutil.copytree(src, dst)
            print(f"Copied {src} to {dst}")
        else:
            print(f"Target path {dst} already exists, skipping copy.")
    else:
        dst = os.path.join(target_dir, model_name.replace("/", "--"))
        if not os.path.exists(dst):
            shutil.copytree(snapshot_path, dst)
            print(f"Copied {snapshot_path} to {dst}")
        else:
            print(f"Target path {dst} already exists, skipping copy.")

if __name__ == "__main__":
    os.makedirs("pretrained_models", exist_ok=True)
    # For clip-ViT-B-32, only copy the 0_CLIPModel subfolder
    download_and_copy("sentence-transformers/clip-ViT-B-32", "pretrained_models", subfolder="0_CLIPModel")
    # For multilingual, copy the whole snapshot
    download_and_copy("sentence-transformers/clip-ViT-B-32-multilingual-v1", "pretrained_models")
    print("All models downloaded and copied.")



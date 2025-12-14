import os
import subprocess

# List all model/dataset combinations
configs = [
    ("disentangled", "default"),
    ("disentangled", "grouped")
    # ("finetuned", "default"),
    # ("finetuned", "grouped")
]

for model_kind, dataset_type in configs:
    print(f"\nTraining model: {model_kind}, {dataset_type}")
    output_dir = f"/mnt/netstorage/projects/clip/{model_kind}_{dataset_type}_clip"
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "python", "-m", "src.train_clip",
        "--model_kind", model_kind,
        "--dataset_type", dataset_type,
        "--output_directory", output_dir, 
        "--patience", "1000000",
        "--num_epochs", "20",
        "--save_every", "5"
    ]
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"Training failed for {model_kind}, {dataset_type}: {e}")
        continue
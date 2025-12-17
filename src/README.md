# Training Disentangled CLIP Model (`train_clip.py`)

This script trains a CLIP-style model with disentangled or finetuned losses on the FEIDEGGER dataset.

## Features

- Loads image and text encoders (CLIP vision and DistilBERT text).
- Supports both disentangled and finetuned contrastive loss.
- Saves model checkpoints and training configuration.
- Logs training progress to TensorBoard.

## Usage

```bash
python -m src.train_clip --csv_path <path_to_csv> --pretrained_dir <pretrained_models_dir> --output_directory <output_dir> --learning_rate 1e-4 --batch_size 16 --num_epochs 10 --model_kind disentangled
```

### Arguments

- `--csv_path`: Path to CSV file with image paths, text, and split info.
- `--pretrained_dir`: Directory containing pretrained CLIP and DistilBERT models.
- `--output_directory`: Directory to save trained models and logs.
- `--learning_rate`: Learning rate for optimizer.
- `--batch_size`: Training batch size.
- `--num_epochs`: Number of training epochs.
- `--model_kind`: Model type (`disentangled` or `finetuned`).
- `--save_every`: Save model every N epochs.

## Output

- Model checkpoints in the output directory.
- TensorBoard logs in `output_dir/tensorboard_logs`.
- Training configuration in `output_dir/training_config.json`.

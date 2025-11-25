import pandas as pd
from PIL import Image
from sentence_transformers import util
from torch.utils.data import DataLoader
from sentence_transformers import datasets
from torch.utils.tensorboard import SummaryWriter
from sentence_transformers import SentenceTransformer, losses, InputExample


# how to finetune a CLIP model with text-image pairs
# https://github.com/huggingface/sentence-transformers/blob/1ec4902ddb73a8f33cb5ac2ae0b8f77a930b14d6/examples/training/clip/train_clip.ipynb
# https://github.com/ShawhinT/YouTube-Blog/blob/main/multimodal-ai/4-ft-mm-embeddings/2-finetune_clip_sbert.ipynb

# Load CSV
csv_path = "visualization_explorer/feidegger_visualization_data_valid.csv"
df = pd.read_csv(csv_path)


# finetuning? 
#https://huggingface.co/transformers/v3.4.0/custom_datasets.html

# pre-processing 
#https://huggingface.co/docs/transformers/main/en/preprocessing



# might want to use this nstead 
#https://www.kaggle.com/code/gouthamraj511/clip-from-scratch

# Build InputExample list
train_examples = []
test_examples = []
val_examples = []
for _, row in df.iterrows():
    img_path = row["image_path"]
    text = row["text"] 
    if row["mamba_split"] == "train":   
        train_examples.append(InputExample(texts=[img_path, text], label =1))  
    elif row["mamba_split"] == "test":
        test_examples.append(InputExample(texts=[img_path, text]))
    elif row["mamba_split"] == "val":
        val_examples.append(InputExample(texts=[img_path, text]))


print(f"Number of training examples: {len(train_examples)}")
model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')
num_epochs = 10  # Set as needed
# writer = SummaryWriter(log_dir="output/tensorboard_logs")

# from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

# val_texts = [ex.texts[1] for ex in val_examples]
# val_images = [ex.texts[0] for ex in val_examples]
# val_scores = [1.0] * len(val_examples)  # All pairs are matching

# evaluator = EmbeddingSimilarityEvaluator(val_images, val_texts, val_scores)

# def tensorboard_callback(score, epoch, steps):
#     writer.add_scalar("Loss/train_epoch", score, epoch)
#     print(f"Epoch {epoch+1}: Loss={score}")

# train_dataloader = datasets.NoDuplicatesDataLoader(train_examples, batch_size=8)
# train_loss = losses.MultipleNegativesRankingLoss(model)
# first_batch = next(iter(train_dataloader))
# for i, example in enumerate(first_batch):
#     print(f"Example {i}:")
#     print("  Image path:", example.texts[0])
#     print("  Text:", example.texts[1])

# We'll create a DataLoader that batches our data and prepare a contrastive loss function

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=4)
train_loss = losses.ContrastiveLoss(model=model)
model.to('cuda')
# Now we can fine-tune our model on the labeled image pairs
model.fit([(train_dataloader, train_loss)], epochs=num_epochs, show_progress_bar=True)
# After fine-tuning
model.save("output/finetuned-clip-multilingual")
# train_loss = losses.MultipleNegativesRankingLoss(model)
# warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)
# model = model.to('cuda')
# model.fit(
#     train_objectives=[(train_dataloader, train_loss)],
#     epochs=num_epochs,
#     # evaluator=evaluator,
#     warmup_steps=warmup_steps,
#     output_path="output/finetuned-clip-multilingual",
#     # callback=tensorboard_callback,
#     # evaluation_steps=3

# )

# writer.close()


from german_retrieval_custom import retrieve_images_by_image, retrieve_images_by_text, plot_images, compute_image_embeddings, compute_text_embeddings

# img_model = SentenceTransformer('clip-ViT-B-32')
# text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')

# Compute or load embeddings
import os 
test_df = df[df["mamba_split"] == "test"]

test_image_paths = test_df['image_path'].tolist()
test_texts = test_df['text'].tolist()

IMG_EMB_PATH = "data/test_img_embeddings.npy"
TXT_EMB_PATH = "data/test_txt_embeddings.npy"

print("Number of test images:", len(test_image_paths))
image_embeddings = compute_image_embeddings(model, test_image_paths, IMG_EMB_PATH)
text_embeddings = compute_text_embeddings(model, test_texts, TXT_EMB_PATH)
print("text_embeddings shape:", text_embeddings.shape)
print("image_embeddings shape:", image_embeddings.shape)
query = "A dress that whispers blood."
print("Text-to-Image Retrieval Example:")
results = retrieve_images_by_text(query, model, image_embeddings, test_image_paths, top_k=5)
plot_images(results, "Text-to-Image Retrieval (M-CLIP)", query=query, query_type="text")

print("\nImage-to-Image Retrieval Example:")
example_image = results[0][0]
print(f"Using example image: {example_image}")
results = retrieve_images_by_image(example_image, model, image_embeddings, test_image_paths, top_k=5)
plot_images(results, "Image-to-Image Retrieval (M-CLIP)", query=example_image, query_type="image")